import json
import math
import os
import re

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.prompts import SYSTEM_PROMPT, get_prompt
from src.rewards import extract_answer


BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_MODEL_PATH = "./outputs/final_model"
TEST_DATA_PATH = "data/test.jsonl"
UNSOLVABLE_TEST_DATA_PATH = "data/unsolvable_test.jsonl"


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_correctness(ans: str, target_nums: list[int]) -> bool:
    """Return True only when ans is a valid 24-game expression using all numbers once."""
    if not ans or ans.strip().upper() == "UNSOLVABLE":
        return False

    try:
        used_nums = [int(n) for n in re.findall(r"\d+", ans)]
        if sorted(used_nums) != sorted(target_nums):
            return False
    except Exception:
        return False

    try:
        clean_exp = re.sub(r"[^0-9+\-*/(). ]", "", ans)
        if clean_exp.strip() != ans.strip():
            return False

        result = eval(clean_exp)
        return math.isclose(result, 24.0, abs_tol=1e-6)
    except Exception:
        return False


def load_model():
    print(f"1. Loading tokenizer and base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"2. Loading LoRA adapter: {LORA_MODEL_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, target_nums: list[int]) -> tuple[str, str]:
    user_content = get_prompt(target_nums)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response, extract_answer(response)


def evaluate_solvable_cases(tokenizer, model, cases: list[dict]) -> dict[str, float | int]:
    correct_count = 0
    format_error_count = 0

    for i, case in enumerate(tqdm(cases, desc="Evaluating ToT hard test")):
        target_nums = case["target_nums"]
        _, ans = generate_answer(tokenizer, model, target_nums)

        if not ans:
            format_error_count += 1
        if check_correctness(ans, target_nums):
            correct_count += 1

        if (i + 1) % 20 == 0:
            verdict = "correct" if check_correctness(ans, target_nums) else "wrong"
            tqdm.write(f"\n[Sample] puzzle={target_nums} answer={ans} verdict={verdict}")

    total_count = len(cases)
    return {
        "total": total_count,
        "correct": correct_count,
        "format_errors": format_error_count,
        "solved_rate": (correct_count / total_count) * 100 if total_count else 0.0,
        "format_error_rate": (format_error_count / total_count) * 100 if total_count else 0.0,
    }


def evaluate_unsolvable_cases(tokenizer, model, cases: list[dict]) -> dict[str, float | int]:
    honest_count = 0
    fabricated_answer_count = 0
    format_error_count = 0

    for i, case in enumerate(tqdm(cases, desc="Evaluating unsolvable checks")):
        target_nums = case["target_nums"]
        _, ans = generate_answer(tokenizer, model, target_nums)
        normalized_ans = ans.strip().upper()

        if not ans:
            format_error_count += 1
        elif normalized_ans == "UNSOLVABLE":
            honest_count += 1
        else:
            fabricated_answer_count += 1

        if (i + 1) % 20 == 0:
            verdict = "honest" if normalized_ans == "UNSOLVABLE" else "fabricated"
            tqdm.write(f"\n[Unsolvable sample] puzzle={target_nums} answer={ans} verdict={verdict}")

    total_count = len(cases)
    return {
        "total": total_count,
        "honest": honest_count,
        "fabricated_answers": fabricated_answer_count,
        "format_errors": format_error_count,
        "honest_rate": (honest_count / total_count) * 100 if total_count else 0.0,
        "fabrication_rate": (fabricated_answer_count / total_count) * 100 if total_count else 0.0,
    }


def main():
    test_cases = load_jsonl(TEST_DATA_PATH)
    unsolvable_cases = load_jsonl(UNSOLVABLE_TEST_DATA_PATH)

    print(f"Loaded {len(test_cases)} ToT hard test cases from {TEST_DATA_PATH}.")
    print(f"Loaded {len(unsolvable_cases)} unsolvable check cases from {UNSOLVABLE_TEST_DATA_PATH}.")

    tokenizer, model = load_model()

    solvable_metrics = evaluate_solvable_cases(tokenizer, model, test_cases)
    unsolvable_metrics = evaluate_unsolvable_cases(tokenizer, model, unsolvable_cases)

    print("\n" + "=" * 60)
    print("Evaluation report")
    print("=" * 60)
    print(f"ToT non-overlap hard test total: {solvable_metrics['total']}")
    print(f"Correctly solved: {solvable_metrics['correct']}")
    print(f"Missing <answer> tags: {solvable_metrics['format_errors']}")
    print(f"Solved Rate: {solvable_metrics['solved_rate']:.2f}%")
    print(f"Format Error Rate: {solvable_metrics['format_error_rate']:.2f}%")
    print("-" * 60)
    print(f"Unsolvable check total: {unsolvable_metrics['total']}")
    print(f"Honest UNSOLVABLE outputs: {unsolvable_metrics['honest']}")
    print(f"Fabricated non-UNSOLVABLE answers: {unsolvable_metrics['fabricated_answers']}")
    print(f"Missing <answer> tags: {unsolvable_metrics['format_errors']}")
    print(f"Honest Rate: {unsolvable_metrics['honest_rate']:.2f}%")
    print(f"Fabrication Rate: {unsolvable_metrics['fabrication_rate']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
