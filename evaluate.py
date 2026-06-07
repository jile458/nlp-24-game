import torch
import json
import re
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.prompts import SYSTEM_PROMPT, get_prompt
from src.rewards import extract_answer

def check_correctness(ans: str, target_nums: list[int]) -> bool:
    """复用训练时的验证逻辑，判断单个算式是否完全正确"""
    if not ans:
        return False
    
    # 1. 验证数字约束
    try:
        used_nums = [int(n) for n in re.findall(r'\d+', ans)]
        if sorted(used_nums) != sorted(target_nums):
            return False
    except Exception:
        return False

    # 2. 验证数学计算
    try:
        clean_exp = re.sub(r'[^0-9+\-*/(). ]', '', ans)
        if clean_exp.strip() != ans.strip():
            return False
        
        result = eval(clean_exp)
        # 允许 10^-6 浮点误差
        return math.isclose(result, 24.0, abs_tol=1e-6)
    except Exception:
        return False

def main():
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_model_path = "./outputs/final_model"
    test_data_path = "data/test.jsonl"
    
    print("⏳ 1. 正在初始化测试环境 (4-bit 量化加载)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    print("⏳ 2. 正在挂载强化学习 LoRA 权重...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.eval() # 切换到评估模式

    print("📚 3. 正在读取测试集...")
    test_cases = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_cases.append(json.loads(line))
                
    total_count = len(test_cases)
    correct_count = 0
    format_error_count = 0
    
    print(f"\n🚀 开始自动化评估 (共 {total_count} 题)...")
    print("由于在 6GB 显存下单批次生成，这可能需要一些时间，请耐心等待。\n")
    
    # 使用 tqdm 显示华丽的进度条
    for i, case in enumerate(tqdm(test_cases, desc="Evaluating")):
        target_nums = case["target_nums"]
        
        # 构建输入
        user_content = get_prompt(target_nums)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 生成回答
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256, 
                temperature=0.1,  # 极低温度，追求确定性最优解
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 提取答案并验证
        ans = extract_answer(response)
        
        if not ans:
            format_error_count += 1
            
        if check_correctness(ans, target_nums):
            correct_count += 1
            
        # 为了不让输出太枯燥，每 20 题打印一次当前解答
        if (i + 1) % 20 == 0:
            tqdm.write(f"\n[抽查] 题目: {target_nums}")
            tqdm.write(f"[抽查] 模型答案: {ans}")
            tqdm.write(f"[抽查] 判定: {'✅ 正确' if check_correctness(ans, target_nums) else '❌ 错误'}")
            
    # 统计与输出报告
    solved_rate = (correct_count / total_count) * 100
    format_error_rate = (format_error_count / total_count) * 100
    
    print("\n" + "="*50)
    print("📊 强化学习模型评估报告")
    print("="*50)
    print(f"总测试题数: {total_count}")
    print(f"准确解出题数: {correct_count}")
    print(f"未输出答案标签数: {format_error_count}")
    print("-" * 50)
    print(f"🌟 测试集准确率 (Solved Rate): {solved_rate:.2f}%")
    print(f"⚠️ 格式错误率: {format_error_rate:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()