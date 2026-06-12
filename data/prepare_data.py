import json
import os
import re
from typing import Any

from datasets import load_dataset


TRAIN_PATH = "data/train.jsonl"
TEST_PATH = "data/test.jsonl"
UNSOLVABLE_TEST_PATH = "data/unsolvable_test.jsonl"


def parse_nums(row: dict[str, Any], keys: tuple[str, ...]) -> list[int]:
    for key in keys:
        if key in row and row[key] is not None:
            return [int(x) for x in re.findall(r"\d+", str(row[key]))]
    return []


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def puzzle_key(nums: list[int]) -> tuple[int, ...]:
    return tuple(sorted(nums))


def get_first(row: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def parse_solved_rate(row: dict[str, Any]) -> float | None:
    value = get_first(
        row,
        (
            "solved_rate",
            "solve_rate",
            "success_rate",
            "solved rate",
            "Solved Rate",
            "solved",
        ),
    )
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_jsonl(path: str, samples: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def prepare_data():
    os.makedirs("data", exist_ok=True)

    print("1. Loading nlile/24-game for training and unsolvable checks...")
    nlile_ds = load_dataset("nlile/24-game", split="train")

    train_samples: list[dict[str, Any]] = []
    unsolvable_samples: list[dict[str, Any]] = []
    train_keys: set[tuple[int, ...]] = set()

    for row in nlile_ds:
        row = dict(row)
        nums = parse_nums(row, ("puzzle", "Puzzle", "numbers", "nums"))
        if len(nums) != 4:
            continue

        solvable = parse_bool(get_first(row, ("solvable", "Solvable"), True))
        sample = {"target_nums": nums, "solvable": solvable, "source": "nlile/24-game"}

        if solvable:
            train_samples.append(sample)
            train_keys.add(puzzle_key(nums))
        elif len(unsolvable_samples) < 100:
            train_samples.append(sample)
            train_keys.add(puzzle_key(nums))
            unsolvable_samples.append(sample)

    write_jsonl(TRAIN_PATH, train_samples)
    write_jsonl(UNSOLVABLE_TEST_PATH, unsolvable_samples)

    train_solvable_count = sum(1 for sample in train_samples if sample["solvable"])
    train_unsolvable_count = sum(1 for sample in train_samples if not sample["solvable"])

    print(
        f"   Wrote {TRAIN_PATH}: {len(train_samples)} training cases "
        f"({train_solvable_count} solvable, {train_unsolvable_count} unsolvable)."
    )
    print(f"   Wrote {UNSOLVABLE_TEST_PATH}: {len(unsolvable_samples)} unsolvable check cases.")
    if train_solvable_count != 1262:
        print(f"   Warning: expected 1262 solvable cases, got {train_solvable_count}.")

    print("\n2. Loading test-time-compute/game-of-24 for non-overlapping ToT hard tests...")
    tot_ds = load_dataset("test-time-compute/game-of-24", split="train")

    paper_hard_samples: list[dict[str, Any]] = []
    overlap_count = 0

    for index, row in enumerate(tot_ds):
        if not 900 <= index < 1000:
            continue

        row = dict(row)
        nums = parse_nums(row, ("Puzzles", "puzzle", "Puzzle", "numbers", "nums"))
        if len(nums) != 4:
            continue

        key = puzzle_key(nums)
        if key in train_keys:
            overlap_count += 1
            continue

        sample = {
            "target_nums": nums,
            "solvable": True,
            "source": "test-time-compute/game-of-24",
            "source_index": index,
        }
        solved_rate = parse_solved_rate(row)
        if solved_rate is not None:
            sample["solved_rate"] = solved_rate
        paper_hard_samples.append(sample)

    if paper_hard_samples and "solved_rate" in paper_hard_samples[0]:
        paper_hard_samples.sort(key=lambda sample: sample["solved_rate"])

    write_jsonl(TEST_PATH, paper_hard_samples)

    print(f"   Wrote {TEST_PATH}: {len(paper_hard_samples)} non-overlapping ToT hard cases.")
    print(f"   Removed {overlap_count} overlapping cases from indices 900-1000.")
    if paper_hard_samples and "solved_rate" in paper_hard_samples[0]:
        rates = [sample["solved_rate"] for sample in paper_hard_samples]
        print(f"   Solved-rate range: {min(rates):.4f} - {max(rates):.4f}.")
    else:
        print("   No solved_rate column found; kept paper hard split order.")


if __name__ == "__main__":
    prepare_data()
