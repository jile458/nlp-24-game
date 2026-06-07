import json
import os
from datasets import load_dataset

def prepare_data():
    os.makedirs("data", exist_ok=True)
    
    print("⏳ 正在拉取训练集 (如卡住请在终端执行 export HF_ENDPOINT=https://hf-mirror.com)...")
    
    # 1. 处理训练集: nlile/24-game
    # 包含了全部组合，特征通常包含 'puzzle' 和 'solvable'
    train_ds = load_dataset("nlile/24-game", split="train")
    
    train_samples = []
    false_count = 0
    
    for row in train_ds:
        # 兼容不同的列名格式
        puzzle_str = str(row.get('puzzle', row.get('Puzzle', '')))
        solvable = row.get('solvable', row.get('Solvable', True))
        
        # 从字符串 "3 3 8 8" 中提取出整数列表 [3, 3, 8, 8]
        nums = [int(x) for x in puzzle_str.replace(',', ' ').split() if x.isdigit()]
        if len(nums) != 4:
            continue
            
        if solvable:
            train_samples.append({"target_nums": nums, "solvable": True})
        else:
            # 限制无解数据的数量在 100 条左右，用于防幻觉测试
            if false_count < 100: 
                train_samples.append({"target_nums": nums, "solvable": False})
                false_count += 1
                
    print(f"✅ 训练集处理完毕: 共 {len(train_samples)} 条 (包含 {false_count} 条 solvable=False 数据)")
    
    with open("data/train.jsonl", "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 2. 处理测试集: test-time-compute/game-of-24
    print("\n⏳ 正在处理测试集 (提取 indices 900-1000 作为 OOD 难题)...")
    try:
        # HF datasets 支持直接用切片语法读取特定范围的数据
        test_ds = load_dataset("test-time-compute/game-of-24", split="train[900:1000]")
        test_samples = []
        
        for row in test_ds:
            puzzle_str = str(row.get('Puzzles', row.get('puzzle', '')))
            nums = [int(x) for x in puzzle_str.replace(',', ' ').split() if x.isdigit()]
            if len(nums) == 4:
                test_samples.append({"target_nums": nums, "solvable": True})
                
        with open("data/test.jsonl", "w", encoding="utf-8") as f:
            for sample in test_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"✅ 测试集准备完毕: 共 {len(test_samples)} 条")
        
    except Exception as e:
        print(f"⚠️ 测试集拉取失败，请检查网络或列名匹配。报错: {e}")

if __name__ == "__main__":
    prepare_data()