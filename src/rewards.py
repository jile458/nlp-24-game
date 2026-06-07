import re
import math
import os
from datetime import datetime
from collections import deque

_log_step = 0
_history_correct = deque(maxlen=100)
_history_total = deque(maxlen=100)

# ================= 新增：日志缓冲池 =================
_csv_buffer = []
_log_buffer = []

METRICS_FILE = "training_metrics.csv"
if not os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        # 写入表头
        f.write("step,batch_accuracy,smoothed_accuracy\n")
# =========================================================

def extract_answer(completion: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def format_reward(completions, **kwargs) -> list[float]:
    rewards = []
    for comp in completions:
        if "<think>" in comp and "</think>" in comp and "<answer>" in comp and "</answer>" in comp:
            rewards.append(0.5) 
        else:
            rewards.append(-1.0) 
    return rewards

def correctness_reward(completions, target_nums, solvable, **kwargs) -> list[float]:
    global _log_step, _history_correct, _history_total
    global _csv_buffer, _log_buffer
    
    _log_step += 1
    
    rewards = []
    batch_correct = 0  
    batch_total = len(completions)
    
    for comp, nums, is_solvable in zip(completions, target_nums, solvable):
        ans = extract_answer(comp)
        
        if not is_solvable:
            if ans.strip().upper() == "UNSOLVABLE":
                rewards.append(2.0)
                batch_correct += 1
            else:
                rewards.append(-1.5)
            continue
            
        if not ans:
            rewards.append(-0.5)
            continue

        try:
            used_nums = [int(n) for n in re.findall(r'\d+', ans)]
            if sorted(used_nums) != sorted(nums):
                rewards.append(-0.5)
                continue
        except Exception:
             rewards.append(-0.5)
             continue

        try:
            clean_exp = re.sub(r'[^0-9+\-*/(). ]', '', ans)
            if clean_exp.strip() != ans.strip():
                rewards.append(-0.5)
                continue
            
            result = eval(clean_exp)
            if math.isclose(result, 24.0, abs_tol=1e-6):
                rewards.append(2.0)
                batch_correct += 1 
            else:
                rewards.append(0.0) 
        except ZeroDivisionError:
            rewards.append(-0.5)
        except Exception:
            rewards.append(-0.5)
            
    # --- 统计更新 ---
    _history_correct.append(batch_correct)
    _history_total.append(batch_total)
    
    recent_correct = sum(_history_correct)
    recent_total = sum(_history_total)
    smoothed_accuracy = (recent_correct / recent_total) * 100 if recent_total > 0 else 0
    batch_accuracy = (batch_correct / batch_total) * 100

    print(f"\n[Step {_log_step}] 局部正确率: {batch_accuracy:5.1f}% | 📈 总体趋势(近100次): {smoothed_accuracy:5.1f}% ({recent_correct}/{recent_total})")

    # ================= 【已修改】缓冲写入逻辑 =================
    
    # 1. 存入 CSV 缓存
    _csv_buffer.append(f"{_log_step},{batch_accuracy:.2f},{smoothed_accuracy:.2f}\n")
    
    # 2. 存入 Log 缓存
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_content = f"\n{'='*20} Step {_log_step} [{time_str}] | 局部: {batch_accuracy:.1f}% | 总体趋势: {smoothed_accuracy:.1f}% {'='*20}\n"
    for i, (comp, nums, r) in enumerate(zip(completions, target_nums, rewards)):
        log_content += f"\n--- [样本 {i+1}] 题目: {nums} | 本次得分 (Reward): {r} ---\n"
        compact_comp = re.sub(r'\n\s*\n', '\n', comp).strip()
        log_content += f"{compact_comp}\n"
    _log_buffer.append(log_content)

    # 3. 攒够 10 条后，一次性写入并清空缓冲
    if len(_csv_buffer) >= 10:
        try:
            with open(METRICS_FILE, "a", encoding="utf-8") as f:
                f.writelines(_csv_buffer)
            _csv_buffer.clear()
        except Exception as e:
            print(f"⚠️ 写入 CSV 失败: {e}")

        try:
            with open("train_log.txt", "a", encoding="utf-8") as f:
                f.writelines(_log_buffer)
            _log_buffer.clear()
        except Exception as e:
            pass
    # ============================================================
        
    return rewards