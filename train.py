import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

# 导入自定义模块
from src.prompts import SYSTEM_PROMPT, get_prompt
from src.rewards import format_reward, correctness_reward

# ==========================================================================================
# 🛠️ 训练超参数配置中心 (Hyperparameter Configuration)
# ------------------------------------------------------------------------------------------
# 提示：在进行消融实验或适配不同显卡时，请主要修改此处的参数。
# 当前默认配置专为 6GB 显存 (如 RTX 2060/3060/4050 Laptop) 极限微调设计。
# ==========================================================================================

# [1. 基础路径配置]
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 基座模型，如果不翻墙可配合 export HF_ENDPOINT=https://hf-mirror.com 使用
TRAIN_DATA_PATH = "data/train.jsonl"       # 由 prepare_data.py 生成的训练集路径
OUTPUT_DIR = "./outputs/final_model"       # 训练完毕后 LoRA 权重的保存位置

# [2. GRPO 与 强化学习核心参数]
# ------------------------------------------------------------------------------------------
NUM_GENERATIONS = 2          # 【极其吃显存】每次出题让模型并行生成的回答数。GRPO机制要求最少为2。6GB显存死守2，如有24GB显存可设为8或16。
KL_COEF = 0.05               # 【防止学傻的护城河】KL散度惩罚系数。模型如果输出乱码/标点符号，调大它(如0.1)；如果不愿意尝试新的推导格式，调小它(如0.01)。
MAX_PROMPT_LENGTH = 128      # 提示词最大截断长度。24点题目很短，128绝对够用。
MAX_COMPLETION_LENGTH = 384  # 【已修改】推理过程（<think>内部）最大长度。适当调大以防模型没算完就被截断扣分。

# [3. 学习率与批处理 (Optimizer & Batch)]
# ------------------------------------------------------------------------------------------
LEARNING_RATE = 5e-6         # 【必须极小】RL的试错过程极不稳定，学习率通常是 SFT 的十分之一。除非一直不收敛，否则不要轻易调大。
PER_DEVICE_BATCH_SIZE = 1    # 单卡物理 Batch Size，6GB显存只能设为 1。
GRAD_ACCUM_STEPS = 8         # 梯度累加步数。真实 Batch Size = 物理Batch * 累加步数。这里等效为8，保证梯度下降方向平稳。
LOGGING_STEPS = 10           # 每跑多少步在控制台打印一次 loss。

# [4. LoRA 适配器参数]
# ------------------------------------------------------------------------------------------
LORA_RANK = 8                # 【模型大脑增量】秩大小。推理任务 8 够用，如果发现怎么都学不会，可以尝试提至 16（会稍微增加显存）。
LORA_ALPHA = 16              # 缩放系数，业界惯例通常设置为 LORA_RANK 的 2 倍。

# ==========================================================================================
# 👇 以下为系统核心运行逻辑，非必要请勿修改 👇
# ==========================================================================================

def preprocess_dataset(example):
    """将 JSONL 中的目标数字转化为标准的对话 Prompt"""
    user_content = get_prompt(example["target_nums"])
    example["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    return example

def main():
    print(f"🤖 1. 正在加载 Tokenizer: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 【已修改】显式设置左侧 Padding，防止因果模型生成混乱
    tokenizer.padding_side = "left"

    print("🧠 2. 正在以 4-bit 量化加载基座模型 (极限省显存)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print(f"🔌 3. 正在注入 LoRA 适配器 (Rank={LORA_RANK})...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    print("📚 4. 正在准备和格式化训练数据...")
    dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
    dataset = dataset.map(preprocess_dataset)

    print("⚙️ 5. 正在注入 GRPO 训练参数...")
    training_args = GRPOConfig(
        output_dir="./outputs/checkpoints",
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        kl_coef=KL_COEF,
        gradient_checkpointing=True,   # 开启：用计算时间换显存
        bf16=True,                     # 混合精度
        optim="paged_adamw_8bit"       # 显存溢出时自动借用系统物理内存
    )

    print("🚀 6. 启动 GRPOTrainer，进入强化学习循环...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, correctness_reward], 
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer
    )
    
    trainer.train()
    
    print(f"💾 训练大功告成！正在保存 LoRA 权重到 {OUTPUT_DIR} ...")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()