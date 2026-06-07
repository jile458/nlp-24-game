# 🎲 Qwen-24Game-GRPO: 基于强化学习的 24 点推导大模型微调

本项目旨在极限低显存（6GB VRAM）环境下，利用 **GRPO (Group Relative Policy Optimization)** 强化学习算法，对 `Qwen2.5-1.5B-Instruct` 进行可验证奖励（RLVR）微调，使其能够通过 `<think>...</think>` 树状推导过程，完美解决 24 点游戏。

注意这里显存配置预设很低，如果觉得结果不理想，可以自行修改相关内容
高显存提升效果建议:
1、修改 NUM_GENERATIONS,增大
2、修改 MAX_COMPLETION_LENGTH：从 384 提升至 1024 或 2048
3、关闭 4-bit 量化，使用原生 bfloat16：
修改方法： 在 train.py 中删掉（或注释掉） BitsAndBytesConfig 相关的代码。直接用 torch.bfloat16 加载模型。
4、提升 LORA_RANK：从 8 提升至 64 或 128
5、修改优化器：从 paged_adamw_8bit 改回标准的 adamw_torch
## 📂 项目结构

```text
24game-grpo/
├── data/
│   ├── prepare_data.py      # 数据获取与清洗脚本
│   ├── train.jsonl          # 训练集 (包含防幻觉测试数据)
│   └── test.jsonl           # 测试集 (game-of-24 难题)
├── src/
│   ├── prompts.py           # System Prompt 及对话模板
│   └── rewards.py           # 环境裁判系统 (正则提取、eval打分、日志记录)
├── outputs/                 # 训练过程中生成的 Checkpoints 和最终权重
├── train.py                 # GRPO 训练主程序 (核心入口)
├── evaluate.py              # 测试集自动化评估脚本 (计算正确率)
├── play_24.py               # 交互式测试脚本 (出题给模型做)
├── plot_curve.py            # 训练曲线可视化脚本
└── requirements.txt         # 依赖清单
```

## 🛠️ 环境准备 (推荐 WSL2/Ubuntu)

1. **创建并激活虚拟环境** (推荐使用 conda 或 venv)：
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **安装核心依赖**：
   ```bash
   pip install -r requirements.txt
   ```

3. **网络加速配置** (国内环境必备，防止 Hugging Face 连接超时)：
   ```bash
   export HF_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com)
   ```

## 🚀 快速开始

请严格按照以下顺序执行脚本：

### 第一步：准备数据集
从 Hugging Face 拉取 24 点原始数据并格式化。
```bash
python data/prepare_data.py
```
> **输出含义**：在 `data/` 下生成 `train.jsonl` 和 `test.jsonl`，这是模型训练的“课本”。

### 第二步：启动强化学习训练
启动带有 4-bit 量化和 LoRA 的极低显存微调。所有超参数均在 `train.py` 顶部，可根据显存情况自行调节。
```bash
python train.py
```
> **输出含义**：
> * `outputs/final_model/`: 训练完成后保存的 LoRA 权重。这是模型学会的“24点专属技能包”。
> * `train_log.txt`: 详细记录了模型每一轮的推导过程（如 `<think>` 内的挣扎和计算）。这是理解模型思维模式演变的“行车记录仪”。
> * `training_metrics.csv`: 记录了每个 Step 的正确率数值，用于后续画图。

### 第三步：绘制训练学习曲线 (训练中途亦可执行)
将 CSV 数据转化为可视化的折线图。
```bash
python plot_curve.py
```
> **输出含义**：生成 `accuracy_curve.png`。
> 图中的 **灰线 (Batch Accuracy)** 代表模型在单个批次（2道题）上的局部正确率，会有剧烈震荡；**红线 (Smoothed Accuracy)** 代表最近100次测验的滑动平均正确率，代表模型真实的学习上升趋势。

### 第四步：在测试集上评估
加载训练好的 LoRA 权重，在未见过的难题集上自动答题并统计准确率。
```bash
python evaluate.py
```
> **输出含义**：终端将打印最终的 OOD 测试集准确率（Solved Rate）和格式错误率。这是衡量项目成功与否的核心量化指标。

