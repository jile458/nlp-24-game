import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics():
    # 1. 读取 CSV 数据
    try:
        df = pd.read_csv("training_metrics.csv")
    except FileNotFoundError:
        print("❌ 未找到 training_metrics.csv，请先运行训练脚本！")
        return

    if len(df) == 0:
        print("⚠️ CSV 文件为空，还没开始记录数据。")
        return

    # 2. 设置画布
    plt.figure(figsize=(12, 6))
    
    # 3. 绘制两条曲线
    # 第一条：每一轮的局部正确率（设置透明度低一点，作为背景散点/细线）
    plt.plot(df['step'], df['batch_accuracy'], label='Batch Accuracy (Per Step)', 
             color='lightgray', alpha=0.5, linestyle='-', linewidth=1)
    
    # 第二条：每隔 100 轮的滑动平滑曲线（加粗，高亮显示）
    plt.plot(df['step'], df['smoothed_accuracy'], label='Smoothed Accuracy (Moving Avg 100)', 
             color='red', linewidth=2.5)

    # 4. 图表装饰
    plt.title('GRPO Training Accuracy Curve (24-Game)', fontsize=14, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(-5, 105) # 正确率范围 0-100
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    # 5. 保存图片
    output_file = "accuracy_curve.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 成功！曲线图已保存为当前目录下的: {output_file}")

if __name__ == "__main__":
    plot_metrics()