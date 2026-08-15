import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from adjustText import adjust_text  # 自动调整标签位置防止重叠 (需 pip install adjustText)
except ImportError:
    adjust_text = None

import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. 加载数据
# ─────────────────────────────────────────────
# 假设这是你之前生成的包含5个参数的 CSV 文件路径
data_path = r"D:\aaaSCNU\0data\Raman\EDA\eda_results\key_wavenumbers.csv"
OUTPUT_DIR = r"D:\aaaSCNU\0data\Raman\EDA\ppt_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(data_path)
    print("✅ 成功加载特征评价矩阵！")
except FileNotFoundError:
    print("⚠️ 未找到文件，生成模拟测试数据以展示绘图效果...")
    # 如果找不到文件，生成带有些许真实感的数据保证代码顺利跑通
    np.random.seed(42)
    wavenumbers = np.linspace(400, 1800, 500)
    neg_log10_p = np.random.exponential(scale=2, size=500)
    var_ratio = np.random.lognormal(mean=0, sigma=1, size=500)
    F_statistic = var_ratio * 10 + np.random.normal(0, 5, 500)
    between_var = np.random.uniform(0.01, 0.5, 500)
    df = pd.DataFrame({
        "wavenumber_cm-1": wavenumbers,
        "F_statistic": F_statistic,
        "neg_log10_p": neg_log10_p,
        "between_var": between_var,
        "var_ratio": var_ratio
    })

# ─────────────────────────────────────────────
# 2. 阈值设定与数据分类
# ─────────────────────────────────────────────
# 设定显著性阈值：p = 0.05 对应 -log10(p) ≈ 1.30
p_threshold = -np.log10(0.05)
# 设定效应量阈值：取 var_ratio 的 85% 分位数作为高变异筛选线
ratio_threshold = np.percentile(df["var_ratio"], 85)

# 给数据打上标签，用于设定颜色
def get_status(row):
    if row["neg_log10_p"] > p_threshold and row["var_ratio"] > ratio_threshold:
        return "Core Biomarkers (High Sig. & High Var. Ratio)"
    elif row["neg_log10_p"] > p_threshold:
        return "Statistically Significant Only"
    else:
        return "Non-significant"

df["Status"] = df.apply(get_status, axis=1)

# ─────────────────────────────────────────────
# 3. 绘制高级变体火山图
# ─────────────────────────────────────────────
print("🎨 正在生成变体火山图...")
plt.rcParams.update({
    "font.family": "Arial",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=(10, 7))

# 调色盘：突显核心标志物 (深红色)，显著组 (浅蓝色)，无意义组 (浅灰色)
color_dict = {
    "Core Biomarkers (High Sig. & High Var. Ratio)": "#D1603D", 
    "Statistically Significant Only": "#89B4C4", 
    "Non-significant": "#E0E0E0"
}

# 绘制散点，点的大小映射 F_statistic，使其具备 3D 信息量
sns.scatterplot(
    data=df, 
    x="var_ratio", 
    y="neg_log10_p", 
    hue="Status", 
    palette=color_dict,
    size="F_statistic", 
    sizes=(10, 150), 
    alpha=0.85, 
    edgecolor="black", 
    linewidth=0.3,
    ax=ax
)

# 绘制辅助阈值线
ax.axhline(p_threshold, color="#666666", linestyle="--", linewidth=1.2, zorder=0)
ax.axvline(ratio_threshold, color="#666666", linestyle="--", linewidth=1.2, zorder=0)

# 标注 Top 核心波数 (取右上角象限中 F_statistic 最高的 10 个点)
top_features = df[df["Status"] == "Core Biomarkers (High Sig. & High Var. Ratio)"].nlargest(10, "F_statistic")

texts = []
for idx, row in top_features.iterrows():
    # 标签格式保留一位小数，例如 702.1 cm⁻¹
    texts.append(ax.text(row["var_ratio"], row["neg_log10_p"], f"{row['wavenumber_cm-1']:.1f}", 
                         fontsize=9, fontweight="bold", color="#333333"))

# 使用 adjustText 自动弹开重叠的文字标签 (这一步让图看起来非常专业)
try:
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
except NameError:
    print("提示: 若标签重叠，建议 pip install adjustText 并重新运行。")

# 美化图表
ax.set_title("Variant Volcano Plot of SERS Mechanism Signatures", fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel("Variance Ratio (Effect Size)", fontsize=13, fontweight="bold")
ax.set_ylabel(r"$-\log_{10}(P\text{-value})$", fontsize=13, fontweight="bold")

# 定制图例
handles, labels = ax.get_legend_handles_labels()
# 去掉 Seaborn 默认带的 size 图例，只保留颜色分类
keep_idx = [labels.index(status) for status in color_dict.keys() if status in labels]
ax.legend([handles[i] for i in keep_idx], [labels[i] for i in keep_idx], 
          loc="upper right", framealpha=0.9, title="Feature Significance")

plt.tight_layout()

# 保存高清图
out_svg = os.path.join(OUTPUT_DIR, "ppt_variant_volcano.svg")
out_png = os.path.join(OUTPUT_DIR, "ppt_variant_volcano.png")
plt.savefig(out_svg, format="svg", transparent=True)
plt.savefig(out_png, format="png", dpi=400, transparent=True)
plt.close()

print(f"🎉 变体火山图已就绪！")
print(f"  👉 {out_svg}")