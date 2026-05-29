"""
拉曼光谱三分类 EDA 分析
数据结构：每类一个文件夹，包含 processed_spectra_statistics.txt / pca_loadings.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. 配置：修改路径和类别名称
# ─────────────────────────────────────────────
DATA_DIRS = {
    "0hRAW":   r"D:\aaaSCNU\0data\Raman\EDA\0hRAW\1200",
    "24hFoam": r"D:\aaaSCNU\0data\Raman\EDA\24hfoam\1200",
    "48hFoam": r"D:\aaaSCNU\0data\Raman\EDA\48hfoam\1200",
}
SAMPLE_COUNTS = {"0hRAW": 1017, "24hFoam": 1342, "48hFoam": 1500}

COLORS  = {"0hRAW": "#378ADD", "24hFoam": "#1D9E75", "48hFoam": "#D85A30"}
ALPHAS  = {"0hRAW": 0.15,      "24hFoam": 0.12,      "48hFoam": 0.10}
OUTPUT  = r"D:\aaaSCNU\0data\Raman\EDA\eda_results"

import os
os.makedirs(OUTPUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ─────────────────────────────────────────────
# 1. 读取数据
# ─────────────────────────────────────────────
def load_statistics(folder):
    path = os.path.join(folder, "processed_spectra_statistics.txt")
    df = pd.read_csv(path, sep='\t', comment='#',
                     names=["wavenumber", "mean", "std", "variance"])
    return df

def load_pca_loadings(folder):
    path = os.path.join(folder, "pca_loadings.txt")
    with open(path) as f:
        header = f.readline().strip().lstrip('#').strip()
    cols = header.split()          # Wavenumber PC1(xx%) PC2...
    df = pd.read_csv(path, sep=r'\s+', comment='#',
                     names=cols, engine='python')
    return df

def load_spectra_matrix(folder):
    """读取光谱矩阵，行=波数，列=样本；转置为 样本×波数"""
    path = os.path.join(folder, "processed_spectra_matrix.txt")
    df = pd.read_csv(path, sep='\t', comment='#', header=None)
    wavenumbers = df.iloc[:, 0].values
    matrix = df.iloc[:, 1:].values.T   # shape: (n_samples, n_wavenumbers)
    return wavenumbers, matrix

print("读取数据中...")
stats_dict   = {k: load_statistics(v)   for k, v in DATA_DIRS.items()}
pca_dict     = {k: load_pca_loadings(v) for k, v in DATA_DIRS.items()}

# 以第一个类的波数为基准
wavenumbers = stats_dict["0hRAW"]["wavenumber"].values
print(f"  波数范围: {wavenumbers[0]:.1f} ~ {wavenumbers[-1]:.1f} cm⁻¹  ({len(wavenumbers)} 点)")
for k, df in stats_dict.items():
    print(f"  {k}: mean 范围 [{df['mean'].min():.3f}, {df['mean'].max():.3f}]")

# ─────────────────────────────────────────────
# 2. 图1：均值光谱 + 置信带
# ─────────────────────────────────────────────
print("\n绘制 图1：均值光谱对比...")
fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                         gridspec_kw={"height_ratios": [3, 1.2]})

ax = axes[0]
for name, df in stats_dict.items():
    wn  = df["wavenumber"].values
    mn  = df["mean"].values
    sd  = df["std"].values
    c   = COLORS[name]
    ax.plot(wn, mn, color=c, lw=1.2, label=name, zorder=3)
    ax.fill_between(wn, mn - sd, mn + sd, color=c,
                    alpha=ALPHAS[name], zorder=2)

ax.set_xlabel("")
ax.set_ylabel("Normalized intensity", fontsize=11)
ax.set_title("Mean Raman spectra (±1 SD) — three classes", fontsize=13, fontweight='500')
ax.legend(fontsize=10, framealpha=0.3)
ax.set_xlim(wavenumbers[0], wavenumbers[-1])

# 差值曲线（下图）
ax2 = axes[1]
pairs = [("0hRAW","24hFoam"), ("0hRAW","48hFoam"), ("24hFoam","48hFoam")]
pair_colors = ["#534AB7", "#993C1D", "#888780"]
for (a, b), col in zip(pairs, pair_colors):
    diff = stats_dict[a]["mean"].values - stats_dict[b]["mean"].values
    ax2.plot(wavenumbers, diff, color=col, lw=0.9,
             label=f"{a} − {b}", alpha=0.85)
ax2.axhline(0, color="gray", lw=0.6, ls="--")
ax2.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
ax2.set_ylabel("Δ intensity", fontsize=10)
ax2.legend(fontsize=9, framealpha=0.3)
ax2.set_xlim(wavenumbers[0], wavenumbers[-1])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig1_mean_spectra.png"))
plt.close()
print("  已保存 fig1_mean_spectra.png")

# ─────────────────────────────────────────────
# 3. 图2：逐波数 ANOVA F值 + 差异显著性掩膜
# ─────────────────────────────────────────────
print("计算逐波数 ANOVA（基于均值/方差/n估计）...")

def pointwise_anova_from_stats(stats_list, n_list):
    """
    用各组的 mean / std / n，在每个波数点做单因素 ANOVA。
    返回 F 值数组。（等效于完整样本计算，适用于已有统计量的情况）
    """
    k = len(stats_list)
    n_total = sum(n_list)
    n_wn = len(stats_list[0]["mean"])
    F_arr = np.zeros(n_wn)
    p_arr = np.zeros(n_wn)

    means  = np.array([df["mean"].values  for df in stats_list])   # (k, n_wn)
    vars_  = np.array([df["variance"].values for df in stats_list]) # (k, n_wn)
    ns     = np.array(n_list)

    grand_mean = np.average(means, axis=0, weights=ns)             # (n_wn,)

    SS_between = np.sum(ns[:, None] * (means - grand_mean[None, :]) ** 2, axis=0)
    df_between = k - 1

    SS_within = np.sum((ns[:, None] - 1) * vars_, axis=0)
    df_within = n_total - k

    MS_between = SS_between / df_between
    MS_within  = SS_within  / df_within

    with np.errstate(divide='ignore', invalid='ignore'):
        F_arr = np.where(MS_within > 0, MS_between / MS_within, 0)

    p_arr = stats.f.sf(F_arr, df_between, df_within)
    return F_arr, p_arr

stat_list = [stats_dict[k] for k in DATA_DIRS]
n_list    = [SAMPLE_COUNTS[k] for k in DATA_DIRS]
F_vals, p_vals = pointwise_anova_from_stats(stat_list, n_list)

sig_mask  = p_vals < 0.05    # Bonferroni 校正可改为 0.05/1015
print(f"  显著差异波数点 (p<0.05): {sig_mask.sum()} / {len(sig_mask)}")

fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                         gridspec_kw={"height_ratios": [2, 1, 0.5]})

# F值
ax = axes[0]
ax.plot(wavenumbers, F_vals, color="#378ADD", lw=0.8, alpha=0.85)
ax.set_ylabel("F statistic", fontsize=11)
ax.set_title("Pointwise one-way ANOVA across three classes", fontsize=13, fontweight='500')
ax.set_xlim(wavenumbers[0], wavenumbers[-1])

# -log10(p)
ax2 = axes[1]
log_p = -np.log10(np.clip(p_vals, 1e-300, 1))
ax2.plot(wavenumbers, log_p, color="#D85A30", lw=0.8, alpha=0.85)
ax2.axhline(-np.log10(0.05), color="gray", lw=0.8, ls="--",
            label="p=0.05")
ax2.set_ylabel("−log₁₀(p)", fontsize=11)
ax2.legend(fontsize=9)
ax2.set_xlim(wavenumbers[0], wavenumbers[-1])

# 显著性掩膜条带
ax3 = axes[2]
ax3.fill_between(wavenumbers, 0, sig_mask.astype(float),
                 color="#1D9E75", alpha=0.7, step='mid')
ax3.set_ylabel("Sig.", fontsize=10)
ax3.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
ax3.set_ylim(0, 1.2)
ax3.set_xlim(wavenumbers[0], wavenumbers[-1])
ax3.set_yticks([])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig2_anova.png"))
plt.close()
print("  已保存 fig2_anova.png")

# ─────────────────────────────────────────────
# 4. 图3：PCA 载荷对比
# ─────────────────────────────────────────────
print("绘制 图3：PCA 载荷对比...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

for pc_idx, (ax, pc_label) in enumerate(zip(axes, ["PC1", "PC2"])):
    for name, df in pca_dict.items():
        # 找PC列（列名可能含括号，用startswith匹配）
        pc_cols = [c for c in df.columns if c.startswith(pc_label)]
        if not pc_cols:
            continue
        pc_col = pc_cols[0]
        wn = df.iloc[:, 0].values
        ax.plot(wn, df[pc_col].values, color=COLORS[name],
                lw=1.0, label=f"{name}  {pc_col}", alpha=0.85)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_ylabel(f"{pc_label} loading", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.set_xlim(wavenumbers[0], wavenumbers[-1])

axes[1].set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
axes[0].set_title("PCA loadings comparison — PC1 & PC2", fontsize=13, fontweight='500')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig3_pca_loadings.png"))
plt.close()
print("  已保存 fig3_pca_loadings.png")

# ─────────────────────────────────────────────
# 5. 图4：类内方差 vs 类间方差
# ─────────────────────────────────────────────
print("绘制 图4：类内/类间方差...")
fig, ax = plt.subplots(figsize=(14, 5))

# 类内方差（每类各自的方差均值）
for name, df in stats_dict.items():
    ax.plot(df["wavenumber"], df["variance"],
            color=COLORS[name], lw=0.7, alpha=0.6, label=f"{name} (within)")

# 类间方差（均值的方差）
means_matrix = np.array([stats_dict[k]["mean"].values for k in DATA_DIRS])
between_var  = np.var(means_matrix, axis=0)
ax.plot(wavenumbers, between_var, color="black", lw=1.2,
        ls="--", label="Between-class variance", zorder=5)

ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
ax.set_ylabel("Variance", fontsize=11)
ax.set_title("Within-class variance vs between-class variance", fontsize=13, fontweight='500')
ax.legend(fontsize=9, framealpha=0.3)
ax.set_xlim(wavenumbers[0], wavenumbers[-1])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig4_variance.png"))
plt.close()
print("  已保存 fig4_variance.png")

# ─────────────────────────────────────────────
# 6. 输出：关键差异波数列表
# ─────────────────────────────────────────────
print("\n整理关键差异波数...")

# 方法1: ANOVA 显著 + 高 F 值（取 top 20%）
f_threshold = np.percentile(F_vals[sig_mask], 80) if sig_mask.sum() > 0 else 0
key_mask = sig_mask & (F_vals > f_threshold)

# 方法2: 类间方差显著高于类内平均方差
within_mean_var = np.mean([stats_dict[k]["variance"].values for k in DATA_DIRS], axis=0)
ratio = between_var / (within_mean_var + 1e-10)
ratio_mask = ratio > np.percentile(ratio, 80)

# 合并
combined_mask = key_mask & ratio_mask

key_df = pd.DataFrame({
    "wavenumber_cm-1": wavenumbers[combined_mask],
    "F_statistic":     np.round(F_vals[combined_mask], 2),
    "neg_log10_p":     np.round(-np.log10(np.clip(p_vals[combined_mask], 1e-300, 1)), 2),
    "between_var":     np.round(between_var[combined_mask], 6),
    "var_ratio":       np.round(ratio[combined_mask], 2),
})
key_df = key_df.sort_values("F_statistic", ascending=False).reset_index(drop=True)

out_csv = os.path.join(OUTPUT, "key_wavenumbers.csv")
key_df.to_csv(out_csv, index=False)
print(f"  关键差异波数: {len(key_df)} 个 → 已保存 key_wavenumbers.csv")
print(key_df.head(15).to_string())

# ─────────────────────────────────────────────
# 7. 图5：关键波数区间高亮（叠在均值光谱上）
# ─────────────────────────────────────────────
print("\n绘制 图5：关键波数高亮图...")
fig, ax = plt.subplots(figsize=(14, 6))

# 背景高亮
for wn_val in key_df["wavenumber_cm-1"].values:
    ax.axvspan(wn_val - 2, wn_val + 2, color="#FAC775", alpha=0.25, zorder=1)

for name, df in stats_dict.items():
    ax.plot(df["wavenumber"], df["mean"],
            color=COLORS[name], lw=1.2, label=name, zorder=3)

ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
ax.set_ylabel("Normalized intensity", fontsize=11)
ax.set_title("Key discriminative wavenumbers (highlighted)", fontsize=13, fontweight='500')
ax.legend(fontsize=10, framealpha=0.3)
ax.set_xlim(wavenumbers[0], wavenumbers[-1])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig5_key_wavenumbers.png"))
plt.close()
print("  已保存 fig5_key_wavenumbers.png")

print("\n✓ EDA 完成！所有结果保存在:", OUTPUT)
print("  fig1_mean_spectra.png   — 均值光谱 + 差分曲线")
print("  fig2_anova.png          — 逐波数 ANOVA")
print("  fig3_pca_loadings.png   — PCA 载荷对比")
print("  fig4_variance.png       — 类内/类间方差")
print("  fig5_key_wavenumbers.png — 关键波数高亮")
print("  key_wavenumbers.csv     — 差异显著波数列表")
