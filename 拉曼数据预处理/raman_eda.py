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
    "Mac":   r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\000_0hRaw_processed\1200",
    "Foam": r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\000_24hFoam_processed\1200",
    "Ra": r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\8hRa_processed\1200",
    "HP-CD": r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\8hHP-CD_processed\1200",
}

COLORS = {
    "Mac": "#378ADD", "Foam": "#1D9E75", "Ra": "#8E5DB7", "HP-CD": "#D85A30"
}

ALPHAS = {
    "Mac": 0.15, "Foam": 0.12, "Ra": 0.12, "HP-CD": 0.12
}
OUTPUT  = r"D:\aaaSCNU\0data\Raman\EDA\eda_results"

import os
os.makedirs(OUTPUT, exist_ok=True)


def save_csv_with_fallback(df, filename, output_dir=OUTPUT):
    """优先写入指定目录；若权限受限则回退到工作区 output 目录。"""
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, filename)
    try:
        df.to_csv(target_path, index=False)
        return target_path
    except PermissionError:
        fallback_dir = os.path.join(os.getcwd(), "output", "raman_eda_results")
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_path = os.path.join(fallback_dir, filename)
        df.to_csv(fallback_path, index=False)
        return fallback_path

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# 放大默认字体（整体放大约 2 倍）
try:
    scale = 2.0
    plt.rcParams['font.size'] = plt.rcParams.get('font.size', 10) * scale
    plt.rcParams['axes.titlesize'] = plt.rcParams.get('axes.titlesize', 12) * scale
    plt.rcParams['axes.labelsize'] = plt.rcParams.get('axes.labelsize', 11) * scale
    plt.rcParams['xtick.labelsize'] = plt.rcParams.get('xtick.labelsize', 10) * scale
    plt.rcParams['ytick.labelsize'] = plt.rcParams.get('ytick.labelsize', 10) * scale
    plt.rcParams['legend.fontsize'] = plt.rcParams.get('legend.fontsize', 10) * scale
except Exception:
    pass

# ─────────────────────────────────────────────
# 1. 读取数据
# ─────────────────────────────────────────────
def find_file(root_folder, filename):
    """在根目录下递归查找指定文件。
    支持目标文件不在同一层级，只要属于该类根目录即可。"""
    


    for dirpath, _, files in os.walk(root_folder):

        if filename in files:
            print("找到:", os.path.join(dirpath, filename))
            return os.path.join(dirpath, filename)

    def find_file(root_folder, filename):

        if os.path.isfile(root_folder) and \
            os.path.basename(root_folder) == filename:
            return root_folder

        for dirpath, _, files in os.walk(root_folder):
            if filename in files:
                return os.path.join(dirpath, filename)

        raise FileNotFoundError(
            f"未找到文件 {filename}，请检查路径: {root_folder}"
    )


def load_statistics(folder):
    path = find_file(folder, "processed_spectra_statistics.txt")
    df = pd.read_csv(path, sep='\t', comment='#',
                     names=["wavenumber", "mean", "std", "variance"])
    return df


def load_pca_loadings(folder):
    path = find_file(folder, "pca_loadings.txt")
    with open(path, encoding='utf-8') as f:
        header = f.readline().strip().lstrip('#').strip()
    cols = header.split()          # Wavenumber PC1(xx%) PC2...
    df = pd.read_csv(path, sep=r'\s+', comment='#',
                     names=cols, engine='python')
    return df


def load_spectra_matrix(folder):
    """读取光谱矩阵，行=波数，列=样本；转置为 样本×波数"""
    path = find_file(folder, "processed_spectra_matrix.txt")
    df = pd.read_csv(path, sep='\t', comment='#', header=None)
    wavenumbers = df.iloc[:, 0].values
    matrix = df.iloc[:, 1:].values.T   # shape: (n_samples, n_wavenumbers)
    return wavenumbers, matrix

def get_sample_count(folder):
    path = find_file(folder, "processed_spectra_matrix.txt")

    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        nrows=1,
        header=None
    )

    # 第一列是波数，所以减1
    return df.shape[1] - 1

print("读取数据中...")
stats_dict   = {k: load_statistics(v)   for k, v in DATA_DIRS.items()}
pca_dict     = {k: load_pca_loadings(v) for k, v in DATA_DIRS.items()}
sample_counts = {k: get_sample_count(v) for k, v in DATA_DIRS.items()}

# 以第一个类的波数为基准
wavenumbers = stats_dict["Mac"]["wavenumber"].values
trim_points = 20
trim_start = trim_points
trim_end = len(wavenumbers) - trim_points
wavenumbers_trimmed = wavenumbers[trim_start:trim_end]
print(f"  原始波数范围: {wavenumbers[0]:.1f} ~ {wavenumbers[-1]:.1f} cm⁻¹  ({len(wavenumbers)} 点)")
print(f"  裁剪后波数范围: {wavenumbers_trimmed[0]:.1f} ~ {wavenumbers_trimmed[-1]:.1f} cm⁻¹  ({len(wavenumbers_trimmed)} 点)")
for k, df in stats_dict.items():
    print(f"  {k}: mean 范围 [{df['mean'].min():.3f}, {df['mean'].max():.3f}]")

# ─────────────────────────────────────────────
# 1.5. 输出汇总统计和绘图数据
# ─────────────────────────────────────────────
print("\n导出组统计汇总和绘图数据...")
summary_rows = []
for group_name, df in stats_dict.items():
    summary_rows.append(pd.DataFrame({
        "group": group_name,
        "wavenumber_cm-1": df["wavenumber"].values,
        "mean": df["mean"].values,
        "std": df["std"].values,
        "variance": df["variance"].values,
    }))

summary_df = pd.concat(summary_rows, ignore_index=True)
summary_df["cv_percent"] = np.where(
    np.abs(summary_df["mean"]) > 1e-12,
    summary_df["std"] / np.abs(summary_df["mean"]) * 100,
    np.nan,
)
summary_df = summary_df[["group", "wavenumber_cm-1", "mean", "std", "variance", "cv_percent"]]
summary_csv = save_csv_with_fallback(summary_df, "group_statistics_summary.csv")
print(f"  已保存 {summary_csv}")

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
ax.set_ylabel("Normalized intensity", fontsize=13)
ax.set_title("Mean Raman spectra (±1 SD) — four classes", fontsize=15, fontweight='500')
ax.legend(fontsize=12, framealpha=0.3)
ax.set_xlim(wavenumbers_trimmed[0], 1800)

# 差值曲线（下图）
ax2 = axes[1]
from itertools import combinations

from itertools import combinations

pairs = list(combinations(DATA_DIRS.keys(), 2))

for a, b in pairs:
    diff = stats_dict[a]["mean"].values - stats_dict[b]["mean"].values

    ax2.plot(
        wavenumbers,
        diff,
        lw=0.9,
        alpha=0.85,
        label=f"{a} - {b}"
    )

ax2.axhline(0, color="gray", lw=0.6, ls="--")
ax2.set_xlabel("Wavenumber (cm⁻¹)", fontsize=13)
ax2.set_ylabel("Δ intensity", fontsize=12)
ax2.legend(fontsize=11, framealpha=0.3)
ax2.set_xlim(wavenumbers_trimmed[0], 1800)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig1_mean_spectra.svg"))
plt.close()
print("  已保存 fig1_mean_spectra.svg")

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
n_list    = [sample_counts[k] for k in DATA_DIRS]
F_vals, p_vals = pointwise_anova_from_stats(stat_list, n_list)

sig_mask  = p_vals < 0.05    # Bonferroni 校正可改为 0.05/1015
print(f"  显著差异波数点 (p<0.05): {sig_mask.sum()} / {len(sig_mask)}")

means_matrix = np.array([stats_dict[k]["mean"].values for k in DATA_DIRS])
between_var  = np.var(means_matrix, axis=0)
within_mean_var = np.mean([stats_dict[k]["variance"].values for k in DATA_DIRS], axis=0)
ratio = between_var / (within_mean_var + 1e-10)

plot_data_df = pd.DataFrame({
    "wavenumber_cm-1": wavenumbers,
    "anova_F": F_vals,
    "anova_p": p_vals,
    "neg_log10_p": -np.log10(np.clip(p_vals, 1e-300, 1)),
    "sig_p_0.05": sig_mask.astype(int),
    "between_class_variance": between_var,
    "within_class_mean_variance": within_mean_var,
    "variance_ratio": ratio,
})
for group_name in DATA_DIRS:
    df = stats_dict[group_name]
    plot_data_df[f"{group_name}_mean"] = df["mean"].values
    plot_data_df[f"{group_name}_std"] = df["std"].values
    plot_data_df[f"{group_name}_variance"] = df["variance"].values
    plot_data_df[f"{group_name}_cv_percent"] = np.where(
        np.abs(df["mean"].values) > 1e-12,
        df["std"].values / np.abs(df["mean"].values) * 100,
        np.nan,
    )
plot_data_csv = save_csv_with_fallback(plot_data_df, "plotting_data_summary.csv")
print(f"  已保存 {plot_data_csv}")

fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                         gridspec_kw={"height_ratios": [2, 1, 0.5]})

# F值
ax = axes[0]
ax.plot(wavenumbers, F_vals, color="#378ADD", lw=0.8, alpha=0.85)
ax.set_ylabel("F statistic", fontsize=11)
ax.set_title("Pointwise one-way ANOVA across four classes", fontsize=13, fontweight='500')
ax.set_xlim(wavenumbers_trimmed[0], 1800)

# -log10(p)
ax2 = axes[1]
log_p = -np.log10(np.clip(p_vals, 1e-300, 1))
ax2.plot(wavenumbers, log_p, color="#D85A30", lw=0.8, alpha=0.85)
ax2.axhline(-np.log10(0.05), color="gray", lw=0.8, ls="--",
            label="p=0.05")
ax2.set_ylabel("−log₁₀(p)", fontsize=13)
ax2.legend(fontsize=11)
ax2.set_xlim(wavenumbers_trimmed[0], 1800)

# 显著性掩膜条带
ax3 = axes[2]
ax3.fill_between(wavenumbers, 0, sig_mask.astype(float),
                 color="#1D9E75", alpha=0.7, step='mid')
ax3.set_ylabel("Sig.", fontsize=12)
ax3.set_xlabel("Wavenumber (cm⁻¹)", fontsize=13)
ax3.set_ylim(0, 1.2)
ax3.set_xlim(wavenumbers_trimmed[0], 1800)
ax3.set_yticks([])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig2_anova.svg"))
plt.close()
print("  已保存 fig2_anova.svg")

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
    ax.set_ylabel(f"{pc_label} loading", fontsize=13)
    ax.legend(fontsize=11, framealpha=0.3)
    ax.set_xlim(wavenumbers_trimmed[0], 1800)

axes[1].set_xlabel("Wavenumber (cm⁻¹)", fontsize=13)
axes[0].set_title("PCA loadings comparison — PC1 & PC2", fontsize=15, fontweight='500')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig3_pca_loadings.svg"))
plt.close()
print("  已保存 fig3_pca_loadings.svg")

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

ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=13)
ax.set_ylabel("Variance", fontsize=13)
ax.set_title("Within-class variance vs between-class variance", fontsize=15, fontweight='500')
ax.legend(fontsize=11, framealpha=0.3)
ax.set_xlim(wavenumbers_trimmed[0], 1800)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig4_variance.svg"))
plt.close()
print("  已保存 fig4_variance.svg")

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

ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=13)
ax.set_ylabel("Normalized intensity", fontsize=13)
ax.set_title("Key discriminative wavenumbers (highlighted)", fontsize=15, fontweight='500')
ax.legend(fontsize=12, framealpha=0.3)
ax.set_xlim(wavenumbers_trimmed[0], 1800)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, "fig5_key_wavenumbers.svg"))
plt.close()
print("  已保存 fig5_key_wavenumbers.svg")

print("\n✓ EDA 完成！所有结果保存在:", OUTPUT)
print("  fig1_mean_spectra.svg   — 均值光谱 + 差分曲线")
print("  fig2_anova.svg          — 逐波数 ANOVA")
print("  fig3_pca_loadings.svg   — PCA 载荷对比")
print("  fig4_variance.svg       — 类内/类间方差")
print("  fig5_key_wavenumbers.svg — 关键波数高亮")
print("  key_wavenumbers.csv     — 差异显著波数列表")
