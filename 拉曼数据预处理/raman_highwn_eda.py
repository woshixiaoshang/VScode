"""
高波数 EDA：三组 2850/2930 cm⁻¹ 峰面积比对比
柱状图（均值+误差棒）+ 显著性检验（Tukey HSD）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from itertools import combinations

# ─────────────────────────────────────────────
# 0. 配置：路径 / 颜色 / 输出
# ─────────────────────────────────────────────
DATA_PATHS = {
    "0h RAW":   r"D:\aaaSCNU\0data\Raman\EDA\0hRAW\3000\peak_area_ratio_2850_2930_0h.csv",
    "24h Foam": r"D:\aaaSCNU\0data\Raman\EDA\24hfoam\3000\peak_area_ratio_2850_2930_24h.csv",
    "48h Foam": r"D:\aaaSCNU\0data\Raman\EDA\48hfoam\3000\peak_area_ratio_2850_2930_48h.csv",
}

COLORS  = ["#378ADD", "#1D9E75", "#D85A30"]
OUTPUT  = r"D:\aaaSCNU\0data\Raman\EDA\eda_results"
os.makedirs(OUTPUT, exist_ok=True)

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# ─────────────────────────────────────────────
# 1. 读取数据
# ─────────────────────────────────────────────
groups = {}
for name, path in DATA_PATHS.items():
    df = pd.read_csv(path, encoding="utf-8-sig")   # 处理 BOM
    groups[name] = df["ratio_2850_2930"].dropna().values
    print(f"{name:10s}  n={len(groups[name])}  "
          f"mean={groups[name].mean():.4f}  std={groups[name].std():.4f}")

labels = list(groups.keys())
data   = [groups[k] for k in labels]

# ─────────────────────────────────────────────
# 2. 统计检验
# ─────────────────────────────────────────────
# 2a. 单因素 ANOVA
F_stat, p_anova = stats.f_oneway(*data)
print(f"\nOne-way ANOVA:  F={F_stat:.3f},  p={p_anova:.2e}")

# 2b. 两两 Welch t-test + Bonferroni 校正 + Cohen's d 效应量
pairs     = list(combinations(range(len(labels)), 2))
n_pairs   = len(pairs)
pair_results = []
for i, j in pairs:
    d1, d2   = data[i], data[j]
    t, p_raw = stats.ttest_ind(d1, d2, equal_var=False)
    p_adj    = min(p_raw * n_pairs, 1.0)          # Bonferroni
    # Cohen's d（pooled SD）
    sp       = np.sqrt((d1.std()**2 + d2.std()**2) / 2)
    cohens_d = abs(d1.mean() - d2.mean()) / sp
    d_interp = ("negligible" if cohens_d < 0.2 else
                "small"      if cohens_d < 0.5 else
                "medium"     if cohens_d < 0.8 else "large")
    pair_results.append({
        "pair":      f"{labels[i]} vs {labels[j]}",
        "mean_diff": d1.mean() - d2.mean(),
        "t":         t,
        "p_raw":     p_raw,
        "p_adj":     p_adj,
        "cohens_d":  cohens_d,
        "d_interp":  d_interp,
    })
    print(f"  {labels[i]:10s} vs {labels[j]:10s}: "
          f"Δmean={d1.mean()-d2.mean():+.5f}  t={t:7.3f}  "
          f"p_Bonf={p_adj:.2e}  Cohen's d={cohens_d:.3f} ({d_interp})")

def sig_label(p, d):
    """同时考虑统计显著性和效应量，避免大样本虚假显著"""
    if p >= 0.05:              return "ns"
    if d < 0.2:                return "† (d<0.2)"   # 统计显著但效应可忽略
    if   p < 0.001:            return "***"
    elif p < 0.01:             return "**"
    else:                      return "*"

# ─────────────────────────────────────────────
# 3. 图1：柱状图 + 误差棒 + 显著性标注
# ─────────────────────────────────────────────
means = [d.mean() for d in data]
sems  = [d.std() / np.sqrt(len(d)) for d in data]   # SEM 作误差棒

fig, ax = plt.subplots(figsize=(7, 6))

x = np.arange(len(labels))
bars = ax.bar(x, means, yerr=sems, width=0.52,
              color=COLORS, alpha=0.85, capsize=6,
              error_kw={"lw": 1.5, "ecolor": "gray"}, zorder=3)

# 散点（jitter）叠加，展示数据分布
rng = np.random.default_rng(42)
for xi, d, c in zip(x, data, COLORS):
    jitter = rng.uniform(-0.18, 0.18, size=min(len(d), 400))
    sample = d if len(d) <= 400 else rng.choice(d, 400, replace=False)
    ax.scatter(xi + jitter, sample, color=c,
               alpha=0.18, s=6, zorder=2, linewidths=0)

# 显著性括号
y_max  = max(d.max() for d in data)
y_step = (y_max - min(d.min() for d in data)) * 0.07
bracket_y = y_max + y_step * 0.8

def draw_bracket(ax, x1, x2, y, label):
    h = y_step * 0.25
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="black")
    ax.text((x1 + x2) / 2, y + h + y_step * 0.05, label,
            ha="center", va="bottom", fontsize=11)

offsets = [0, 1, 2]   # 括号高度层叠
for idx, (res, (i, j)) in enumerate(zip(pair_results, pairs)):
    lbl = sig_label(res["p_adj"], res["cohens_d"])
    draw_bracket(ax, x[i], x[j],
                 bracket_y + offsets[idx] * y_step * 1.5, lbl)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Peak area ratio  (2850 / 2930 cm⁻¹)", fontsize=11)
ax.set_title("CH₂/CH₃ peak area ratio across three groups\n"
             f"One-way ANOVA  F={F_stat:.1f},  p={p_anova:.1e}",
             fontsize=12, pad=12)

# 图例：显著性说明
legend_text = ("* p<0.05   ** p<0.01   *** p<0.001   ns: not significant\n"
               "† statistically significant but effect negligible (Cohen's d<0.2)\n"
               "(Welch t-test, Bonferroni corrected)")
ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.5,
        color="gray", style="italic")

plt.tight_layout()
out1 = os.path.join(OUTPUT, "highwn_bar_significance.png")
plt.savefig(out1)
plt.close()
print(f"\n已保存: {out1}")

# ─────────────────────────────────────────────
# 4. 图2：箱线图（补充，审稿人常要求）
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                medianprops={"color": "white", "lw": 2},
                whiskerprops={"lw": 1.2},
                capprops={"lw": 1.2},
                flierprops={"marker": "o", "markersize": 2,
                            "alpha": 0.3, "linestyle": "none"})
for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Peak area ratio  (2850 / 2930 cm⁻¹)", fontsize=11)
ax.set_title("CH₂/CH₃ peak area ratio — distribution", fontsize=12)

plt.tight_layout()
out2 = os.path.join(OUTPUT, "highwn_boxplot.png")
plt.savefig(out2)
plt.close()
print(f"已保存: {out2}")

# ─────────────────────────────────────────────
# 5. 统计摘要 CSV
# ─────────────────────────────────────────────
summary_rows = []
for name, d in groups.items():
    summary_rows.append({
        "group":  name,
        "n":      len(d),
        "mean":   round(d.mean(), 6),
        "median": round(np.median(d), 6),
        "std":    round(d.std(), 6),
        "sem":    round(d.std() / np.sqrt(len(d)), 6),
        "cv_%":   round(d.std() / d.mean() * 100, 2),
    })
summary_df = pd.DataFrame(summary_rows)
out3 = os.path.join(OUTPUT, "highwn_group_stats.csv")
summary_df.to_csv(out3, index=False)

# 效应量摘要
effect_df = pd.DataFrame([{
    "pair":      r["pair"],
    "mean_diff": round(r["mean_diff"], 6),
    "p_bonf":    f"{r['p_adj']:.2e}",
    "cohens_d":  round(r["cohens_d"], 4),
    "effect":    r["d_interp"],
} for r in pair_results])
out4 = os.path.join(OUTPUT, "highwn_effect_size.csv")
effect_df.to_csv(out4, index=False)
print(f"已保存: {out4}")
print(effect_df.to_string(index=False))
print(f"已保存: {out3}")
print(summary_df.to_string(index=False))

print("\n✓ 完成")
