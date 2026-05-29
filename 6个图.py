"""
拉曼光谱综合分析脚本
包含：重复性热图 / 强度热图 / 平均谱 / PCA聚类 / PCC分析 / 平均谱堆叠
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================================
# ✏️  修改这里：你的数据根目录（会自动递归读取所有子文件夹的 .txt）
# ============================================================
folder = r"D:\aaaSCNU\0data\Raman\00024h_foam_processed\1200\Data"

# ============================================================
# 可选：每组的标签（用于 PCA / 堆叠图）
# 若所有文件在同一文件夹下不分组，保持 groups = None 即可
# 若要分组，按子文件夹名称自动分组，设 AUTO_GROUP = True
# ============================================================
AUTO_GROUP = True   # True = 按子文件夹名自动分组；False = 所有文件视为一组

# ============================================================
# 全局绘图风格
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────
# 1. 读取数据
# ─────────────────────────────────────────────
spectra     = []
labels      = []   # 每条谱对应的组标签（子文件夹名 or "All"）
wavenumber  = None
expected_len = None

print(f"📂 扫描路径: {folder}\n")

for root_dir, dirs, files in os.walk(folder):
    group_name = os.path.basename(root_dir) if AUTO_GROUP else "All"
    for file in sorted(files):
        if not file.endswith(".txt"):
            continue
        path = os.path.join(root_dir, file)
        try:
            data = np.loadtxt(path)
            if data.ndim < 2 or data.shape[1] < 2:
                print(f"  ⚠️ 格式不符，跳过: {file}")
                continue
            x, y = data[:, 0], data[:, 1]
            if wavenumber is None:
                wavenumber    = x
                expected_len  = len(x)
            if len(x) != expected_len:
                print(f"  ⚠️ 点数不匹配，跳过: {file} ({len(x)} vs {expected_len})")
                continue
            spectra.append(y)
            labels.append(group_name)
            print(f"  ✅ {os.path.relpath(path, folder)}")
        except Exception as e:
            print(f"  ⚠️ 读取失败: {file} → {e}")

if len(spectra) == 0:
    raise RuntimeError("❌ 未读取到任何有效数据，请检查路径和文件格式！")

spectra = np.array(spectra)          # shape: (n_spectra, n_points)
labels  = np.array(labels)
n_spec, n_pts = spectra.shape
print(f"\n✅ 共读取 {n_spec} 条光谱，{n_pts} 个数据点")
print(f"   波数范围: {wavenumber.min():.1f} ~ {wavenumber.max():.1f} cm⁻¹")

unique_groups = list(dict.fromkeys(labels))   # 保持顺序
n_groups      = len(unique_groups)
group_colors  = plt.cm.tab10(np.linspace(0, 0.9, n_groups))
group_color_map = dict(zip(unique_groups, group_colors))

# 每组的均值谱
group_means = {g: spectra[labels == g].mean(axis=0) for g in unique_groups}

# ─────────────────────────────────────────────
# 自定义色图
# ─────────────────────────────────────────────
raman_cmap = LinearSegmentedColormap.from_list(
    "raman", ["#0d1b2a", "#1f4e79", "#2e86c1", "#27ae60", "#f4d03f", "#e74c3c", "#ffffff"]
)
cool_warm = LinearSegmentedColormap.from_list(
    "cool_warm", ["#2980b9", "#ecf0f1", "#c0392b"]
)

# ─────────────────────────────────────────────
# 2. 创建大画布（2行 × 3列）
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("#f7f9fc")

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    hspace=0.42,
    wspace=0.38,
    left=0.07, right=0.97,
    top=0.93, bottom=0.07
)

ax1 = fig.add_subplot(gs[0, 0])   # 重复性热图
ax2 = fig.add_subplot(gs[0, 1])   # 强度热图
ax3 = fig.add_subplot(gs[0, 2])   # 平均谱
ax4 = fig.add_subplot(gs[1, 0])   # PCA 聚类
ax5 = fig.add_subplot(gs[1, 1])   # PCC 分析
ax6 = fig.add_subplot(gs[1, 2])   # 平均谱堆叠

# ─────────────────────────────────────────────
# 图1 · 重复性热图（归一化后，展示各谱形状差异）
# ─────────────────────────────────────────────
spectra_norm = spectra / (spectra.max(axis=1, keepdims=True) + 1e-10)
im1 = ax1.imshow(
    spectra_norm[:, ::-1],  # <--- 重点1：水平翻转矩阵，让小波数排在左边
    aspect="auto",
    cmap=raman_cmap,
    extent=[wavenumber.min(), wavenumber.max(), n_spec, 0], # <--- 重点2：左边最小值，右边最大值
    interpolation="nearest"
)
ax1.set_xlabel("Raman Shift (cm⁻¹)")
ax1.set_ylabel("Spectrum Index")
ax1.set_title("Reproducibility Heatmap\n(Normalized)", fontsize=11, fontweight="bold")
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label("Norm. Intensity", fontsize=8)

# 如果有多组，在右侧加组标注色块
if n_groups > 1:
    ax1b = ax1.twinx()
    for i, lbl in enumerate(labels):
        ax1b.barh(i + 0.5, 1, color=group_color_map[lbl], alpha=0.6, height=1)
    ax1b.set_ylim(n_spec, 0)
    ax1b.set_xlim(0, 1)
    ax1b.set_yticks([])
    ax1b.set_xticks([])
    ax1b.spines[:].set_visible(False)

# ─────────────────────────────────────────────
# 图2 · 强度热图（原始强度，按均值排序）
# ─────────────────────────────────────────────
sort_idx      = np.argsort(spectra.mean(axis=1))
spectra_sorted = spectra[sort_idx]

im2 = ax2.imshow(
    spectra_sorted[:, ::-1],  # <--- 重点1：水平翻转矩阵
    aspect="auto",
    cmap="inferno",
    extent=[wavenumber.min(), wavenumber.max(), n_spec, 0], # <--- 重点2：左边最小值，右边最大值
    interpolation="nearest"
)
ax2.set_xlabel("Raman Shift (cm⁻¹)")
ax2.set_ylabel("Spectrum (sorted by mean intensity)")
ax2.set_title("Intensity Heatmap\n(Raw, sorted)", fontsize=11, fontweight="bold")
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("Intensity (a.u.)", fontsize=8)

# ─────────────────────────────────────────────
# 图3 · 平均谱 ± 标准差
# ─────────────────────────────────────────────
mean_spec = spectra.mean(axis=0)
std_spec  = spectra.std(axis=0)

ax3.plot(wavenumber, mean_spec, color="#2c3e50", lw=1.5, label="Mean")
ax3.fill_between(
    wavenumber,
    mean_spec - std_spec,
    mean_spec + std_spec,
    alpha=0.25, color="#2980b9", label="±1 SD"
)
ax3.set_xlabel("Raman Shift (cm⁻¹)")
ax3.set_ylabel("Intensity (a.u.)")
ax3.set_title("Mean Spectrum ± SD", fontsize=11, fontweight="bold")
ax3.set_xlim(wavenumber.min(), wavenumber.max())
ax3.legend(fontsize=8)

# ─────────────────────────────────────────────
# 图4 · PCA 聚类（PC1 vs PC2）
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(spectra)

pca = PCA(n_components=min(5, n_spec, n_pts))
scores = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_ * 100

for g in unique_groups:
    mask = labels == g
    ax4.scatter(
        scores[mask, 0], scores[mask, 1],
        label=g,
        color=group_color_map[g],
        s=60, alpha=0.8, edgecolors="white", linewidths=0.5
    )

# 添加95%置信椭圆（简单版：均值±2σ）
for g in unique_groups:
    mask = labels == g
    if mask.sum() >= 3:
        cx, cy = scores[mask, 0].mean(), scores[mask, 1].mean()
        sx, sy = scores[mask, 0].std() * 2, scores[mask, 1].std() * 2
        theta  = np.linspace(0, 2 * np.pi, 100)
        ax4.plot(
            cx + sx * np.cos(theta),
            cy + sy * np.sin(theta),
            color=group_color_map[g], lw=1, ls="--", alpha=0.5
        )

ax4.axhline(0, color="gray", lw=0.5, ls=":")
ax4.axvline(0, color="gray", lw=0.5, ls=":")
ax4.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
ax4.set_ylabel(f"PC2 ({explained[1]:.1f}%)" if len(explained) > 1 else "PC2")
ax4.set_title("PCA Clustering", fontsize=11, fontweight="bold")
if n_groups > 1:
    ax4.legend(fontsize=8, markerscale=0.9)

# ─────────────────────────────────────────────
# 图5 · PCC 分析（谱间皮尔逊相关系数矩阵）
# ─────────────────────────────────────────────
pcc_matrix = np.corrcoef(spectra)   # shape: (n_spec, n_spec)

im5 = ax5.imshow(pcc_matrix, cmap=cool_warm, vmin=-1, vmax=1,
                  aspect="auto", interpolation="nearest")
cbar5 = fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
cbar5.set_label("Pearson r", fontsize=8)

# 若谱数少，显示数值
if n_spec <= 20:
    for i in range(n_spec):
        for j in range(n_spec):
            ax5.text(j, i, f"{pcc_matrix[i,j]:.2f}",
                     ha="center", va="center", fontsize=6,
                     color="black" if abs(pcc_matrix[i,j]) < 0.7 else "white")

ax5.set_xlabel("Spectrum Index")
ax5.set_ylabel("Spectrum Index")
ax5.set_title("Pearson Correlation\nCoefficient Matrix", fontsize=11, fontweight="bold")

# 标注均值±std PCC（去掉自相关的对角线）
mask_off = ~np.eye(n_spec, dtype=bool)
pcc_vals = pcc_matrix[mask_off]
ax5.set_xlabel(
    f"Spectrum Index\nmean PCC = {pcc_vals.mean():.4f} ± {pcc_vals.std():.4f}",
    fontsize=8
)

# ─────────────────────────────────────────────
# 图6 · 平均谱堆叠（各组 or 全部谱竖向偏移）
# ─────────────────────────────────────────────
if n_groups > 1:
    # 按组显示均值谱，竖向堆叠
    offset_step = max(gm.max() - gm.min() for gm in group_means.values()) * 1.3
    for i, g in enumerate(unique_groups):
        gm = group_means[g]
        offset = i * offset_step
        ax6.plot(
            wavenumber, gm + offset,
            color=group_color_map[g], lw=1.5, label=g
        )
        ax6.fill_between(
            wavenumber,
            gm + offset - spectra[labels == g].std(axis=0),
            gm + offset + spectra[labels == g].std(axis=0),
            alpha=0.15, color=group_color_map[g]
        )
        ax6.text(
            wavenumber.max(), gm.mean() + offset,
            f" {g}", fontsize=7, va="center",
            color=group_color_map[g]
        )
    ax6.set_title("Stacked Mean Spectra\n(by Group)", fontsize=11, fontweight="bold")
    ax6.legend(fontsize=7, loc="upper left")
else:
    # 只有一组：堆叠所有单条谱
    max_range   = (spectra.max(axis=1) - spectra.min(axis=1)).max()
    offset_step = max_range * 1.1
    cmap_stack  = plt.cm.viridis(np.linspace(0, 1, n_spec))
    for i, sp in enumerate(spectra):
        ax6.plot(wavenumber, sp + i * offset_step,
                 color=cmap_stack[i], lw=0.8, alpha=0.85)
    ax6.set_title("Stacked Individual Spectra", fontsize=11, fontweight="bold")

ax6.set_xlabel("Raman Shift (cm⁻¹)")
ax6.set_ylabel("Intensity (offset, a.u.)")
ax6.set_xlim(wavenumber.min(), wavenumber.max())
ax6.set_yticks([])

# ─────────────────────────────────────────────
# 总标题
# ─────────────────────────────────────────────
fig.suptitle(
    f"Raman Spectra Analysis  ·  {n_spec} spectra  ·  "
    f"{wavenumber.min():.0f}–{wavenumber.max():.0f} cm⁻¹",
    fontsize=13, fontweight="bold", color="#1a252f", y=0.98
)

# ─────────────────────────────────────────────
# 保存
# ─────────────────────────────────────────────
save_svg = os.path.join(folder, "raman_analysis.svg")
save_png = os.path.join(folder, "raman_analysis.png")

fig.savefig(save_svg, dpi=300, bbox_inches="tight")
fig.savefig(save_png, dpi=300, bbox_inches="tight")

print(f"\n📸 图片已保存:")
print(f"   {save_svg}")
print(f"   {save_png}")

# ─────────────────────────────────────────────
# 同时保存 PCC 统计数据
# ─────────────────────────────────────────────
pcc_save = os.path.join(folder, "pcc_matrix.txt")
np.savetxt(pcc_save, pcc_matrix, fmt="%.6f",
           header=f"Pearson Correlation Coefficient Matrix ({n_spec}x{n_spec})")
print(f"   {pcc_save}")

# ─────────────────────────────────────────────
# PCA loadings（可选，方便后续精细分析）
# ─────────────────────────────────────────────
loadings_save = os.path.join(folder, "pca_loadings.txt")
loadings_header = "Wavenumber " + " ".join(
    [f"PC{i+1}({explained[i]:.1f}%)" for i in range(pca.n_components_)]
)
np.savetxt(
    loadings_save,
    np.column_stack([wavenumber, pca.components_.T]),
    header=loadings_header, fmt="%.6f"
)
print(f"   {loadings_save}")

plt.show()
print("\n✅ 分析完成！")