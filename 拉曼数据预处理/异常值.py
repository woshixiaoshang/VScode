"""
拉曼光谱异常谱剔除脚本
适用场景：多细胞群体代谢谱，剔除技术噪声（荧光污染/宇宙射线/焦平面漂移等）
          保留真实的细胞间生物学异质性

剔除策略（四重过滤，从严到宽可单独调阈值）：
  1. 宇宙射线检测   —— 单点强度尖峰
  2. 信噪比过低     —— 整体强度极弱或为噪声
  3. PCC 异常值     —— PCC 与群体均值谱相关性过低（迭代均值，防止坏谱污染基准）
  4. 孤立点检测     —— PCA 空间中的 3σ 孤立谱
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 解决 matplotlib 保存图像时中文乱码/方块问题
plt.switch_backend('agg')  # <--- 加上这一行！它的作用是强制纯后台画图
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
preferred_fonts = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial"]
for font_name in preferred_fonts:
    if font_name in available_fonts:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_name]
        break
else:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18

# ============================================================
# ✏️  读取 target 文件中的读取路径
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

candidate_names = ["target", "target.txt"]
target_file = None
for base_dir in [desktop_path, current_dir]:
    for name in candidate_names:
        candidate = os.path.join(base_dir, name)
        if os.path.isfile(candidate):
            target_file = candidate
            break
    if target_file:
        break

# 也允许桌面/脚本目录下存在类似的 target* 文件，避免扩展名差异问题
if target_file is None:
    for base_dir in [desktop_path, current_dir]:
        if os.path.isdir(base_dir):
            for fname in os.listdir(base_dir):
                if fname.lower().startswith("target"):
                    candidate = os.path.join(base_dir, fname)
                    if os.path.isfile(candidate):
                        target_file = candidate
                        break
        if target_file:
            break

folder = None
task_label = None
if target_file is not None:
    with open(target_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 查找"异常值"标签及其下一行
    for i, line in enumerate(lines):
        if "异常值" in line:
            task_label = line
            # 获取下一行作为读取路径
            if i + 1 < len(lines):
                folder = lines[i + 1]
            break
    
    # 若未找到"异常值"，则按旧格式处理
    if folder is None and len(lines) >= 2:
        task_label = lines[0]
        folder = lines[1]
    elif folder is None and len(lines) >= 1:
        folder = lines[0]
        task_label = "异常值剔除"

if folder is None:
    raise RuntimeError(
        "❌ 未找到 target 文件，或 target 文件内容不正确。\n"
        "预期格式（多行配置）：\n"
        "  预处理\n"
        "  /path/to/preprocess\n"
        "  异常值\n"
        "  /path/to/outlier\n"
        "或简化格式：\n"
        "  异常值\n"
        "  /path/to/data"
    )
print(f"🔎 已找到 target 文件: {target_file}")
print(f"   任务标签: {task_label}")

AUTO_GROUP = True   # True = 按子文件夹名分组；False = 全部一组

# 💾 输出文件夹：自动创建在读取文件夹的同级目录
parent_dir = os.path.dirname(folder)
folder_name = os.path.basename(folder)
output_folder = os.path.join(parent_dir, folder_name + "_cleaned")
os.makedirs(output_folder, exist_ok=True)
print(f"📁 读取路径：{folder}")
print(f"📁 输出文件夹：{output_folder}\n")

# ── 剔除阈值（可按需调整）──────────────────────────────────
# [1] 宇宙射线：单点强度超过相邻均值的倍数
COSMIC_SPIKE_RATIO   = 20.0    # 默认 8 倍，越小越严格

# [2] 信噪比：谱的最大值/基线噪声估计值
SNR_THRESHOLD        = 5    # 默认 5，越大越严格

# [3] PCC 阈值：与全体谱均值谱的皮尔逊相关系数
#     细胞群体建议 0.70~0.75（比均一样品宽松）
PCC_THRESHOLD        = 0.75   # 默认 0.70

# [4] PCA 孤立点：在 PC1-PC5 空间中，Mahalanobis 距离超过此 σ 数
PCA_SIGMA_THRESHOLD  = 3.5    # 默认 3.5σ
# ─────────────────────────────────────────────────────────

# ============================================================
# 1. 读取数据（复用之前的读取逻辑）
# ============================================================
spectra, labels, file_ids = [], [], []
wavenumber, expected_len = None, None

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
                continue
            x, y = data[:, 0], data[:, 1]
            if wavenumber is None:
                wavenumber, expected_len = x, len(x)
            if len(x) != expected_len:
                print(f"  ⚠️ 点数不符，跳过: {file}")
                continue
            spectra.append(y)
            labels.append(group_name)
            file_ids.append(os.path.relpath(path, folder))
        except Exception as e:
            print(f"  ⚠️ 读取失败: {file} → {e}")

if len(spectra) == 0:
    raise RuntimeError("❌ 未读取到任何数据！")

spectra   = np.array(spectra)
labels    = np.array(labels)
file_ids  = np.array(file_ids)
n_total   = len(spectra)
print(f"✅ 共读取 {n_total} 条光谱\n")

# ── 基线校正（在筛选之前，消除荧光背景对PCC的干扰）──────────
print("⚙️  预处理：对原始谱做基线校正（airPLS）...")
from scipy import sparse
from scipy.sparse.linalg import spsolve

def airPLS_baseline(spectra, lam=1e4, porder=0.005, itermax=8):  # itermax改成8
    m = spectra.shape[1]
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(m - 2, m))
    penalty = (lam * D.T.dot(D)).tocsc()
    corrected = []
    for signal in tqdm(spectra, desc="  基线校正进度"):
        w = np.ones(m)
        w[:10], w[-10:] = 1000, 1000
        for i in range(itermax):
            W = sparse.spdiags(w, 0, m, m).tocsc()
            baseline = spsolve(W + penalty, w * signal)
            diff = signal - baseline
            w_new = np.where(diff >= 0, porder,
                             np.exp(i * diff / (np.sum(diff[diff < 0]) + 1e-12)))
            w_new[w_new < 1e-8] = 1e-8
            if np.linalg.norm(w - w_new) / (np.linalg.norm(w) + 1e-12) < 1e-3:
                break
            w = w_new
        corrected.append(signal - baseline)
    return np.array(corrected)

spectra = airPLS_baseline(spectra)
print(f"   → 基线校正完成，进入筛选流程")

# ============================================================
# 2. 四重过滤
# ============================================================
reject_flags  = np.zeros(n_total, dtype=bool)   # True = 剔除
reject_reason = np.full(n_total, "", dtype=object)

def mark(idx_arr, reason):
    """标记剔除并记录原因（同一条谱可能触发多个，记录第一个）"""
    for i in idx_arr:
        if not reject_flags[i]:
            reject_reason[i] = reason
        reject_flags[i] = True

# ── Filter 1：宇宙射线（单点尖峰检测）─────────────────────
print("🔍 [1/4] 宇宙射线检测...")
cosmic_idx = []
for i, sp in enumerate(spectra):
    # 二阶差分检测真正的单点尖峰，对zap填平后的谱更鲁棒
    d2 = np.abs(np.diff(sp, n=2))
    noise_level = np.median(d2) / 0.6745  # 稳健噪声估计
    if np.max(d2) > COSMIC_SPIKE_RATIO * noise_level:
        cosmic_idx.append(i)
mark(cosmic_idx, f"宇宙射线(spike>{COSMIC_SPIKE_RATIO}x)")
print(f"   → 发现 {len(cosmic_idx)} 条含宇宙射线")

# ── Filter 2：信噪比过低 ───────────────────────────────────
print("🔍 [2/4] 信噪比检测...")
snr_idx = []
for i, sp in enumerate(spectra):
    signal    = sp.max() - sp.min()
    # 用首尾各10%区域估计基线噪声
    n10       = max(int(len(sp)*0.1), 3)
    noise_est = np.std(np.concatenate([sp[:n10], sp[-n10:]]))
    snr       = signal / (noise_est + 1e-10)
    if snr < SNR_THRESHOLD:
        snr_idx.append(i)
mark(snr_idx, f"SNR过低(<{SNR_THRESHOLD})")
print(f"   → 发现 {len(snr_idx)} 条低信噪比谱")

# ── Filter 3：PCC 与群体均值谱相关性过低（迭代均值，防止坏谱污染基准）──
print("🔍 [3/4] PCC 相关性检测（迭代均值基准）...")

# 第一轮：用全部谱的均值做初始基准
mean_spec = spectra.mean(axis=0)
pcc_vals  = np.array([np.corrcoef(sp, mean_spec)[0, 1] for sp in spectra])

# 第二轮：用第一轮筛出的"初步干净谱"重新算均值，得到更干净的基准
clean_mask_iter = pcc_vals >= PCC_THRESHOLD
if clean_mask_iter.sum() >= 5:  # 至少有5条谱才做第二轮
    mean_spec_clean = spectra[clean_mask_iter].mean(axis=0)
    pcc_vals = np.array([np.corrcoef(sp, mean_spec_clean)[0, 1] for sp in spectra])
    print(f"   → 迭代完成，第二轮基准使用 {clean_mask_iter.sum()} 条初步干净谱")
else:
    print(f"   → 初步干净谱不足5条，跳过第二轮迭代，使用全局均值基准")

pcc_idx = np.where(pcc_vals < PCC_THRESHOLD)[0].tolist()
mark(pcc_idx, f"PCC过低(<{PCC_THRESHOLD})")
print(f"   → PCC 均值={pcc_vals.mean():.4f}，标准差={pcc_vals.std():.4f}")
print(f"   → 发现 {len(pcc_idx)} 条低PCC谱（阈值 {PCC_THRESHOLD}）")

# ── Filter 4：PCA Mahalanobis 距离孤立点 ──────────────────
print("🔍 [4/4] PCA 空间孤立点检测...")
n_pc = min(10, n_total - 1, spectra.shape[1])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(spectra)
pca = PCA(n_components=n_pc)
scores = pca.fit_transform(X_scaled)   # (n, n_pc)

# 用前5个PC计算 Mahalanobis 距离（简化版：Z-score 的 RMS）
scores5 = scores[:, :5]
mu    = scores5.mean(axis=0)
sigma = scores5.std(axis=0) + 1e-10
z     = (scores5 - mu) / sigma
mahal = np.sqrt((z**2).mean(axis=1))   # RMS of Z-scores

pca_threshold = PCA_SIGMA_THRESHOLD
pca_idx = np.where(mahal > pca_threshold)[0].tolist()
mark(pca_idx, f"PCA孤立点(>{pca_threshold}σ)")
print(f"   → Mahalanobis 距离均值={mahal.mean():.2f}，最大={mahal.max():.2f}")
print(f"   → 发现 {len(pca_idx)} 条 PCA 孤立谱")

# ============================================================
# 3. 汇总剔除结果
# ============================================================
n_reject = reject_flags.sum()
n_keep   = n_total - n_reject
keep_idx = np.where(~reject_flags)[0]

print(f"\n{'='*55}")
print(f"📊 剔除汇总")
print(f"   总计: {n_total} 条  →  保留: {n_keep} 条，剔除: {n_reject} 条")
print(f"   剔除率: {n_reject/n_total*100:.1f}%")
print(f"{'='*55}")

reasons, counts = np.unique(reject_reason[reject_flags], return_counts=True)
for r, c in zip(reasons, counts):
    print(f"   {r}: {c} 条")

# ============================================================
# 4. 保存剔除记录 & 干净数据
# ============================================================
# 4a. 剔除日志
log_path = os.path.join(output_folder, "outlier_removal_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"# 拉曼光谱异常剔除记录\n")
    f.write(f"# 总计: {n_total}  保留: {n_keep}  剔除: {n_reject}\n")
    f.write(f"# 阈值: Cosmic={COSMIC_SPIKE_RATIO}x  SNR={SNR_THRESHOLD}  "
            f"PCC={PCC_THRESHOLD}  PCA={PCA_SIGMA_THRESHOLD}σ\n")
    f.write(f"# Index\tStatus\tReason\tPCC\tMahal\tFile\n")
    for i in range(n_total):
        status = "REJECT" if reject_flags[i] else "KEEP"
        f.write(f"{i}\t{status}\t{reject_reason[i]}\t"
                f"{pcc_vals[i]:.4f}\t{mahal[i]:.4f}\t{file_ids[i]}\n")
print(f"\n💾 剔除记录已保存: {log_path}")

# 4b. 干净数据矩阵（每行一条谱，第一列为波数）
clean_spectra = spectra[keep_idx]
clean_labels  = labels[keep_idx]
clean_matrix  = np.column_stack([wavenumber, clean_spectra.T])

# Header
col_labels = "\t".join([f"spec_{i}" for i in keep_idx])
clean_path = os.path.join(output_folder, "clean_spectra.txt")
np.savetxt(
    clean_path,
    clean_matrix,
    header=f"Wavenumber\t{col_labels}",
    delimiter="\t", fmt="%.6f", comments="#"
)
print(f"💾 干净谱矩阵已保存: {clean_path}  ({n_keep} 条，{len(wavenumber)} 点)")

# 4c. 每组的平均谱（可直接用于作图投稿）
unique_groups = list(dict.fromkeys(clean_labels))
avg_path = os.path.join(output_folder, "clean_group_mean_std.txt")
with open(avg_path, "w") as f:
    f.write(f"# 各组均值谱 ± STD（基于剔除后干净谱）\n")
    for g in unique_groups:
        g_spectra = clean_spectra[clean_labels == g]
        f.write(f"# Group: {g}  n={len(g_spectra)}\n")
        out = np.column_stack([wavenumber,
                               g_spectra.mean(axis=0),
                               g_spectra.std(axis=0)])
        np.savetxt(f, out, fmt="%.6f", delimiter="\t")
        f.write("\n")
print(f"💾 分组均值谱已保存: {avg_path}")

# 4d. 保留的单条光谱（保存到子文件夹，方便后续处理）
spectra_folder = os.path.join(output_folder, "kept_spectra")
os.makedirs(spectra_folder, exist_ok=True)
for idx, keep_i in enumerate(keep_idx):
    spectrum_data = np.column_stack([wavenumber, clean_spectra[idx]])
    # 根据原文件名生成输出文件名
    orig_filename = file_ids[keep_i].replace("/", "_").replace("\\", "_").replace(".txt", "")
    output_file = os.path.join(spectra_folder, f"{keep_i:04d}_{orig_filename}.txt")
    np.savetxt(output_file, spectrum_data, delimiter="\t", fmt="%.6f",
               header=f"Wavenumber_(cm-1)\tIntensity (Index_{keep_i})", comments="#")
print(f"💾 保留的单条光谱已保存到: {spectra_folder}  ({n_keep} 条)")

# ============================================================
# 5. 可视化
# ============================================================
group_colors = plt.cm.tab10(np.linspace(0, 0.9, len(unique_groups)))
gcolor = dict(zip(unique_groups, group_colors))

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#f7f9fc")
gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.42, wspace=0.38,
                       left=0.07, right=0.97,
                       top=0.92, bottom=0.07)

ax1 = fig.add_subplot(gs[0, 0])   # PCC 分布直方图
ax2 = fig.add_subplot(gs[0, 1])   # Mahalanobis 分布
ax3 = fig.add_subplot(gs[0, 2])   # PCA before/after
ax4 = fig.add_subplot(gs[1, 0])   # 剔除谱 vs 保留谱 示例
ax5 = fig.add_subplot(gs[1, 1])   # 干净均值谱 ± SD
ax6 = fig.add_subplot(gs[1, 2])   # 各组样本量对比

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

# 图1：PCC 分布
bins = np.linspace(pcc_vals.min()-0.02, 1.0, 60)
ax1.hist(pcc_vals[~reject_flags], bins=bins, color="#2980b9",
         alpha=0.75, label=f"Kept (n={n_keep})", edgecolor="white", lw=0.3)
ax1.hist(pcc_vals[reject_flags],  bins=bins, color="#e74c3c",
         alpha=0.75, label=f"Rejected (n={n_reject})", edgecolor="white", lw=0.3)
ax1.axvline(PCC_THRESHOLD, color="#c0392b", ls="--", lw=1.5,
            label=f"Threshold={PCC_THRESHOLD}")
ax1.set_xlabel("PCC (vs. mean spectrum)")
ax1.set_ylabel("Count")
ax1.set_title("PCC Distribution", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8)

# 图2：Mahalanobis 分布
ax2.hist(mahal[~reject_flags], bins=50, color="#27ae60",
         alpha=0.75, label="Kept", edgecolor="white", lw=0.3)
ax2.hist(mahal[reject_flags],  bins=50, color="#e74c3c",
         alpha=0.75, label="Rejected", edgecolor="white", lw=0.3)
ax2.axvline(PCA_SIGMA_THRESHOLD, color="#c0392b", ls="--", lw=1.5,
            label=f"Threshold={PCA_SIGMA_THRESHOLD}σ")
ax2.set_xlabel("Mahalanobis Distance (PCA space)")
ax2.set_ylabel("Count")
ax2.set_title("PCA Outlier Detection", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8)

# 图3：PCA 散点图 before/after 对比
ax3.scatter(scores[reject_flags, 0], scores[reject_flags, 1],
            c="#e74c3c", s=15, alpha=0.6, label="Rejected", zorder=2)
ax3.scatter(scores[~reject_flags, 0], scores[~reject_flags, 1],
            c="#2980b9", s=10, alpha=0.5, label="Kept", zorder=1)
ax3.axhline(0, color="gray", lw=0.4, ls=":")
ax3.axvline(0, color="gray", lw=0.4, ls=":")
expl = pca.explained_variance_ratio_
ax3.set_xlabel(f"PC1 ({expl[0]*100:.1f}%)")
ax3.set_ylabel(f"PC2 ({expl[1]*100:.1f}%)")
ax3.set_title("PCA: Before Removal", fontsize=11, fontweight="bold")
ax3.legend(fontsize=8, markerscale=1.5)

# 图4：异常谱 vs 正常谱示例
n_show = min(5, n_reject, n_keep)
offset = (spectra.max() - spectra.min()) * 0.15
for i, idx in enumerate(np.where(reject_flags)[0][:n_show]):
    ax4.plot(wavenumber, spectra[idx] + i * offset,
             color="#e74c3c", lw=0.8, alpha=0.7,
             label="Rejected" if i == 0 else "_")
for i, idx in enumerate(np.where(~reject_flags)[0][:n_show]):
    ax4.plot(wavenumber, spectra[idx] + i * offset,
             color="#2980b9", lw=0.8, alpha=0.7,
             label="Kept" if i == 0 else "_")
ax4.set_xlabel("Raman Shift (cm⁻¹)")
ax4.set_ylabel("Intensity (offset)")
ax4.set_title("Example Spectra\n(rejected vs. kept)", fontsize=11, fontweight="bold")
ax4.set_xlim(wavenumber.min(), wavenumber.max())
ax4.legend(fontsize=8)
ax4.set_yticks([])

# 图5：干净均值谱
for g in unique_groups:
    g_spectra = clean_spectra[clean_labels == g]
    gm  = g_spectra.mean(axis=0)
    gsd = g_spectra.std(axis=0)
    ax5.plot(wavenumber, gm, color=gcolor[g], lw=1.5,
             label=f"{g} (n={len(g_spectra)})")
    ax5.fill_between(wavenumber, gm - gsd, gm + gsd,
                     alpha=0.15, color=gcolor[g])
ax5.set_xlabel("Raman Shift (cm⁻¹)")
ax5.set_ylabel("Intensity (a.u.)")
ax5.set_title("Clean Mean Spectra ± SD", fontsize=11, fontweight="bold")
ax5.set_xlim(wavenumber.min(), wavenumber.max())
if len(unique_groups) > 1:
    ax5.legend(fontsize=7)

# 图6：各组剔除前后样本量
x = np.arange(len(unique_groups))
w = 0.35
n_before = [np.sum(labels == g)       for g in unique_groups]
n_after  = [np.sum(clean_labels == g) for g in unique_groups]
bars1 = ax6.bar(x - w/2, n_before, w, color="#95a5a6",
                label="Before", edgecolor="white")
bars2 = ax6.bar(x + w/2, n_after,  w,
                color=[gcolor[g] for g in unique_groups],
                label="After", edgecolor="white")
ax6.set_xticks(x)
ax6.set_xticklabels(unique_groups, rotation=30, ha="right", fontsize=8)
ax6.set_ylabel("Spectrum Count")
ax6.set_title("Sample Count per Group\n(Before vs. After)", fontsize=11, fontweight="bold")
ax6.legend(fontsize=8)
for bar in bars2:
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(int(bar.get_height())), ha="center", va="bottom", fontsize=7)

# 总标题
reject_rate = n_reject / n_total * 100
fig.suptitle(
    f"Outlier Removal Report  ·  {n_total} → {n_keep} spectra  "
    f"(rejected {n_reject}, {reject_rate:.1f}%)",
    fontsize=13, fontweight="bold", color="#1a252f", y=0.98
)

# 保存图
fig_path = os.path.join(output_folder, "outlier_removal_report.png")
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
fig_svg  = os.path.join(output_folder, "outlier_removal_report.svg")
fig.savefig(fig_svg, bbox_inches="tight")
print(f"\n📸 报告图已保存: {fig_path}")

plt.show()
print("\n✅ 完成！")