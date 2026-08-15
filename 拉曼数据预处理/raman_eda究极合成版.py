"""
拉曼光谱双区域整合 EDA（纯矩阵直读 + 深度文件检索版）
- 数据结构要求：组别 -> 1200/3000 -> 任意子文件夹 -> 矩阵文件/PCA文件
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. 全局配置
# ─────────────────────────────────────────────
ROOT_DIR = r"D:\aaaSCNU\0data\Raman\0处理\原始数据"  # 请修改为你的真实路径
OUTPUT   = r"D:\aaaSCNU\0data\Raman\0处理\EDA"

COLORS   = ["#378ADD", "#1D9E75", "#D85A30", "#534AB7", "#FAC775"]
ALPHAS   = [0.15, 0.12, 0.10, 0.15, 0.15]

# 高波数区脂质机制峰位配置: (中心波数, 积分半窗口大小, 名称)
HW_PEAKS = {
    "CH2_sym":  (2850, 15),
    "CH2_asy":  (2880, 15),
    "CH3_sym":  (2930, 15),
    "CH3_asy":  (2960, 15),
    "Unsat_CH": (3010, 15)
}

HW_RATIOS = [
    ("CH2_sym", "CH3_sym", "Lipid Chain Length (2850/2930)"),
    ("CH2_asy", "CH3_sym", "Lipid Packing (2880/2930)"),
    ("Unsat_CH", "CH2_sym", "Lipid Unsaturation (3010/2850)")
]

os.makedirs(os.path.join(OUTPUT, "1200_Fingerprint"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT, "3000_HighWavenumber"), exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"
})

# ─────────────────────────────────────────────
# 1. 核心数据加载与检索函数
# ─────────────────────────────────────────────
def scan_directory(root_path):
    """扫描文件夹，提取各组路径"""
    groups = {}
    for g in sorted(os.listdir(root_path)):
        p = os.path.join(root_path, g)
        if os.path.isdir(p):
            groups[g] = {
                "1200": os.path.join(p, "1200") if os.path.exists(os.path.join(p, "1200")) else None,
                "3000": os.path.join(p, "3000") if os.path.exists(os.path.join(p, "3000")) else None
            }
    return groups

def find_deep_file(base_path, pattern):
    """深度递归查找文件，无视子文件夹层级"""
    if not base_path or not os.path.exists(base_path):
        return None
    matches = list(Path(base_path).rglob(pattern))
    return str(matches[0]) if matches else None

def load_matrix(filepath):
    """读取矩阵并强制升序排列"""
    if not filepath or not os.path.exists(filepath):
        return None, None
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', index_col=0)
        wn = df.index.values.astype(float)
        mat = df.values.T  
        if wn[0] > wn[-1]:
            wn = wn[::-1]
            mat = mat[:, ::-1]
        return wn, mat
    except Exception as e:
        print(f"⚠️ 读取 {filepath} 失败: {e}")
        return None, None

def calc_peak_area(wn, intensities, center, window):
    """数值积分算面积"""
    mask = (wn >= center - window) & (wn <= center + window)
    if not np.any(mask): return np.nan
    return np.trapz(intensities[mask], wn[mask])

# ─────────────────────────────────────────────
# 2. 指纹区 (1200) 分析模块
# ─────────────────────────────────────────────
def run_fingerprint_eda(groups_data):
    print("\n" + "="*40 + "\n🚀 [1/2] 运行 1200 指纹区分析\n" + "="*40)
    out_dir = os.path.join(OUTPUT, "1200_Fingerprint")
    
    matrices = {}
    pca_dict = {}
    
    for grp, paths in groups_data.items():
        if not paths["1200"]: continue
            
        # 使用雷达深度搜索矩阵文件
        mat_path = find_deep_file(paths["1200"], "*processed_spectra_matrix*.txt")
        wn, mat = load_matrix(mat_path)
        
        if mat is not None:
            matrices[grp] = {
                "mat": mat, "mean": np.mean(mat, axis=0),
                "std": np.std(mat, axis=0), "var": np.var(mat, axis=0)
            }
            print(f"  [{grp}] 1200区: 成功加载 {mat.shape[0]} 条光谱")
            
        # 使用雷达深度搜索 PCA 文件 (无视 Data 文件夹嵌套)
        pca_path = find_deep_file(paths["1200"], "*pca_loadings*.txt")
        if pca_path:
            with open(pca_path) as f: 
                header = f.readline().strip().lstrip('#').strip().split()
            pca_dict[grp] = pd.read_csv(pca_path, sep=r'\s+', comment='#', names=header, engine='python')

    if not matrices: return
    labels = list(matrices.keys())
    
    # ── 绘图逻辑 ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1.2]})
    for i, grp in enumerate(labels):
        m, s = matrices[grp]["mean"], matrices[grp]["std"]
        axes[0].plot(wn, m, color=COLORS[i%len(COLORS)], lw=1.2, label=grp)
        axes[0].fill_between(wn, m-s, m+s, color=COLORS[i%len(COLORS)], alpha=ALPHAS[i%len(ALPHAS)])
    axes[0].set_title("Fig 1: Mean Spectra (Fingerprint)"); axes[0].legend()
    
    for idx, (a, b) in enumerate(combinations(labels, 2)):
        diff = matrices[a]["mean"] - matrices[b]["mean"]
        axes[1].plot(wn, diff, lw=1.0, label=f"{a}-{b}", alpha=0.8)
    axes[1].axhline(0, color="gray", lw=0.6, ls="--"); axes[1].legend(ncol=3)
    plt.savefig(os.path.join(out_dir, "Fig1_mean_spectra.png")); plt.close()

    mats = [matrices[k]["mat"] for k in labels]
    F_vals, p_vals = stats.f_oneway(*mats)
    sig_mask = p_vals < 0.05

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1, 0.5]})
    axes[0].plot(wn, F_vals, color="#378ADD"); axes[0].set_title("Fig 2: Pointwise ANOVA")
    axes[1].plot(wn, -np.log10(np.clip(p_vals, 1e-300, 1)), color="#D85A30")
    axes[2].fill_between(wn, 0, sig_mask.astype(float), color="#1D9E75", step='mid')
    plt.savefig(os.path.join(out_dir, "Fig2_anova.png")); plt.close()

    if pca_dict:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        for pc_idx, ax in enumerate(axes):
            for i, grp in enumerate(labels):
                if grp in pca_dict:
                    cols = [c for c in pca_dict[grp].columns if c.startswith(f"PC{pc_idx+1}")]
                    if cols: ax.plot(wn, pca_dict[grp][cols[0]].values, color=COLORS[i%len(COLORS)], label=grp)
            ax.set_title(f"Fig 3: PC{pc_idx+1} Loadings"); ax.legend()
        plt.savefig(os.path.join(out_dir, "Fig3_pca.png")); plt.close()

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, grp in enumerate(labels): 
        ax.plot(wn, matrices[grp]["var"], color=COLORS[i], label=f"{grp} (within)", alpha=0.5)
    means_array = np.array([matrices[k]["mean"] for k in labels])
    ax.plot(wn, np.var(means_array, axis=0), color="black", lw=1.2, ls="--", label="Between-class var")
    ax.set_title("Fig 4: Variance Analysis"); ax.legend()
    plt.savefig(os.path.join(out_dir, "Fig4_variance.png")); plt.close()

    f_threshold = np.percentile(F_vals[sig_mask], 80) if np.sum(sig_mask) > 0 else 0
    key_mask = sig_mask & (F_vals > f_threshold)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, grp in enumerate(labels): ax.plot(wn, matrices[grp]["mean"], color=COLORS[i], label=grp)
    for w in wn[key_mask]: ax.axvspan(w-2, w+2, color="#FAC775", alpha=0.25, zorder=1)
    ax.set_title("Fig 5: Key Wavenumbers Highlight"); ax.legend()
    plt.savefig(os.path.join(out_dir, "Fig5_key_wavenumbers.png")); plt.close()
    
    pd.DataFrame({"wavenumber_cm-1": wn[key_mask], "F_statistic": F_vals[key_mask]}).to_csv(os.path.join(out_dir, "key_wavenumbers.csv"), index=False)
    print("✓ 指纹区分析完毕。")


# ─────────────────────────────────────────────
# 3. 高波数区 (3000) 分析模块
# ─────────────────────────────────────────────
def run_highwn_eda(groups_data):
    print("\n" + "="*40 + "\n🚀 [2/2] 运行 3000 高波数区(脂质结构)分析\n" + "="*40)
    out_dir = os.path.join(OUTPUT, "3000_HighWavenumber")
    area_data = {}
    
    for grp, paths in groups_data.items():
        if not paths["3000"]: continue
        
        # 深度检索，优先找 hw_spectra_matrix，没有就找带 matrix 的 txt
        mat_path = find_deep_file(paths["3000"], "*hw_spectra_matrix*.txt")
        if not mat_path:
            mat_path = find_deep_file(paths["3000"], "*matrix*.txt")
            
        wn, mat = load_matrix(mat_path)
        if mat is None: continue
        
        area_data[grp] = {p: [] for p in HW_PEAKS.keys()}
        for spec in mat:
            for peak_name, (center, window) in HW_PEAKS.items():
                area_data[grp][peak_name].append(calc_peak_area(wn, spec, center, window))
                
        for p in HW_PEAKS.keys():
            area_data[grp][p] = np.array(area_data[grp][p])
        print(f"  [{grp}] 3000区: 成功积分 {mat.shape[0]} 条光谱的 5 大特征峰")

    if not area_data: return
    labels = list(area_data.keys())

    for (num_key, den_key, title) in HW_RATIOS:
        ratio_arrays = []
        for grp in labels:
            num = area_data[grp][num_key]
            den = area_data[grp][den_key]
            valid = (den > 0)
            ratio_arrays.append(num[valid] / den[valid])
            
        F_stat, p_anova = stats.f_oneway(*ratio_arrays)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        means = [d.mean() for d in ratio_arrays]
        sems = [d.std() / np.sqrt(len(d)) for d in ratio_arrays]
        x = np.arange(len(labels))
        
        ax.bar(x, means, yerr=sems, width=0.55, color=COLORS[:len(labels)], alpha=0.8, capsize=6)
        rng = np.random.default_rng(42)
        for i, d in enumerate(ratio_arrays):
            jitter = rng.uniform(-0.15, 0.15, size=min(len(d), 300))
            sample = rng.choice(d, len(jitter), replace=False) if len(d)>300 else d
            ax.scatter(x[i]+jitter, sample, color=COLORS[i], alpha=0.3, s=10, zorder=3, edgecolors='none')

        pairs = list(combinations(range(len(labels)), 2))
        y_max = max([d.max() for d in ratio_arrays])
        y_step = y_max * 0.08
        curr_y = y_max + y_step
        
        for i, j in pairs:
            _, p_raw = stats.ttest_ind(ratio_arrays[i], ratio_arrays[j], equal_var=False)
            p_adj = min(p_raw * len(pairs), 1.0)
            
            if p_adj < 0.001: sig_txt = "***"
            elif p_adj < 0.01: sig_txt = "**"
            elif p_adj < 0.05: sig_txt = "*"
            else: sig_txt = "ns"
            
            h = y_step * 0.2
            ax.plot([x[i], x[i], x[j], x[j]], [curr_y, curr_y+h, curr_y+h, curr_y], lw=1, color="black")
            ax.text((x[i]+x[j])/2, curr_y+h, sig_txt, ha="center", va="bottom", fontsize=11)
            curr_y += y_step * 1.5

        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel("Peak Area Ratio", fontsize=11)
        ax.set_title(f"{title}\nANOVA F={F_stat:.1f}, p={p_anova:.2e}", fontsize=12, pad=15)
        
        safe_name = title.split(" ")[0] + "_" + title.split("(")[-1].replace(")", "").replace("/", "_")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{safe_name}_barplot.png"))
        plt.close()
        
    print("✓ 高波数区脂质结构分析完毕。")

if __name__ == "__main__":
    groups_data = scan_directory(ROOT_DIR)
    if not groups_data:
        print("未找到数据文件夹！请检查 ROOT_DIR 路径。")
    else:
        run_fingerprint_eda(groups_data)
        run_highwn_eda(groups_data)
        print(f"\n🎉 全部 EDA 分析顺利完成！结果保存在: {OUTPUT}")