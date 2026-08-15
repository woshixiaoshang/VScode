import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import warnings

warnings.filterwarnings('ignore')


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return None
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y)
    ellipse.set_transform(transf.translate(np.mean(x), np.mean(y)) + ax.transData)
    ax.add_patch(ellipse)
    return ellipse

# ─────────────────────────────────────────────
# 0. 配置：路径与图形参数
# ─────────────────────────────────────────────
# 原始数据路径 (用于提取单样本光谱矩阵)
DATA_DIRS = {
    "Mac":   r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\000_0hRaw_processed\1200",
    "Foam": r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\000_24hFoam_processed\1200",
    "Ra":    r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\8hRa_processed\1200",
    "HP-CD": r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\8hHP-CD_processed\1200",
}

# 基础 EDA 的输出目录
EDA_OUTPUT_DIR = r"D:\aaaSCNU\0data\Raman\EDA\eda_results"
# 进阶分析输出目录
ADV_OUTPUT_DIR = os.path.join(EDA_OUTPUT_DIR, "advanced_analysis")
os.makedirs(ADV_OUTPUT_DIR, exist_ok=True)

COLORS = {"Mac": "#378ADD", "Foam": "#1D9E75", "Ra": "#8E5DB7", "HP-CD": "#D85A30"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ─────────────────────────────────────────────
# 1. 数据加载与预处理
# ─────────────────────────────────────────────
def find_file(root_folder, filename):
    for dirpath, _, files in os.walk(root_folder):
        if filename in files:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"未找到 {filename} 于 {root_folder}")

print("正在加载单细胞光谱矩阵与基础 EDA 结果...")
X_list, y_list = [], []
wavenumbers = None

# 加载原始光谱用于散点和统计分布
for group_label, folder in DATA_DIRS.items():
    path = find_file(folder, "processed_spectra_matrix.txt")
    df = pd.read_csv(path, sep='\t', comment='#', header=None)
    if wavenumbers is None:
        wavenumbers = df.iloc[:, 0].values
    
    matrix = df.iloc[:, 1:].values.T  # 转换为 样本 × 波数
    X_list.append(matrix)
    y_list.extend([group_label] * matrix.shape[0])

X_all = np.vstack(X_list)
y_all = np.array(y_list)
df_all = pd.DataFrame(X_all, columns=np.round(wavenumbers, 1))
df_all['Group'] = y_all

# 加载基础 EDA 筛选出的关键波数
key_df = pd.read_csv(os.path.join(EDA_OUTPUT_DIR, "key_wavenumbers.csv"))
top_wavenumbers = key_df['wavenumber_cm-1'].head(20).values
top_wn_cols = [float(np.round(wn, 1)) for wn in top_wavenumbers]

# ─────────────────────────────────────────────
# 功能 1: PCA 得分图 (带 95% 置信椭圆)
# ─────────────────────────────────────────────
from sklearn.cross_decomposition import PLSRegression

print("1. 绘制 PLS-DA 得分图 (替代 PCA)...")
# PLS-DA 需要将文本标签转换为独热编码 (One-Hot Encoding) 矩阵
Y_onehot = pd.get_dummies(y_all).values

# 拟合 PLS-DA 模型
plsda = PLSRegression(n_components=2)
X_scores, Y_scores = plsda.fit_transform(X_all, Y_onehot)

df_plsda = pd.DataFrame({"LV1": X_scores[:, 0], "LV2": X_scores[:, 1], "Group": y_all})

fig, ax = plt.subplots(figsize=(8, 6))
# 绘制散点图
sns.scatterplot(data=df_plsda, x="LV1", y="LV2", hue="Group", palette=COLORS, s=30, alpha=0.6, edgecolor=None, ax=ax)

# 添加 95% 置信椭圆
for group in DATA_DIRS.keys():
    group_data = df_plsda[df_plsda["Group"] == group]
    confidence_ellipse(group_data["LV1"], group_data["LV2"], ax, n_std=2.0, 
                       edgecolor=COLORS[group], lw=1.5, ls='--')

ax.set_title("PLS-DA Scores Plot", fontsize=14, fontweight='500')
ax.set_xlabel("Latent Variable 1 (LV1)", fontsize=12)
ax.set_ylabel("Latent Variable 2 (LV2)", fontsize=12)
plt.savefig(os.path.join(ADV_OUTPUT_DIR, "adv1_plsda_scores.svg"))
plt.close()

# ─────────────────────────────────────────────
# 功能 2: 关键生物标记物比值分布 (小提琴图)
# ─────────────────────────────────────────────
print("2. 绘制特征波段比值分布...")
# 自动选取差异最显著的两个波段构成比值（常用于揭示脂质蓄积或大分子相变状态）
if len(top_wn_cols) >= 2:
    wn_num = top_wn_cols[0]  # ANOVA F值最高的波段 (分子)
    wn_den = top_wn_cols[1]  # ANOVA F值第二高的波段 (分母)
    
    # 获取最接近的实际列名
    col_num = min(df_all.columns[:-1], key=lambda x: abs(x - wn_num))
    col_den = min(df_all.columns[:-1], key=lambda x: abs(x - wn_den))
    
    ratio_name = f"Ratio ({col_num} / {col_den})"
    df_all[ratio_name] = df_all[col_num] / (df_all[col_den] + 1e-8) # 避免除零

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(x="Group", y=ratio_name, data=df_all, palette=COLORS, inner=None, alpha=0.4, ax=ax)
    sns.stripplot(x="Group", y=ratio_name, data=df_all, palette=COLORS, size=3, jitter=True, alpha=0.7, ax=ax)
    
    ax.set_title(f"Biomarker Ratio Dynamics: {col_num} cm⁻¹ / {col_den} cm⁻¹", fontsize=14)
    ax.set_ylabel("Intensity Ratio", fontsize=12)
    plt.savefig(os.path.join(ADV_OUTPUT_DIR, "adv2_band_ratio.svg"))
    plt.close()

# ─────────────────────────────────────────────
# 功能 3: 特征波段相关性热图 (聚类网络预演)
# ─────────────────────────────────────────────
print("3. 绘制关键差异波段的相关性聚类热图...")
# 提取用于观测协同机制的前 15 个关键波数
actual_top_cols = []
for wn in top_wn_cols[:15]:
    closest_col = min(df_all.columns[:-2], key=lambda x: abs(x - wn)) # 排除 Group 和 Ratio
    if closest_col not in actual_top_cols:
        actual_top_cols.append(closest_col)

df_top_features = df_all[actual_top_cols]
corr_matrix = df_top_features.corr(method='spearman')

# 绘制带有层次聚类的热图
g = sns.clustermap(corr_matrix, cmap="vlag", center=0, annot=True, fmt=".2f", 
                   annot_kws={"size": 8}, figsize=(10, 10), linewidths=0.5,
                   cbar_kws={'label': 'Spearman Correlation'})
g.fig.suptitle("Co-evolution of Key Wavenumbers (Mechanism Network)", y=1.02, fontsize=15)
plt.savefig(os.path.join(ADV_OUTPUT_DIR, "adv3_feature_correlation.svg"))
plt.close()

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 功能 4: 基于组均值光谱的系统层次聚类
# ─────────────────────────────────────────────
print("4. 绘制组别层次聚类树状图...")
# 直接读取基础 EDA 的组平均值进行宏观距离计算
summary_df = pd.read_csv(os.path.join(EDA_OUTPUT_DIR, "group_statistics_summary.csv"))

# 【修复代码 1】：将波数四舍五入到保留两位小数，强制不同组的波数完全对齐
summary_df['wavenumber_cm-1'] = summary_df['wavenumber_cm-1'].round(2)

# 进行数据透视
mean_spectra = summary_df.pivot(index='group', columns='wavenumber_cm-1', values='mean')

# 【修复代码 2】：清理矩阵，丢弃任何依然存在 NaN 的列，确保全是有限值 (finite values)
mean_spectra = mean_spectra.dropna(axis=1)

# 计算欧氏距离并使用 Ward 方差最小化法连接
distances = pdist(mean_spectra.values, metric='euclidean')
linkage_matrix = linkage(distances, method='ward')

fig, ax = plt.subplots(figsize=(7, 5))
dendro = dendrogram(linkage_matrix, labels=mean_spectra.index, ax=ax, 
                    leaf_rotation=0, leaf_font_size=12, color_threshold=0)
ax.set_title("Hierarchical Clustering of Macrophage Phenotypes", fontsize=14)
ax.set_ylabel("Ward's Distance", fontsize=12)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(False)
plt.savefig(os.path.join(ADV_OUTPUT_DIR, "adv4_hierarchical_clustering.svg"))
plt.close()

print(f"\n✓ 高级 EDA 分析完成！结果已保存在: {ADV_OUTPUT_DIR}")