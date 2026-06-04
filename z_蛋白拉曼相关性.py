import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 调全局字体字号
plt.rcParams['pdf.fonttype'] = 42   # 有这一句，导出的pdf才能用ai改字，不然字会被识别为图片
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 22

# 蛋白表格和带shap值的拉曼表格地址
protein_df = pd.read_excel("数据/郭/重要差异蛋白.xlsx")
raman_df = pd.read_excel("数据/郭/拉曼峰 SHAP 值.xlsx")

# 列名
protein_cols = [
    "EAEC_Mean",
    "EIEC_Mean",
    "EPEC_Mean",
    "ETEC_Mean",
    "STEC_Mean"
]

raman_cols_raw = [  # 这个是原始光谱的平均
    "EAEC",
    "EIEC",
    "EPEC",
    "ETEC",
    "STEC"
]

raman_cols_norm = [  # 这个是归一化后光谱的平均
    "EAEC min_max",
    "EIEC min_max",
    "EPEC min_max",
    "ETEC min_max",
    "STEC min_max"
]

# 转为数值（防止Excel空白字符串）
protein_df[protein_cols] = protein_df[protein_cols].apply(pd.to_numeric, errors="coerce")
raman_df[raman_cols_raw] = raman_df[raman_cols_raw].apply(pd.to_numeric, errors="coerce")
raman_df[raman_cols_norm] = raman_df[raman_cols_norm].apply(pd.to_numeric, errors="coerce")

# 蛋白空白填充：全局最小值 * 0.5
global_min = protein_df[protein_cols].min().min()
fill_value = global_min * 0.5
protein_df[protein_cols] = protein_df[protein_cols].fillna(fill_value)

# SHAP求和筛选Top50 Raman
raman_df = raman_df.sort_values("sum |SHAP value|", ascending=False).head(50)

# Raman位移取整数
raman_df["feature_names"] = raman_df["feature_names"].round().astype(int)

# 提取矩阵
protein_mat = protein_df[protein_cols].values
raman_mat_raw = raman_df[raman_cols_raw].values
raman_mat_norm = raman_df[raman_cols_norm].values

# 删除标准差为0的行
protein_mask = np.std(protein_mat, axis=1) != 0
protein_mat = protein_mat[protein_mask]
protein_df = protein_df.loc[protein_mask]

raman_mask_raw = np.std(raman_mat_raw, axis=1) != 0
raman_mat_raw = raman_mat_raw[raman_mask_raw]
raman_df_raw = raman_df.loc[raman_mask_raw]

raman_mask_norm = np.std(raman_mat_norm, axis=1) != 0
raman_mat_norm = raman_mat_norm[raman_mask_norm]
raman_df_norm = raman_df.loc[raman_mask_norm]


# 计算相关矩阵
def compute_corr(protein_mat, raman_mat):

    n_protein = protein_mat.shape[0]
    n_raman = raman_mat.shape[0]

    corr_matrix = np.zeros((n_protein, n_raman))

    for i in range(n_protein):
        for j in range(n_raman):

            p = protein_mat[i]
            r = raman_mat[j]

            if np.std(p) == 0 or np.std(r) == 0:
                corr = 0
            else:
                corr = np.corrcoef(p, r)[0,1]

            corr_matrix[i, j] = corr

    corr_matrix = np.nan_to_num(corr_matrix)

    return corr_matrix


corr_raw = compute_corr(raman_mat_raw, protein_mat)
corr_norm = compute_corr(raman_mat_norm, protein_mat)


# 输出相关性矩阵
def export_corr_table(corr, protein_df, raman_labels, filename):

    # 构建相关矩阵 DataFrame
    corr_df = pd.DataFrame(
        corr,
        index=raman_labels
    ).T  # 转置：行为蛋白，列为拉曼

    # 重置索引方便拼接
    corr_df.reset_index(drop=True, inplace=True)

    # 提取蛋白信息列（顺序要和 protein_mat 一致）
    protein_info = protein_df[[
        "Protein_ID",
        "Protein_Description",
        "蛋白质描述",
        "名字"
    ]].reset_index(drop=True)

    # 合并
    final_df = pd.concat([protein_info, corr_df], axis=1)

    # 保存
    final_df.to_excel(filename, index=False)


# 画层次聚类热图
def plot_clustermap(corr, protein_labels, raman_labels, filename):

    df = pd.DataFrame(
        corr,
        index=raman_labels,
        columns=protein_labels
    )

    g = sns.clustermap(
        df,
        cmap="coolwarm",
        center=0,
        figsize=(28, 24),
        xticklabels=True,
        yticklabels=True,
        tree_kws={'linewidths': 3},  # 聚类树线条宽度
    )
    g.ax_heatmap.tick_params(axis='x', labelsize=20)
    plt.savefig(filename, dpi=600)
    plt.close()


# 标签
raman_labels_raw = raman_df_raw["feature_names"].astype(str).values
raman_labels_norm = raman_df_norm["feature_names"].astype(str).values

protein_id = protein_df["名字"].values
gene_name = protein_df["Gene_Name"].values

# 画图
plot_clustermap(
    corr_raw,
    protein_id,
    raman_labels_raw,
    "输出结果/ProteinID_RamanRaw_heatmap.pdf"
)

plot_clustermap(
    corr_norm,
    protein_id,
    raman_labels_norm,
    "输出结果/ProteinID_RamanNorm_heatmap.pdf"
)

# 导出相关矩阵（Raw）
export_corr_table(
    corr_raw,
    protein_df,
    raman_labels_raw,
    "输出结果/Protein_RamanRaw_corr.xlsx"
)

# 导出相关矩阵（Norm）
export_corr_table(
    corr_norm,
    protein_df,
    raman_labels_norm,
    "输出结果/Protein_RamanNorm_corr.xlsx"
)
