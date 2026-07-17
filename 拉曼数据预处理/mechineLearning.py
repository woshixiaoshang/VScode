import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. 配置：路径与图形参数
# ─────────────────────────────────────────────
DATA_DIRS = {
    "Mac":   r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\000_0hRaw_processed\1200",
    "Foam": r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\000_24hFoam_processed\1200",
    "Rapa":    r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\8hRa_processed\1200",
    "HP-CD": r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\8hHP-CD_processed\1200",
}

SHORT_LABELS = ["Mac", "Foam", "Rapa", "HP-CD"]
OUTPUT_DIR = r"D:\aaaSCNU\0data\Raman\EDA\ppt_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. 数据极速加载模块
# ─────────────────────────────────────────────
def find_file(root_folder, filename):
    for dirpath, _, files in os.walk(root_folder):
        if filename in files:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"未找到 {filename} 于 {root_folder}")

print("🚀 正在极速加载表面增强拉曼散射 (SERS) 光谱矩阵数据...")
X_list, y_list = [], []
wavenumbers = None
group_keys = list(DATA_DIRS.keys())

for group_label in group_keys:
    folder = DATA_DIRS[group_label]
    try:
        path = find_file(folder, "processed_spectra_matrix.txt")
        df = pd.read_csv(path, sep='\t', comment='#', header=None)
        if wavenumbers is None:
            wavenumbers = df.iloc[:, 0].values
        matrix = df.iloc[:, 1:].values.T
        X_list.append(matrix)
        y_list.extend([group_label] * matrix.shape[0])
    except Exception as e:
        print(f"  ❌ {group_label} 加载失败: {e}")

X_all = np.vstack(X_list)
y_all = np.array(y_list)
print(f"✅ 数据合并完毕！共 {X_all.shape[0]} 条光谱。\n")

# ─────────────────────────────────────────────
# 3. 训练大乱斗模型 & 提取评估数据
# ─────────────────────────────────────────────
print("⚔️ 正在划分数据集 (80% 训练, 20% 测试)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# 注意：SVM 必须开启 probability=True 才能画 ROC 曲线
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    "SVM (RBF)": SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []
confusion_matrices = {}
roc_data = {} # 存储每个模型的 ROC 数据

# 统一图形字体配置
plt.rcParams.update({"font.family": "Arial", "figure.dpi": 300})

print("🔥 开始训练多模型并生成评估数据...")
for model_name, model in models.items():
    print(f"  -> 正在处理 {model_name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test) # 获取预测概率
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    results.append({"Model": model_name, "Metric": "Accuracy", "Score": acc})
    results.append({"Model": model_name, "Metric": "Macro F1", "Score": f1})
    
    # 获取混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=group_keys)
    confusion_matrices[model_name] = cm
    
    # ---------------------------------------------
    # 计算 Macro-average ROC
    # ---------------------------------------------
    # 标签二值化处理 (针对多分类 ROC)
    Y_test_bin = label_binarize(y_test, classes=model.classes_)
    n_classes = Y_test_bin.shape[1]
    
    fpr = dict()
    tpr = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], y_score[:, i])
    
    # 汇总所有的 FPR
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    macro_fpr = all_fpr
    macro_tpr = mean_tpr
    macro_auc = auc(macro_fpr, macro_tpr)
    roc_data[model_name] = (macro_fpr, macro_tpr, macro_auc)

df_results = pd.DataFrame(results)

# ─────────────────────────────────────────────
# 4. 出图 1: 无网格线的综合性能柱状图
# ─────────────────────────────────────────────
print("\n🎨 正在生成无网格线性能柱状图...")
fig_bar, ax_bar = plt.subplots(figsize=(10, 6.5))

sns.barplot(
    data=df_results, x="Model", y="Score", hue="Metric", 
    palette=["#2B5B84", "#D1603D"], edgecolor="black", linewidth=1.2, ax=ax_bar
)

for container in ax_bar.containers:
    ax_bar.bar_label(container, fmt='%.3f', label_type='edge', padding=4, fontsize=11, fontweight='bold')

ax_bar.set_title("Benchmarking Machine Learning Classifiers on Multiplexed SERS Data", 
                 fontsize=17, fontweight='bold', pad=20)
ax_bar.set_ylabel("Performance Score", fontsize=14, fontweight='bold', labelpad=10)
ax_bar.set_xlabel("Algorithm Architecture", fontsize=14, fontweight='bold', labelpad=10)
ax_bar.set_ylim(0, 1.15) 
ax_bar.legend(title="", fontsize=12, loc='upper right', framealpha=0.9, edgecolor="black")

# 【已移除】这里去掉了 Y 轴的辅助网格线
ax_bar.grid(False) 
plt.xticks(fontsize=12, fontweight='bold')

bar_path = os.path.join(OUTPUT_DIR, "ppt_01_SERS_model_benchmark_nogrid.svg")
# 强制背景透明
plt.savefig(bar_path, format="svg", transparent=True, bbox_inches='tight', facecolor='none', edgecolor='none')
plt.close(fig_bar)

# ─────────────────────────────────────────────
# 5. 出图 2: 混淆矩阵全家福
# ─────────────────────────────────────────────
print("🎨 正在拼装混淆矩阵全家福...")
fig_all, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()

for ax, (model_name, cm) in zip(axes, confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS, 
                cbar=False, annot_kws={"size": 12, "weight": "bold"}, ax=ax)
    # 为热图强制设置透明背景
    ax.patch.set_alpha(0)
    
    ax.set_title(model_name, fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    if ax == axes[0]:
        ax.set_ylabel('True Phenotype', fontsize=13, fontweight='bold')

# Hide the unused subplot if present
for ax in axes[len(confusion_matrices):]:
    ax.axis('off')

plt.tight_layout()
all_cm_path = os.path.join(OUTPUT_DIR, "ppt_02_confusion_matrices_ALL.svg")
plt.savefig(all_cm_path, format="svg", transparent=True, bbox_inches='tight', facecolor='none', edgecolor='none')
plt.close(fig_all)

# ─────────────────────────────────────────────
# 6. 出图 3: 多模型 Macro-ROC 对比曲线图
# ─────────────────────────────────────────────
print("🎨 正在生成多模型 ROC 曲线图...")
fig_roc, ax_roc = plt.subplots(figsize=(8, 7))

# 定义 5 种不同算法的配色和线型
line_styles = [
    {"color": "#D1603D", "ls": "-"},   # RF - 陶土红实线
    {"color": "#2B5B84", "ls": "--"},  # GB - 藏青蓝虚线
    {"color": "#2BAE66", "ls": "-."},  # SVM - 翠绿点划线
    {"color": "#8E5DB7", "ls": ":"},   # NN - 紫色点线
    {"color": "#666666", "ls": "-"}    # KNN - 灰色实线
]

for (model_name, (fpr, tpr, roc_auc)), style in zip(roc_data.items(), line_styles):
    ax_roc.plot(fpr, tpr, color=style["color"], linestyle=style["ls"], linewidth=2.5,
                label=f"{model_name} (AUC = {roc_auc:.3f})")

# 绘制随机猜测的对角线
ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)

ax_roc.set_title("Multi-class ROC Curves (Macro-average)", fontsize=16, fontweight='bold', pad=20)
ax_roc.set_xlabel("False Positive Rate", fontsize=14, fontweight='bold')
ax_roc.set_ylabel("True Positive Rate", fontsize=14, fontweight='bold')
ax_roc.set_xlim([-0.02, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.grid(False) # ROC 图同样无网格线
ax_roc.legend(loc="lower right", fontsize=11, framealpha=0.9, edgecolor="black")

roc_path = os.path.join(OUTPUT_DIR, "ppt_03_ROC_Curves_comparison.svg")
plt.savefig(roc_path, format="svg", transparent=True, bbox_inches='tight', facecolor='none', edgecolor='none')
plt.close(fig_roc)

print(f"🎉 全部出图完成！你的终极 PPT 素材已准备就绪：")
print(f"  👉 1. 柱状图 (无网格): ppt_01_SERS_model_benchmark_nogrid.svg")
print(f"  👉 2. 混淆矩阵 (透明) : ppt_02_confusion_matrices_ALL.svg")
print(f"  👉 3. ROC曲线 (含AUC): ppt_03_ROC_Curves_comparison.svg")