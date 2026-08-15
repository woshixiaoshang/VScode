import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# 只抑制已知无害的弃用/未来警告，保留 ConvergenceWarning 等重要提示
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ─────────────────────────────────────────────
# 1. 配置：路径、切分参数、随机种子
#    建议：把绝对路径改为相对路径 / 环境变量 / 命令行参数以提升可移植性
# ─────────────────────────────────────────────
CONFIG = {
    "data_dirs": {
        "Mac":    r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\000_0hRaw_processed\1200",
        "Foam":   r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\000_24hFoam_processed\1200",
        "Rapa":   r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\8hRa_processed\1200",
        "HP-CD":  r"D:\aaaSCNU\0data\Raman\0处理\预处理processed\8hHP-CD_processed\1200",
    },
    "output_dir": r"D:\aaaSCNU\0data\Raman\EDA\ppt_results",
    "test_size": 0.2,
    "random_state": 42,
    "mlp_max_iter": 200,
}

# 调色板集中管理，便于统一风格
MODEL_COLORS = ["#D1603D", "#2B5B84", "#2BAE66", "#8E5DB7", "#666666"]
METRIC_PALETTE = {"Accuracy": "#2B5B84", "Macro F1": "#D1603D"}


# ─────────────────────────────────────────────
# 2. 数据加载模块
# ─────────────────────────────────────────────
def find_file(root_folder, filename):
    """优先直接拼接路径，找不到再递归遍历目录树。"""
    direct = os.path.join(root_folder, filename)
    if os.path.isfile(direct):
        return direct
    for dirpath, _, files in os.walk(root_folder):
        if filename in files:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"未找到 {filename} 于 {root_folder}")


def load_spectra(data_dirs):
    """加载所有组的光谱矩阵，返回 (X, y, wavenumbers)。"""
    X_list, y_list = [], []
    wavenumbers = None
    for label, folder in data_dirs.items():
        if not os.path.isdir(folder):
            print(f"  ❌ 目录不存在: {folder}")
            continue
        try:
            path = find_file(folder, "processed_spectra_matrix.txt")
            df = pd.read_csv(path, sep='\t', comment='#', header=None)
            if wavenumbers is None:
                wavenumbers = df.iloc[:, 0].values
            matrix = df.iloc[:, 1:].values.T
            if matrix.shape[1] == 0:
                print(f"  ❌ {label} 未包含光谱列，已跳过")
                continue
            X_list.append(matrix)
            y_list.extend([label] * matrix.shape[0])
        except Exception as e:
            print(f"  ❌ {label} 加载失败: {e}")

    if not X_list:
        raise RuntimeError("没有任何光谱加载成功，请检查 data_dirs 路径与文件是否存在")
    return np.vstack(X_list), np.array(y_list), wavenumbers


# ─────────────────────────────────────────────
# 3. 模型构建（统一包 StandardScaler）
# ─────────────────────────────────────────────
def build_models(random_state, mlp_max_iter):
    """返回 模型名 -> Pipeline(标准化 + 分类器) 的字典。"""
    return {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=random_state,
                                           class_weight='balanced', n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=50, random_state=random_state)),
        ]),
        # 注：不使用 probability=True（避免 Platt 校准的额外开销），ROC 改用 decision_function
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel='rbf', class_weight='balanced', random_state=random_state)),
        ]),
        "Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(50,), max_iter=mlp_max_iter,
                                  random_state=random_state)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
    }


# ─────────────────────────────────────────────
# 4. 训练 + 评估
# ─────────────────────────────────────────────
def evaluate_models(models, X_train, X_test, y_train, y_test, class_order):
    """训练各模型，返回 (results, confusion_matrices, roc_data)。"""
    results = []
    confusion_matrices = {}
    roc_data = {}

    # y_test 与类别顺序对所有模型一致，Y_test_bin 只需算一次
    Y_test_bin = label_binarize(y_test, classes=class_order)
    n_classes = Y_test_bin.shape[1]

    for model_name, model in models.items():
        print(f"  -> 正在处理 {model_name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 收敛性检查（主要针对 MLP）
        n_iter = getattr(model, "n_iter_", None)
        if n_iter is not None and n_iter >= CONFIG["mlp_max_iter"]:
            print(f"  ⚠️ {model_name} 未在 {CONFIG['mlp_max_iter']} 次迭代内收敛，建议增大 max_iter")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results.append({"Model": model_name, "Metric": "Accuracy", "Score": acc})
        results.append({"Model": model_name, "Metric": "Macro F1", "Score": f1})

        cm = confusion_matrix(y_test, y_pred, labels=class_order)
        confusion_matrices[model_name] = cm

        # 获取排序得分：优先 predict_proba，否则用 decision_function
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        else:
            y_score = model.decision_function(X_test)
            if y_score.ndim == 1:  # 二分类时补成两列
                y_score = np.column_stack([-y_score, y_score])
        # 将得分列顺序对齐到统一的 class_order
        col_idx = [list(model.classes_).index(c) for c in class_order]
        y_score = y_score[:, col_idx]

        fpr, tpr = {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], y_score[:, i])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes

        roc_data[model_name] = (all_fpr, mean_tpr, auc(all_fpr, mean_tpr))

    return results, confusion_matrices, roc_data


# ─────────────────────────────────────────────
# 5. 绘图辅助
# ─────────────────────────────────────────────
def save_svg(fig, output_dir, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, format="svg", transparent=True,
                bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.close(fig)
    return path


def plot_benchmark(df_results, output_dir, model_colors):
    print("\n🎨 正在生成性能柱状图...")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.barplot(data=df_results, x="Model", y="Score", hue="Metric",
                palette=METRIC_PALETTE, edgecolor="black", linewidth=1.2, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', label_type='edge', padding=4,
                     fontsize=11, fontweight='bold')
    ax.set_title("Benchmarking Machine Learning Classifiers on Multiplexed SERS Data",
                 fontsize=17, fontweight='bold', pad=20)
    ax.set_ylabel("Performance Score", fontsize=14, fontweight='bold', labelpad=10)
    ax.set_xlabel("Algorithm Architecture", fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylim(0, 1.15)
    ax.legend(title="", fontsize=12, loc='upper right', framealpha=0.9, edgecolor="black")
    ax.grid(False)
    plt.xticks(fontsize=12, fontweight='bold')
    return save_svg(fig, output_dir, "ppt_01_SERS_model_benchmark_nogrid.svg")


def plot_confusion_matrices(confusion_matrices, class_order, output_dir):
    print("🎨 正在拼装混淆矩阵全家福...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    for ax, (model_name, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_order, yticklabels=class_order,
                    cbar=False, annot_kws={"size": 12, "weight": "bold"}, ax=ax)
        ax.patch.set_alpha(0)
        ax.set_title(model_name, fontsize=15, fontweight='bold', pad=12)
        ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
        if ax is axes[0]:
            ax.set_ylabel('True Phenotype', fontsize=13, fontweight='bold')
    for ax in axes[len(confusion_matrices):]:
        ax.axis('off')
    plt.tight_layout()
    return save_svg(fig, output_dir, "ppt_02_confusion_matrices_ALL.svg")


def plot_roc(roc_data, output_dir, model_colors):
    print("🎨 正在生成多模型 ROC 曲线图...")
    fig, ax = plt.subplots(figsize=(8, 7))
    line_styles = ['-', '--', '-.', ':', '-']
    for i, (model_name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
        style = line_styles[i % len(line_styles)]
        ax.plot(fpr, tpr, color=model_colors[i % len(model_colors)],
                linestyle=style, linewidth=2.5,
                label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    ax.set_title("Multi-class ROC Curves (Macro-average)", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight='bold')
    ax.set_ylabel("True Positive Rate", fontsize=14, fontweight='bold')
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(False)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9, edgecolor="black")
    return save_svg(fig, output_dir, "ppt_03_ROC_Curves_comparison.svg")


# ─────────────────────────────────────────────
# 6. 主流程
# ─────────────────────────────────────────────
def main():
    data_dirs = CONFIG["data_dirs"]
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "figure.dpi": 300})

    class_order = list(data_dirs.keys())  # SHORT_LABELS 由字典派生，避免顺序错位

    print("🚀 正在加载表面增强拉曼散射 (SERS) 光谱矩阵数据...")
    X_all, y_all, wavenumbers = load_spectra(data_dirs)
    print(f"✅ 数据合并完毕！共 {X_all.shape[0]} 条光谱，{len(class_order)} 个类别。\n")

    print("⚔️ 正在划分数据集 (80% 训练, 20% 测试)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"], stratify=y_all)

    models = build_models(CONFIG["random_state"], CONFIG["mlp_max_iter"])

    print("🔥 开始训练多模型并生成评估数据...")
    results, confusion_matrices, roc_data = evaluate_models(
        models, X_train, X_test, y_train, y_test, class_order)
    df_results = pd.DataFrame(results)

    p1 = plot_benchmark(df_results, output_dir, MODEL_COLORS)
    p2 = plot_confusion_matrices(confusion_matrices, class_order, output_dir)
    p3 = plot_roc(roc_data, output_dir, MODEL_COLORS)

    print(f"\n🎉 全部出图完成！你的终极 PPT 素材已准备就绪：")
    print(f"  👉 1. 柱状图 (无网格): {os.path.basename(p1)}")
    print(f"  👉 2. 混淆矩阵 (透明): {os.path.basename(p2)}")
    print(f"  👉 3. ROC曲线 (含AUC): {os.path.basename(p3)}")


if __name__ == "__main__":
    main()
