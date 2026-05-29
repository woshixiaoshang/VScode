import torch
import torch.nn as nn
import numpy as np
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 还是那个简单的模型和造假数据
# ==========================================
class SimpleRamanModel(nn.Module):
    def __init__(self, input_length=1000, num_classes=2):
        super(SimpleRamanModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * input_length, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.softmax(x)

def create_mock_spectrum(length=1000):
    x = np.linspace(0, 100, length)
    # 制造特征峰
    y = np.exp(-((x - 30)**2) / 2) + 0.5 * np.exp(-((x - 70)**2) / 2) + 0.8 * np.exp(-((x - 50)**2) / 4)
    noise = np.random.normal(0, 0.02, length)
    return x, y + noise

# ==========================================
# 2. 传统 SHAP 分析主程序
# ==========================================
def main():
    # --- 准备模型 ---
    input_len = 500
    model = SimpleRamanModel(input_length=input_len)
    model.eval()

    # --- 准备数据 ---
    # 为了运行 SHAP，我们需要两组数据：
    # 1. 背景数据 (Background): 用来告诉 SHAP "什么都不是" 长什么样（作为基准）
    # 2. 测试数据 (Test): 我们要解释的那条光谱
    
    # 造 100 条背景数据（全是噪声，模拟无信号状态）
    background_data = torch.randn(100, input_len) 
    
    # 造 1 条真实的测试数据（有峰的）
    _, test_numpy = create_mock_spectrum(input_len)
    test_data = torch.tensor(test_numpy, dtype=torch.float32).unsqueeze(0)

    print("正在使用 DeepExplainer 计算传统 SHAP 值...")
    print("这一步是把 500 个点每一个都当成独立特征来算，可能会有点慢...")

    # --- 核心：使用 shap 库 ---
    # DeepExplainer 是专门给深度学习用的（支持 PyTorch）
    explainer = shap.DeepExplainer(model, background_data)
    
    # 计算 SHAP 值
    # 这会返回一个列表，对应每个类别的 SHAP 值
    # 我们看类别 0 的贡献
    shap_values = explainer.shap_values(test_data)
    
    # shap_values 的结构通常是 [类别0的shap, 类别1的shap]
    # 我们取类别 0 的部分，它的大小是 (1, 500)
    # 注意：新版 shap 可能会返回 numpy 数组，或者 list
    if isinstance(shap_values, list):
        target_shap = shap_values[0] # 取类别 0
    else:
        target_shap = shap_values[:, :, 0] # 视版本而定，暂取一种常见情况

    # 把它变成一维数组方便画图
    target_shap = target_shap.squeeze()

    # ==========================================
    # 3. 可视化对比
    # ==========================================
    print("计算完成，正在画图...")
    
    plt.figure(figsize=(12, 8))

    # 子图 1: 原始光谱
    plt.subplot(2, 1, 1)
    plt.plot(test_numpy, color='black', label='Raw Spectrum')
    plt.title("Original Raman Spectrum")
    plt.ylabel("Intensity")
    plt.legend()

    # 子图 2: 传统 SHAP 值 (每个点的贡献)
    plt.subplot(2, 1, 2)
    
    # 还是用红绿配色：红=正贡献，蓝=负贡献
    plt.plot(target_shap, color='blue', linewidth=1, label='SHAP Value (Per Point)')
    
    # 把正贡献涂成红色，负贡献涂成蓝色
    plt.fill_between(range(input_len), target_shap, 0, where=(target_shap > 0), facecolor='red', alpha=0.5, label='Positive Impact')
    plt.fill_between(range(input_len), target_shap, 0, where=(target_shap < 0), facecolor='blue', alpha=0.5, label='Negative Impact')
    
    plt.title("Traditional Pure SHAP Values (Point-wise)")
    plt.xlabel("Wavenumber / Index")
    plt.ylabel("SHAP Value")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()