import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ==========================================
# 第一部分：准备模拟模型和数据 (Mock Setup)
# ==========================================

# 这是一个简单的 1D 卷积神经网络，用来模拟论文中的分类器
class SimpleRamanModel(nn.Module):
    def __init__(self, input_length=1000, num_classes=2):
        super(SimpleRamanModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * input_length, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 输入格式调整: (Batch, Channel, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1) # 展平
        x = self.fc(x)
        return self.softmax(x)

# 造一条假的拉曼光谱数据
# 模拟几个高斯峰作为特征
def create_mock_spectrum(length=1000):
    x = np.linspace(0, 100, length)
    # 制造三个特征峰 (模拟论文中的关键区域)
    y = np.exp(-((x - 30)**2) / 2) + 0.5 * np.exp(-((x - 70)**2) / 2) + 0.8 * np.exp(-((x - 50)**2) / 4)
    # 加一点噪声
    noise = np.random.normal(0, 0.02, length)
    return x, y + noise

# ==========================================
# 第二部分：GRASS 核心算法实现
# ==========================================

# 1. 梯度分析 (Gradient Analysis) [cite: 288]
def get_gradients(model, spectrum_tensor, target_class_index):
    """
    计算输入光谱相对于预测概率的梯度。
    """
    # 关键：开启梯度追踪
    spectrum_tensor.requires_grad = True
    
    # 前向传播
    output = model(spectrum_tensor)
    
    # 获取目标类别的概率
    target_prob = output[0, target_class_index]
    
    # 反向传播：计算梯度
    # 这一步会计算 d(概率)/d(输入)
    target_prob.backward()
    
    # 取出梯度并取绝对值 (论文中用的是梯度的幅度 [cite: 293])
    gradients = spectrum_tensor.grad.detach().numpy().flatten()
    return np.abs(gradients)

# 2. 光谱分割 (Spectral Segmentation) [cite: 297, 298, 301]
def segment_spectrum(gradients, threshold=1e-5):
    """
    根据梯度曲线的谷底将光谱切分为不同区域。
    """
    # 平滑处理：论文提到要先平滑去除小波动 [cite: 297]
    # window_length必须是奇数，polyorder是多项式阶数
    smoothed_grad = savgol_filter(gradients, window_length=31, polyorder=3)
    
    regions = []
    last_cut = 0
    length = len(gradients)
    
    # 寻找切分点：这里简化逻辑，寻找梯度非常小且是一阶导数变号的点
    # 或者简单点：只要梯度低于阈值，并且距离上一个切分点有一定距离，就切一刀
    
    for i in range(10, length - 10):
        # 逻辑：如果当前点梯度极小，且局部是谷底
        if smoothed_grad[i] < threshold:
            # 这里的逻辑是简化版，原论文利用了一阶导数找零点 
            # 我们强制要求两个切点之间至少隔开一些距离，避免切太碎
            if i - last_cut > 20: 
                regions.append((last_cut, i))
                last_cut = i
    
    # 加上最后一段
    regions.append((last_cut, length))
    
    return regions

# 3. SHAP 值计算 (SHAP Calculation) [cite: 468, 472]
def calculate_grass_shap(model, spectrum_array, regions, target_class, num_samples=300):
    """
    使用蒙特卡洛采样计算每个区域的 SHAP 贡献值。
    """
    contributions = []
    base_spectrum = torch.tensor(spectrum_array, dtype=torch.float32).unsqueeze(0)
    
    print(f"开始计算 SHAP 值，共 {len(regions)} 个区域，采样 {num_samples} 次...")
    
    for idx, (start, end) in enumerate(regions):
        shap_value = 0
        
        # 蒙特卡洛循环
        for _ in range(num_samples):
            # 1. 随机生成一个掩码 (Mask)，决定哪些“其他区域”要被遮挡
            # True 表示保留，False 表示遮挡
            # 我们关注的是当前区域 R_i (即 regions[idx]) 的边缘贡献
            
            # 生成一个随机的组合 S (不包含当前区域 R_i)
            feature_indices = list(range(len(regions)))
            feature_indices.remove(idx) # 移除自己
            
            # 随机决定其他区域是否开启 (50% 概率)
            subset_mask = np.random.choice([True, False], size=len(feature_indices))
            
            # 构造两个输入样本：
            # sample_with:  包含当前区域 R_i + 随机组合 S
            # sample_without: 不包含当前区域 R_i + 随机组合 S
            
            input_with = spectrum_array.copy()
            input_without = spectrum_array.copy()
            
            # 填充随机噪声的函数 (模拟 masking )
            def fill_noise(arr, s, e):
                arr[s:e] = np.random.normal(0, 0.01, e-s)

            # 构建 "With" 和 "Without"
            # 先把当前区域处理好
            # With: 保持原样 (不做操作)
            # Without: 填噪音
            fill_noise(input_without, start, end)
            
            # 处理其他区域 (S)
            for i, keep in zip(feature_indices, subset_mask):
                r_s, r_e = regions[i]
                if not keep:
                    # 如果这个区域不在组合里，两个样本都要把它遮住
                    fill_noise(input_with, r_s, r_e)
                    fill_noise(input_without, r_s, r_e)
            
            # 转为 Tensor 喂给模型
            tensor_with = torch.tensor(input_with, dtype=torch.float32).unsqueeze(0)
            tensor_without = torch.tensor(input_without, dtype=torch.float32).unsqueeze(0)
            
            # 获取预测概率
            with torch.no_grad():
                prob_with = model(tensor_with)[0, target_class].item()
                prob_without = model(tensor_without)[0, target_class].item()
            
            # 累加边缘贡献 [cite: 472]
            shap_value += (prob_with - prob_without)
            
        # 取平均 [cite: 476]
        avg_shap = shap_value / num_samples
        contributions.append(avg_shap)
        # print(f"区域 {idx} ({start}-{end}) 完成。贡献值: {avg_shap:.4f}")
        
    return contributions

# ==========================================
# 第三部分：主程序运行 (Main Execution)
# ==========================================

def main():
    # 1. 初始化
    input_len = 500
    model = SimpleRamanModel(input_length=input_len)
    model.eval() # 设为评估模式
    
    # 2. 搞点假数据
    _, spectrum_data = create_mock_spectrum(input_len)
    spectrum_tensor = torch.tensor(spectrum_data, dtype=torch.float32).unsqueeze(0)
    
    # 假设我们预测类别 0
    target_class = 0
    
    print("Step 1: 计算梯度...")
    grads = get_gradients(model, spectrum_tensor, target_class)
    
    print("Step 2: 分割光谱...")
    # 这里阈值设大一点是为了在假数据上演示效果，实际要调小
    regions = segment_spectrum(grads, threshold=0.0001)
    print(f"光谱被切分成了 {len(regions)} 个区域。")
    
    print("Step 3: 计算 GRASS SHAP 值 (可能需要一点时间)...")
    # 采样次数设为 50 以便快速演示，论文建议 300-1000 [cite: 750, 751]
    shap_values = calculate_grass_shap(model, spectrum_data, regions, target_class, num_samples=50)
    
    # 4. 可视化结果
    print("Step 4: 画图...")
    plt.figure(figsize=(12, 6))
    
    # 绘制原始光谱
    plt.plot(spectrum_data, label='Raw Spectrum', color='black', alpha=0.6)
    
    # 给每个区域上色，颜色的深浅代表 SHAP 值的大小
    # 归一化 SHAP 值以便绘图 (映射到 0-1 之间)
    max_shap = max(np.abs(shap_values)) if max(np.abs(shap_values)) > 0 else 1
    
    for i, (start, end) in enumerate(regions):
        val = shap_values[i]
        # 绿色代表正贡献，红色代表负贡献 (类似论文 Fig 8 [cite: 392])
        color = 'green' if val > 0 else 'red'
        alpha = abs(val) / max_shap * 0.8 # 透明度代表重要性
        
        plt.axvspan(start, end, color=color, alpha=alpha)
        # 在区域上方标注贡献值
        mid = (start + end) / 2
        plt.text(mid, max(spectrum_data)*1.1, f"{val:.2f}", ha='center', fontsize=8)

    plt.title("GRASS Analysis Result (Green=Positive, Red=Negative)")
    plt.xlabel("Raman Shift (Indices)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()