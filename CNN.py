import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split # 导入随机切分工具

# ==========================================
# 第一部分：全自动数据管家 (智能读取与对齐)
# ==========================================
def load_real_data_auto(data_dir):
    print(f"🔍 正在从 {data_dir} 加载真实数据，准备进行全盘扫描...")
    
    raw_data_list = []
    y_list = []
    
    sub_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    sub_folders.sort() 
    
    if len(sub_folders) == 0:
        raise ValueError(f"❌ 错误：在 {data_dir} 里没有找到分类文件夹！")
        
    # 第一步：把所有数据先读进内存
    for label, folder_name in enumerate(sub_folders):
        folder_path = os.path.join(data_dir, folder_name)
        print(f"📁 -> 正在读取类别: {folder_name} (标签 {label})")
        
        for filename in os.listdir(folder_path):
            if not filename.endswith('.txt'):
                continue 
            file_path = os.path.join(folder_path, filename)
            try:
                # 跳过第一行英文表头
                data = np.loadtxt(file_path, skiprows=1)
                
                # 提取信号强度 (假设是第二列)
                if data.ndim == 2 and data.shape[1] >= 2:
                    intensity = data[:, 1]
                else:
                    intensity = data 
                    
                raw_data_list.append(intensity)
                y_list.append(label)
            except Exception as e:
                print(f"⚠️ 警告: 读取 {filename} 时出错跳过 -> {e}")
                
    if len(raw_data_list) == 0:
        raise ValueError("❌ 没有成功读取到任何有效数据，请检查 txt 文件格式！")

    # 第二步：计算 95% 分位数作为统一标准长度
    all_lengths = [len(data) for data in raw_data_list]
    target_length = int(np.percentile(all_lengths, 95))
    
    print(f"\n✨ [智能探测完毕] 扫描了 {len(all_lengths)} 个文件。")
    print(f"   📏 最短: {min(all_lengths)}, 最长: {max(all_lengths)}")
    print(f"   🎯 选取 95% 标准长度为: {target_length} 个点！\n")

    # 第三步：插值对齐和归一化
    X_list = []
    for intensity in raw_data_list:
        if len(intensity) != target_length:
            f = interpolate.interp1d(np.linspace(0, 1, len(intensity)), intensity, kind='linear')
            intensity_aligned = f(np.linspace(0, 1, target_length))
        else:
            intensity_aligned = intensity
            
        # Min-Max 归一化 (拉伸到 0~1 之间)
        min_val = np.min(intensity_aligned)
        max_val = np.max(intensity_aligned)
        if max_val - min_val > 0:
            intensity_norm = (intensity_aligned - min_val) / (max_val - min_val)
        else:
            intensity_norm = intensity_aligned
            
        X_list.append(intensity_norm)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    print(f"✅ 数据集彻底准备完毕！统一尺寸为 {X.shape}。")
    return X, y, target_length

# ==========================================
# 第二部分：AI 大脑 (1D-CNN 模型架构)
# ==========================================
class RamanNet(nn.Module):
    def __init__(self, input_len, num_classes=2):
        super(RamanNet, self).__init__()
        
        # 1. 找特征的部门 (卷积层)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=15, padding=7), # 找浅层特征
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(8, 16, kernel_size=15, padding=7), # 找深层特征
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(10) # 强制浓缩成每个通道10个数据点
        )
        
        # 2. 做法官的部门 (全连接层)
        self.classifier = nn.Sequential(
            nn.Linear(16 * 10, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) # 增加通道维度
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1) # 展平
        out = self.classifier(features)
        return out

# ==========================================
# 第三部分：主训练循环 (刷题机器)
# ==========================================
def train():
    X, y, detected_length = load_real_data_auto(data_dir="./my_dataset")
    
    # 1. 整体打包数据
    full_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    
    # ==========================================
    # 【新增逻辑】：8:2 随机切分训练集和测试集
    # ==========================================
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size) # 80% 用来训练
    test_size = total_size - train_size # 20% 用来考试
    
    # 让 PyTorch 帮我们随机打乱并切开
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # 分别装进两个 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) # 考试不需要打乱顺序
    
    print(f"📦 数据切分完毕：训练集 {train_size} 个，测试集 {test_size} 个")

    # 修改后：告诉大脑现在有 3 个类别
    model = RamanNet(input_len=detected_length, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # 记录本
    history_train_loss, history_train_acc = [], []
    history_test_acc = [] # 新增：记录真实考试成绩
    
    epochs = 40
    print("\n🚀 开始训练与独立测试...")
      
    for epoch in range(epochs):
        # ---------------------------
        # 阶段 A：做练习题 (Training)
        # ---------------------------
        model.train() # 告诉模型：现在是学习时间
        correct_train = 0
        total_train = 0
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()            
            outputs = model(batch_X)         
            loss = criterion(outputs, batch_y) 
            loss.backward()                  # 总结错误
            optimizer.step()                 # 修改参数！(只有训练阶段才能做这一步)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # ---------------------------
        # 阶段 B：期末考试 (Testing)
        # ---------------------------
        model.eval() # 告诉模型：现在是考试时间，不许修改参数！
        correct_test = 0
        total_test = 0
        
        with torch.no_grad(): # 考试时不需要算梯度，省内存
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total_test += batch_y.size(0)
                correct_test += (predicted == batch_y).sum().item()
                
        test_acc = 100 * correct_test / total_test
        
        # 登记成绩
        history_train_loss.append(train_loss)
        history_train_acc.append(train_acc)
        history_test_acc.append(test_acc)
        
        print(f"第 {epoch+1:02d} 轮 | 训练误差: {train_loss:.4f} | 训练准确率: {train_acc:.2f}% | ⚠️ 测试准确率: {test_acc:.2f}%")

    # 画出对比图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    
    plt.plot(history_train_acc, color='blue', label='Train Accuracy (练习成绩)')
    plt.plot(history_test_acc, color='green', label='Test Accuracy (真实考试成绩)', linestyle='--')
    plt.title("Train vs Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    train()