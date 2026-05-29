import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 假设你有 9 条光谱数据 (这里用随机数模拟一下)
# 实际就是读取你的 txt 文件，然后把它们叠在一起
# 比如 3组样品，每组3个重复，共9行
data = np.random.rand(9, 1000) 

# 给每行起个名字 (和论文里一样)
names = [
    "P2-S1", "P2-S2", "P2-S3", 
    "P6-S1", "P6-S2", "P6-S3", 
    "P10-S1", "P10-S2", "P10-S3"
]

# 2. 核心步骤：直接计算相关系数矩阵
# pandas 的 .T 是转置，因为我们需要计算"行"（样品）之间的相关性
df = pd.DataFrame(data.T, columns=names)
corr_matrix = df.corr(method='pearson')  # 这一步就把 PCC 矩阵算出来了

# 3. 画热力图
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, 
            annot=False,      # 如果格子大，设为True可以在格子里显示具体数字
            cmap='hot',       # 配色：'hot' 就是红黄黑风格，'viridis' 是蓝绿黄
            vmin=0.8, vmax=1, # 设置颜色范围，通常SERS相似度都很高，所以下限设高点(比如0.8)才能看出区别
            square=True)      # 让格子是正方形

plt.title("Pearson's Correlation Coefficient Matrix")
plt.show()