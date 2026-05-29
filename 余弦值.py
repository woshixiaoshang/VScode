import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from scipy.interpolate import interp1d
from collections import Counter

# ================= 1. 读取目标文件夹路径 =================
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
config_file = os.path.join(desktop_path, "target.txt")

try:
    with open(config_file, 'r', encoding='utf-8') as f:
        data_folder = f.read().strip().strip('"').strip("'")
    print(f"✅ 目标文件夹: {data_folder}")
except Exception as e:
    print(f"❌ 无法读取桌面 target.txt，错误: {e}")
    exit()

if not os.path.isdir(data_folder):
    print("❌ 目标路径不存在，请检查 target.txt！")
    exit()

# 输出文件夹设为数据文件夹
output_folder = data_folder
print(f"📁 输出文件夹: {output_folder}")

# ================= 2. 提取数据并提取特征 =================
raw_data_list = []
print("\n--- 正在扫描并读取光谱文件 ---")

for root_dir, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(root_dir, file)
            try:
                data = np.loadtxt(path)
                # 兼容 2列 或 3列 数据
                if data.ndim >= 2:
                    x_values = data[:, 0] # 第一列: 拉曼位移
                    y_values = data[:, 1] # 第二列: 强度
                    raw_data_list.append({
                        'filename': file, 
                        'x': x_values, 
                        'y': y_values
                    })
            except Exception:
                continue

if not raw_data_list:
    print("❌ 文件夹中未找到有效的光谱 txt 文件！")
    exit()

# ================= 3. 智能对齐 (解决坐标微小偏移) =================
print("\n--- 正在执行智能坐标对齐 ---")
# 找出最主流的光谱长度和首尾坐标
signatures = [(len(item['x']), round(item['x'][0], -1), round(item['x'][-1], -1)) for item in raw_data_list]
majority_sig = Counter(signatures).most_common(1)[0][0]

# 提取基准 X 轴
ref_x = next(item['x'] for item, sig in zip(raw_data_list, signatures) if sig == majority_sig)

valid_spectra = []
repaired_count = 0
tolerance = 10.0 # 允许 10 cm-1 的首尾偏差

for item in raw_data_list:
    start_diff = abs(item['x'][0] - ref_x[0])
    end_diff = abs(item['x'][-1] - ref_x[-1])

    if start_diff <= tolerance and end_diff <= tolerance:
        if np.array_equal(item['x'], ref_x):
            valid_spectra.append(item['y'])
        else:
            # 使用 scipy 线性插值，将微小偏移的 Y 值对齐到标准 X 轴
            f_interp = interp1d(item['x'], item['y'], kind='linear', bounds_error=False, fill_value="extrapolate")
            aligned_y = f_interp(ref_x)
            valid_spectra.append(aligned_y)
            repaired_count += 1

spectra_matrix = np.array(valid_spectra)
print(f"✅ 成功对齐并提取 {len(spectra_matrix)} 条光谱！(其中自动修复了 {repaired_count} 条漂移光谱)")

# ================= 4. 计算异质性 (余弦距离) =================
print("\n--- 正在计算异质性矩阵 ---")
distance_matrix = cosine_distances(spectra_matrix)

# ================= 5. 绘制热图 =================
print("\n--- 正在绘制热图 ---")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

plt.figure(figsize=(10, 8))

# 绘制热图 (距离越大，颜色越深，越异质)
heatmap = plt.imshow(distance_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

cbar = plt.colorbar(heatmap)
cbar.set_label('Cosine Distance (Heterogeneity)', rotation=270, labelpad=20)

plt.title('Raman Spectra Heterogeneity Matrix')
plt.xlabel('Spectrum Sample Index')
plt.ylabel('Spectrum Sample Index')
plt.tight_layout()

# ================= 6. 保存结果和图表 =================
print("\n--- 正在保存结果和图表 ---")

# 保存距离矩阵
matrix_path = os.path.join(output_folder, "heterogeneity_matrix.txt")
np.savetxt(matrix_path, distance_matrix, delimiter="\t", fmt="%.6f", header="Heterogeneity Matrix (Cosine Distance)")
print(f"💾 异质性矩阵已保存: {matrix_path}")

# 保存图表
plot_path = os.path.join(output_folder, "heterogeneity_heatmap.svg")
plt.savefig(plot_path, format='svg', dpi=300, bbox_inches='tight')
print(f"💾 高清 SVG 图表已保存: {plot_path}")

print(f"\n🎉 任务完成！结果已保存至: {output_folder}")
plt.show()