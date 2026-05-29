import os
import math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

folder = r"D:\aaaSCNU\0data\Amyloid王惊华\process\BM\HD-BM"

spectra = []
raw_spectra = []
wavenumber = None
expected_length = None  # 用于记录基准点数，防止长短不一的数据报错

print(f"正在扫描并读取数据，路径: {folder}")

# 🚀 核心修改：使用 os.walk 自动递归遍历所有目录和子目录
for root_dir, dirs, files in os.walk(folder):
    for file in files:
        if file.lower().endswith(".txt"):
            path = os.path.join(root_dir, file)
            try:
                data = np.loadtxt(path)
                
                # 确保读取到的是二维数据
                if data.ndim >= 2:
                    current_x = data[:, 0]
                    current_y = data[:, 1]
                    raw_spectra.append((path, current_x, current_y))
            except Exception as e:
                print(f"⚠️ 读取文件 {path} 时发生错误跳过: {e}")

# ===== 以 95% 以上的数据长度为基准 =====
total_files = len(raw_spectra)
if total_files == 0:
    print("❌ 未找到任何有效的 .txt 文件，程序终止。")
    exit()

length_counts = Counter(len(current_x) for _, current_x, _ in raw_spectra)
most_common_length, most_common_count = length_counts.most_common(1)[0]
threshold = math.ceil(total_files * 0.95)
expected_length = most_common_length

if most_common_count < threshold:
    print(f"⚠️ 没有长度达到95%阈值，使用最常见长度 {expected_length} 点作为基准 ({most_common_count}/{total_files})")
else:
    print(f"✅ 基准长度已选为 {expected_length} 点，覆盖 {most_common_count}/{total_files} 个文件")

for path, current_x, current_y in raw_spectra:
    if len(current_x) == expected_length:
        if wavenumber is None:
            wavenumber = current_x
        spectra.append(current_y)
    else:
        print(f"⚠️ 跳过长度不匹配的文件: {path} (当前: {len(current_x)}点, 基准: {expected_length}点)")

print(f"\n✅ 扫描结束，共提取到 {len(spectra)} 条有效光谱")

# ===== 🛡️ 安全拦截机制 =====
if len(spectra) == 0:
    print("❌ 严重错误：未读取到任何数据！")
    print("可能原因：")
    print("  1. 文件夹路径写错了")
    print("  2. 该路径及其所有子文件夹下，没有任何 .txt 文件")
    print("  3. txt 文件内容格式不正确（不是纯数字两列）")
    exit()  # 直接终止程序，不往下执行画图

spectra = np.array(spectra)

print("正在计算统计量...")
print("数据矩阵维度:", spectra.shape)

# 计算统计量
mean_spectrum = np.mean(spectra, axis=0)
variance_spectrum = np.var(spectra, axis=0)
std_spectrum = np.std(spectra, axis=0)

# ===== 保存统计数据 =====

output_data = np.column_stack((
    wavenumber,
    mean_spectrum,
    variance_spectrum,
    std_spectrum
))

# 无论数据藏在多深的子文件夹里，统计结果都会统一保存在你指定的根文件夹下
save_path = os.path.join(folder, "2800-3000_mean_variance_std.txt")

np.savetxt(
    save_path,
    output_data,
    header="Wavenumber Mean Variance Std",
    fmt="%.6f"
)

print(f"💾 统计数据已保存: {save_path}")

# ===== 绘图 =====

plt.figure(figsize=(8,5))

plt.plot(wavenumber, mean_spectrum, label="Mean spectrum")

plt.fill_between(
    wavenumber,
    mean_spectrum - std_spectrum,
    mean_spectrum + std_spectrum,
    alpha=0.3,
    label="±1 STD"
)

plt.xlabel("Raman Shift (cm$^{-1}$)")
plt.ylabel("Intensity")
plt.legend()
plt.xlim(2800, 3000)
plt.tight_layout()

# ===== 保存图片 =====

fig_path = os.path.join(folder, "mean_spectrum.svg")
plt.savefig(fig_path, dpi=300)

print(f"📸 图片已保存: {fig_path}")

# plt.show() # 如果你不想弹窗，可以把这行注释掉