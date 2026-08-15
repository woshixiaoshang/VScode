import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

# ============================================================
# 读取target.txt
# ============================================================
target_path = r"C:\Users\商琨珏\Desktop\target.txt"

save_dir = r"D:\aaaSCNU\0data\Amyloid王惊华\联合分析\PB"

group_files = {
    "正常": r"D:\aaaSCNU\0data\Amyloid王惊华\联合分析\PB\HD-PBglobal_average.txt",
    "AL前": r"D:\aaaSCNU\0data\Amyloid王惊华\联合分析\PB\AL前PBglobal_average.txt"
}

# ============================================================
# 读取每组平均谱
# ============================================================
spectra = {}
for label, path in group_files.items():
    data = np.loadtxt(path)
    spectra[label] = {'wavenumber': data[:, 0], 'intensity': data[:, 1]}
    print(f"{label} 读取成功，共 {len(data)} 个数据点")
    print(f"  波数范围：{data[:, 0].min():.1f} ~ {data[:, 0].max():.1f} cm⁻¹")

# ============================================================
# 全谱对比图
# ============================================================
colors = {'正常': '#378ADD', 'AL前': '#E24B4A'}

fig, ax = plt.subplots(figsize=(12, 5))
for label, data in spectra.items():
    ax.plot(data['wavenumber'], data['intensity'],
            label=label, color=colors[label], linewidth=1.2)

ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
ax.set_ylabel('Intensity (a.u.)', fontsize=12)
ax.set_title('Average Raman Spectra', fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
save_path1 = os.path.join(save_dir, '全谱对比.svg')
plt.savefig(save_path1, )
plt.show()
print(f"\n全谱对比图已保存：{save_path1}")

# ============================================================
# 940-1100 cm⁻¹段放大图
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
peak_data = {}

for label, data in spectra.items():
    wn = data['wavenumber']
    intensity = data['intensity']
    mask = (wn >= 940) & (wn <= 1100)
    wn_seg = wn[mask]
    int_seg = intensity[mask]
    ax.plot(wn_seg, int_seg, label=label, color=colors[label], linewidth=1.5)
    peak_data[label] = {'wavenumber': wn_seg, 'intensity': int_seg}

ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
ax.set_ylabel('Intensity (a.u.)', fontsize=12)
ax.set_title('940–1100 cm⁻¹ Region', fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
save_path2 = os.path.join(save_dir, '940_1100段对比.svg')
plt.savefig(save_path2, )
plt.show()
print(f"940-1100段对比图已保存：{save_path2}")

# ============================================================
# 峰位检测
# ============================================================
print("\n========== 峰位检测结果 ==========")
for label, data in peak_data.items():
    wn = data['wavenumber']
    intensity = data['intensity']
    peaks, _ = find_peaks(intensity, prominence=0.001, distance=5)
    print(f"\n{label} 检测到的峰位：")
    for p in peaks:
        print(f"  {wn[p]:.1f} cm⁻¹  强度: {intensity[p]:.4f}")