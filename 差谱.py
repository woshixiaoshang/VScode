import os
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===========================
# 数据目录
# ===========================
data_root = r"E:\LYB1\average"

# 输出目录
output_dir = os.path.join(data_root, "DifferenceSpectrum")
os.makedirs(output_dir, exist_ok=True)

# ===========================
# 读取一个组所有光谱
# ===========================
def load_group(folder):

    files = sorted(glob.glob(os.path.join(folder, "*.txt")))

    spectra = []

    for file in files:
        data = np.loadtxt(file)

        shift = data[:,0]
        intensity = data[:,1]

        spectra.append(intensity)

    spectra = np.array(spectra)

    mean_spec = np.mean(spectra, axis=0)

    return shift, spectra, mean_spec

# ===========================
# 所有组
# ===========================
group_dirs = sorted([
    d for d in os.listdir(data_root)
    if os.path.isdir(os.path.join(data_root, d))
    and d != "DifferenceSpectrum"
])

print("Groups:")
print(group_dirs)

group_mean = {}

# ===========================
# 求平均谱
# ===========================
for g in group_dirs:

    folder = os.path.join(data_root, g)

    shift, spectra, mean_spec = load_group(folder)

    group_mean[g] = mean_spec

    save = pd.DataFrame({
        "Shift": shift,
        "MeanIntensity": mean_spec
    })

    save.to_csv(
        os.path.join(output_dir, f"{g}_MeanSpectrum.csv"),
        index=False
    )

# ===========================
# 两两差谱
# ===========================
pairs = list(itertools.combinations(group_dirs, 2))

for g1, g2 in pairs:

    diff = group_mean[g1] - group_mean[g2]

    save = pd.DataFrame({
        "Shift": shift,
        "Difference": diff
    })

    save.to_csv(
        os.path.join(output_dir,
                     f"{g1}_minus_{g2}.csv"),
        index=False
    )

    plt.figure(figsize=(8,4))

    plt.plot(shift, diff, lw=2)

    plt.axhline(0,
                color="black",
                ls="--",
                alpha=0.6)

    plt.xlabel("Raman Shift (cm$^{-1}$)")
    plt.ylabel("Intensity Difference")

    plt.title(f"{g1} - {g2}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir,
                     f"{g1}_minus_{g2}.png"),
        dpi=300
    )

    plt.close()

print("Finished.")