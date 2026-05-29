import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
from collections import Counter
import pywt
import warnings
import traceback
from tqdm import tqdm

plt.switch_backend('agg')
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ============================================================
# 高波数区域配置
# 2800~3000 cm⁻¹：脂质 CH₂/CH₃ 伸缩振动区
# 不做基线校正和归一化，参数一致，强度可直接比较
# ============================================================
WAVENUMBER_MIN = 2800   # 截取起始波数 cm⁻¹
WAVENUMBER_MAX = 3000   # 截取终止波数 cm⁻¹


class RamanHighWavenumberProcessor:
    """高波数区域拉曼光谱处理（2800~3000 cm⁻¹，仅去噪，不做基线和归一化）"""

    def __init__(self):
        self.spectra_data = None
        self.file_names = None
        self.wavenumber_data = None
        self.header = ""
        self.processed_data = None
        self.processed_filenames = None

    # ──────────────────────────────────────────────────────────
    # 数据加载
    # ──────────────────────────────────────────────────────────
    def load_spectra_data(self, folder_path, has_subfolders=False):
        print(f"正在扫描并加载数据从: {folder_path}")
        file_paths = []

        if has_subfolders:
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    for fname in os.listdir(subfolder_path):
                        if fname.endswith(".txt"):
                            file_paths.append((os.path.join(subfolder_path, fname), f"{subfolder}/{fname}"))
        else:
            for fname in os.listdir(folder_path):
                if fname.endswith(".txt"):
                    file_paths.append((os.path.join(folder_path, fname), fname))

        if not file_paths:
            raise ValueError("❌ 未找到任何 .txt 光谱文件！")

        self.spectra_data, self.file_names, self.wavenumber_data, self.header = \
            self._load_and_align(file_paths)

        if self.spectra_data is None:
            raise ValueError("❌ 所有文件均读取失败或格式无效！")

        print(f"\n✅ 成功提取 {len(self.spectra_data)} 条有效光谱")
        print(f"   全谱波数范围: {self.wavenumber_data[0]:.2f} ~ {self.wavenumber_data[-1]:.2f} cm⁻¹")

        # 截取高波数区域
        self._crop_wavenumber(WAVENUMBER_MIN, WAVENUMBER_MAX)
        print(f"   截取后波数范围: {self.wavenumber_data[0]:.2f} ~ {self.wavenumber_data[-1]:.2f} cm⁻¹")
        print(f"   截取后数据维度: {self.spectra_data.shape}")

    def _load_and_align(self, file_paths_with_names, tolerance=10.0):
        raw_data_list = []
        for fpath, rel_name in file_paths_with_names:
            try:
                data = np.loadtxt(fpath)
                if data.ndim == 2 and data.shape[1] >= 2:
                    x_values = data[:, 0]
                    y_values = data[:, 1]
                    if x_values[0] > x_values[-1]:
                        x_values = x_values[::-1]
                        y_values = y_values[::-1]
                    raw_data_list.append({'filename': rel_name, 'x': x_values, 'y': y_values})
            except Exception:
                continue

        if not raw_data_list:
            return None, [], None, ""

        signatures = [(len(d['x']), round(d['x'][0], -1), round(d['x'][-1], -1)) for d in raw_data_list]
        majority_sig = Counter(signatures).most_common(1)[0][0]
        ref_x = next(d['x'] for d, sig in zip(raw_data_list, signatures) if sig == majority_sig)

        valid_spectra, valid_filenames = [], []
        repaired_count = 0
        for item in raw_data_list:
            if abs(item['x'][0] - ref_x[0]) <= tolerance and abs(item['x'][-1] - ref_x[-1]) <= tolerance:
                if np.array_equal(item['x'], ref_x):
                    valid_spectra.append(item['y'])
                else:
                    f_interp = interp1d(item['x'], item['y'], kind='linear', bounds_error=False, fill_value="extrapolate")
                    valid_spectra.append(f_interp(ref_x))
                    repaired_count += 1
                valid_filenames.append(item['filename'])

        if repaired_count > 0:
            print(f"🔧 已将 {repaired_count} 条微漂移光谱插值对齐至标准波数轴")

        return np.array(valid_spectra), valid_filenames, ref_x, "#Wave\t#Intensity"

    def _crop_wavenumber(self, wn_min, wn_max):
        """截取指定波数范围"""
        mask = (self.wavenumber_data >= wn_min) & (self.wavenumber_data <= wn_max)
        if mask.sum() == 0:
            raise ValueError(f"❌ 波数范围 {wn_min}~{wn_max} cm⁻¹ 在数据中不存在！")
        self.wavenumber_data = self.wavenumber_data[mask]
        self.spectra_data = self.spectra_data[:, mask]

    # ──────────────────────────────────────────────────────────
    # 去噪（小波 + SG，针对宽峰调参）
    # ──────────────────────────────────────────────────────────
    def _denoise(self, spectra):
        """
        针对2800~3000 cm⁻¹宽峰区域的去噪：
        - 小波层数限制在1~2，避免过度平滑宽峰
        - SG窗口适当放大，平滑噪声但保留峰形
        - 不做峰位检测（宽包峰find_peaks效果差），改用SNR提升量作为评分
        """
        if len(spectra) == 0:
            return spectra, (0, 0, 0, 0)

        wavelet = 'db8'
        level_candidates = [1, 2]           # 宽峰区域只用低层小波，防止峰形失真
        sg_windows = [7, 11, 15, 21, 27]    # 窗口稍大，匹配宽峰特征
        sg_poly = 3
        threshold_scales = [0.5, 0.75, 1.0]

        n_points = spectra.shape[1]
        max_level = pywt.dwt_max_level(n_points, pywt.Wavelet(wavelet).dec_len)
        level_candidates = [l for l in level_candidates if 0 < l <= max_level]

        def valid_window(w):
            w = min(int(w), n_points - 1 if n_points % 2 == 0 else n_points)
            if w % 2 == 0:
                w -= 1
            min_w = sg_poly + 2 if (sg_poly + 2) % 2 == 1 else sg_poly + 3
            return w if w >= min_w else None

        sg_windows = sorted({w for w in (valid_window(w) for w in sg_windows) if w is not None})

        def noise_sigma(y):
            diff = np.diff(y)
            return max(np.median(np.abs(diff - np.median(diff))) / (0.6745 * np.sqrt(2)), 1e-12)

        def robust_snr(y):
            sigma = noise_sigma(y)
            signal = np.percentile(y, 95) - np.percentile(y, 5)
            return 20 * np.log10((max(signal, np.std(y)) + 1e-12) / sigma)

        def wavelet_denoise(y, level, scale):
            coeffs = pywt.wavedec(y, wavelet=wavelet, level=level, mode='symmetric')
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = scale * sigma * np.sqrt(2 * np.log(len(y)))
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            return pywt.waverec(coeffs, wavelet, mode='symmetric')[:len(y)]

        def shape_loss(original, smoothed):
            """
            宽峰用形状相关性衡量失真度，而不是find_peaks
            corrcoef接近1说明峰形保持好
            """
            r = np.corrcoef(original, smoothed)[0, 1]
            return float(np.clip(1 - r, 0, 1))

        denoised_spectra = []
        selected_params = []
        snr_values = []

        for sp in tqdm(spectra, desc="  去噪进度（高波数）", unit="样本"):
            original_snr = robust_snr(sp)
            best_score = -np.inf
            best_smoothed = sp.copy()
            best_param = (level_candidates[0], sg_windows[0], sg_poly)
            best_snr = original_snr

            for level in level_candidates:
                for scale in threshold_scales:
                    wavelet_smoothed = wavelet_denoise(sp, level, scale)
                    for window in sg_windows:
                        smoothed = savgol_filter(wavelet_smoothed, window_length=window,
                                                 polyorder=sg_poly, mode='interp')
                        after_snr = robust_snr(smoothed)
                        sloss = shape_loss(sp, smoothed)

                        # 评分：SNR提升 - 峰形失真惩罚
                        # 宽峰对形状保护要求高，失真惩罚权重大
                        if sloss > 0.05:  # 形状相关性低于0.95直接跳过
                            continue
                        score = (after_snr - original_snr) - 30 * sloss
                        if score > best_score:
                            best_score = score
                            best_smoothed = smoothed
                            best_param = (level, window, sg_poly)
                            best_snr = after_snr

            denoised_spectra.append(best_smoothed)
            selected_params.append(best_param)
            snr_values.append(best_snr)

        levels = [p[0] for p in selected_params]
        windows = [p[1] for p in selected_params]
        polys = [p[2] for p in selected_params]
        summary = (
            float(np.mean(levels)),
            Counter(windows).most_common(1)[0][0],
            Counter(polys).most_common(1)[0][0],
            float(np.mean(snr_values))
        )
        return np.array(denoised_spectra), summary

    # ──────────────────────────────────────────────────────────
    # 主流程
    # ──────────────────────────────────────────────────────────
    def process_pipeline(self, denoising=True):
        print("\n开始高波数区域处理流程...")
        self.processed_data = self.spectra_data.copy()
        self.processed_filenames = self.file_names.copy()

        print("\n  ✓ 跳过基线校正（水峰包区域，基线难以可靠估计）")
        print("  ✓ 跳过归一化（参数一致，强度可直接比较）")

        if denoising:
            print("\n1. 去噪处理...")
            self.processed_data, denoise_params = self._denoise(self.processed_data)
            print(f"  去噪参数: 小波层数={denoise_params[0]:.1f}, SG窗口={denoise_params[1]}, 阶数={denoise_params[2]}")
            print(f"  平均SNR: {denoise_params[3]:.2f} dB")
        else:
            print("\n1. 跳过去噪步骤")

        print("\n✅ 高波数区域处理完成！")
        return self.processed_data, self.processed_filenames

    # ──────────────────────────────────────────────────────────
    # 保存数据
    # ──────────────────────────────────────────────────────────
    def save_processed_data(self, save_path):
        if self.processed_data is None:
            return False
        os.makedirs(save_path, exist_ok=True)
        saved_count = 0
        for i, spectrum in enumerate(self.processed_data):
            output_data = np.column_stack((self.wavenumber_data, spectrum))
            if i < len(self.processed_filenames):
                fname = self.processed_filenames[i].replace('/', '_')
                output_file = os.path.join(save_path, f"hw_{fname}")
            else:
                output_file = os.path.join(save_path, f"hw_spectrum_{i+1:04d}.txt")
            np.savetxt(output_file, output_data, delimiter="\t", fmt="%.6f",
                       header=self.header, comments='')
            saved_count += 1
        print(f"✅ 已保存 {saved_count} 条高波数光谱至: {save_path}")
        return True

    # ──────────────────────────────────────────────────────────
    # 画图
    # ──────────────────────────────────────────────────────────
    def plot_dashboard(self, save_dir=None):
        if self.processed_data is None:
            return

        spectra = self.processed_data
        wavenumber = self.wavenumber_data
        n_spec = len(spectra)

        raman_cmap = LinearSegmentedColormap.from_list(
            "raman", ["#0d1b2a", "#1f4e79", "#2e86c1", "#27ae60", "#f4d03f", "#e74c3c", "#ffffff"])

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor("#f7f9fc")
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                               left=0.07, right=0.97, top=0.92, bottom=0.08)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        ext = [wavenumber[0], wavenumber[-1], n_spec, 0]

        # 图1：热图（归一化显示）
        spectra_norm = spectra / (spectra.max(axis=1, keepdims=True) + 1e-10)
        im1 = ax1.imshow(spectra_norm, aspect="auto", cmap=raman_cmap,
                         extent=ext, interpolation="nearest")
        ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax1.set_ylabel("Spectrum Index")
        ax1.set_title("Reproducibility Heatmap\n(Normalized)", fontsize=11, fontweight="bold")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).set_label("Norm. Intensity", fontsize=8)

        # 图2：强度热图（原始）
        sort_idx = np.argsort(spectra.mean(axis=1))
        im2 = ax2.imshow(spectra[sort_idx], aspect="auto", cmap="inferno",
                         extent=ext, interpolation="nearest")
        ax2.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax2.set_ylabel("Spectrum (sorted by mean intensity)")
        ax2.set_title("Intensity Heatmap\n(Raw, sorted)", fontsize=11, fontweight="bold")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label("Intensity (a.u.)", fontsize=8)

        # 图3：平均谱 ± SD
        mean_spec = spectra.mean(axis=0)
        std_spec = spectra.std(axis=0)
        ax3.plot(wavenumber, mean_spec, color="#2c3e50", lw=1.5, label="Mean")
        ax3.fill_between(wavenumber, mean_spec - std_spec, mean_spec + std_spec,
                         alpha=0.25, color="#2980b9", label="±1 SD")
        ax3.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax3.set_ylabel("Intensity (a.u.)")
        ax3.set_title("Mean Spectrum ± SD\n(2800~3000 cm⁻¹)", fontsize=11, fontweight="bold")
        ax3.legend(fontsize=8)

        # 图4：抽样叠加谱（显示重复性）
        n_show = min(30, n_spec)
        idx_show = np.linspace(0, n_spec - 1, n_show, dtype=int)
        cmap_lines = plt.cm.viridis(np.linspace(0, 1, n_show))
        for ii, idx in enumerate(idx_show):
            ax4.plot(wavenumber, spectra[idx], color=cmap_lines[ii], lw=0.8, alpha=0.7)
        ax4.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax4.set_ylabel("Intensity (a.u.)")
        ax4.set_title(f"Overlay (n={n_show} sampled)", fontsize=11, fontweight="bold")

        # 图5：PCC矩阵
        pcc_matrix = np.corrcoef(spectra)
        cool_warm = LinearSegmentedColormap.from_list("cw", ["#2980b9", "#ecf0f1", "#c0392b"])
        im5 = ax5.imshow(pcc_matrix, cmap=cool_warm, vmin=-1, vmax=1,
                         aspect="auto", interpolation="nearest")
        fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04).set_label("Pearson r", fontsize=8)
        mask_off = ~np.eye(n_spec, dtype=bool)
        pcc_vals = pcc_matrix[mask_off]
        ax5.set_xlabel(f"Spectrum Index\nmean PCC = {pcc_vals.mean():.4f} ± {pcc_vals.std():.4f}", fontsize=9)
        ax5.set_ylabel("Spectrum Index")
        ax5.set_title("Pearson Correlation\nCoefficient Matrix", fontsize=11, fontweight="bold")

        # 图6：PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(spectra)
        pca = PCA(n_components=min(2, n_spec, spectra.shape[1]))
        scores = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_ * 100
        ax6.scatter(scores[:, 0], scores[:, 1] if scores.shape[1] > 1 else np.zeros(n_spec),
                    s=40, alpha=0.7, color="#2980b9", edgecolors="white", linewidths=0.5)
        ax6.axhline(0, color="gray", lw=0.5, ls=":")
        ax6.axvline(0, color="gray", lw=0.5, ls=":")
        ax6.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax6.set_ylabel(f"PC2 ({explained[1]:.1f}%)" if len(explained) > 1 else "PC2")
        ax6.set_title("PCA Score Plot", fontsize=11, fontweight="bold")

        fig.suptitle(f"High-Wavenumber Raman Analysis  ·  {n_spec} Spectra  ·  "
                     f"{WAVENUMBER_MIN}~{WAVENUMBER_MAX} cm⁻¹",
                     fontsize=14, fontweight="bold", y=0.98)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, "HighWavenumber_Dashboard.svg"),
                        dpi=300, bbox_inches="tight")
            fig.savefig(os.path.join(save_dir, "HighWavenumber_Dashboard.png"),
                        dpi=300, bbox_inches="tight")

            # 保存矩阵数据
            matrix = np.column_stack([wavenumber ,spectra.T])
            np.savetxt(os.path.join(save_dir, "hw_spectra_matrix.txt"),
                       matrix, delimiter="\t", fmt="%.6f",
                       header="Wavenumber(cm-1)\t" + "\t".join(f"Spectrum_{i}" for i in range(n_spec)))

            stats = np.column_stack([wavenumber, mean_spec, std_spec, spectra.var(axis=0)])
            np.savetxt(os.path.join(save_dir, "hw_spectra_statistics.txt"),
                       stats, delimiter="\t", fmt="%.6f",
                       header="Wavenumber(cm-1)\tMean\tStd_Dev\tVariance")

            print(f"  📸 高波数分析面板已保存至: {save_dir}")


# ============================================================
# 主程序入口（和 Raman预处理1.py 共用同一个 target.txt 格式）
# target.txt 格式：
#   高波数
#   /数据文件夹路径（异常值筛选后的 kept_spectra）
#   /保存文件夹路径
# ============================================================
if __name__ == "__main__":
    processor = RamanHighWavenumberProcessor()

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    target_file = None
    for base_dir in [desktop_path, current_dir]:
        for name in ["target", "target.txt"]:
            candidate = os.path.join(base_dir, name)
            if os.path.isfile(candidate):
                target_file = candidate
                break
        if target_file:
            break
    if target_file is None:
        for base_dir in [desktop_path, current_dir]:
            if os.path.isdir(base_dir):
                for fname in os.listdir(base_dir):
                    if fname.lower().startswith("target"):
                        candidate = os.path.join(base_dir, fname)
                        if os.path.isfile(candidate):
                            target_file = candidate
                            break
            if target_file:
                break

    data_folder, save_folder = None, None
    if target_file is not None:
        with open(target_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        for i, line in enumerate(lines):
            if "高波数" in line:
                if i + 1 < len(lines):
                    data_folder = lines[i + 1]
                if i + 2 < len(lines):
                    save_folder = lines[i + 2]
                break

    if data_folder is None or save_folder is None:
        print("❌ 未找到 target 文件或未找到\"高波数\"标签")
        print("   target.txt 格式：")
        print("   高波数")
        print("   /数据文件夹路径")
        print("   /保存文件夹路径")
        exit()

    print(f"🔎 target 文件: {target_file}")
    print(f"   数据路径: {data_folder}")
    print(f"   保存路径: {save_folder}")
    print(f"   截取波数: {WAVENUMBER_MIN} ~ {WAVENUMBER_MAX} cm⁻¹")

    # ── 配置 ──────────────────────────────────────────────────
    config = {
        'has_subfolders': False,
        'denoising': True,
        'generate_plots': True,
    }
    # ─────────────────────────────────────────────────────────

    print("\n" + "="*50)
    print("🚀 高波数拉曼处理流水线启动")
    print("="*50)

    try:
        processor.load_spectra_data(data_folder, has_subfolders=config['has_subfolders'])
    except Exception as e:
        print(f"\n❌ 加载数据失败: {e}")
        exit()

    try:
        processor.process_pipeline(denoising=config['denoising'])
    except Exception as e:
        print(f"\n❌ 处理失败:\n")
        traceback.print_exc()
        exit()

    data_save_dir = os.path.join(save_folder, "Data")
    processor.save_processed_data(data_save_dir)

    if config['generate_plots']:
        print("\n--- 正在生成图表 ---")
        plot_save_dir = os.path.join(save_folder, "Plots")
        processor.plot_dashboard(save_dir=plot_save_dir)
        print("📸 图表已保存")

    print(f"\n🎉 全部完成！结果保存在: {save_folder}")
