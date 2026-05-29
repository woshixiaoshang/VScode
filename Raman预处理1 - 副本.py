import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
plt.switch_backend('agg')  # 强制纯后台画图
from sklearn.decomposition import PCA
from scipy.stats import chi2
from scipy import sparse
from scipy.sparse.linalg import spsolve
import pywt
from scipy.interpolate import interp1d
from collections import Counter
from scipy.signal import savgol_filter, peak_widths
import pandas as pd
from scipy.signal import find_peaks
import warnings
import traceback

warnings.filterwarnings('ignore')

# 1. 设置中文字体（优先使用微软雅黑，备选黑体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial'] 
# 2. 解决坐标轴负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 12
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

class RamanDataProcessor:
    """拉曼光谱数据处理主类"""  

    def __init__(self):
        self.spectra_data = None
        self.file_names = None
        self.wavenumber_data = None
        self.header = ""
        self.background_data = None
        self.processed_data = None
        self.current_step = 0
        self.steps_config = None
        self.processed_filenames = None

    def load_spectra_data(self, folder_path, has_subfolders=False):
        """加载光谱数据，并自动过滤波数范围异常的文件"""
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

        self.spectra_data, self.file_names, self.wavenumber_data, self.header = self._load_and_filter_files(file_paths)
        
        if self.spectra_data is None:
            raise ValueError("❌ 所有文件均读取失败或格式无效！")

        print(f"\n✅ 成功提取 {len(self.spectra_data)} 条有效光谱数据")
        print(f"   数据维度: {self.spectra_data.shape}")
        print(f"   对齐波数: {self.wavenumber_data[0]:.2f} 到 {self.wavenumber_data[-1]:.2f} cm⁻¹")

    def _load_and_filter_files(self, file_paths_with_names, tolerance=10.0):
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

                    header = "#Wave\t#Intensity"
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        first_line = f.readline().strip()
                        if any(k in first_line.lower() for k in ['raman', 'wave', 'intensity', 'shift', 'cm']):
                            header = first_line

                    raw_data_list.append({
                        'filename': rel_name, 'x': x_values, 'y': y_values, 'header': header
                    })
            except Exception:
                continue

        if not raw_data_list: return None, [], None, ""

        signatures = [(len(item['x']), round(item['x'][0], -1), round(item['x'][-1], -1)) for item in raw_data_list]
        majority_sig = Counter(signatures).most_common(1)[0][0]

        ref_x = None
        ref_header = ""
        for item, sig in zip(raw_data_list, signatures):
            if sig == majority_sig:
                ref_x = item['x']
                ref_header = item['header']
                break

        valid_spectra = []
        valid_filenames = []
        skipped_files = []
        repaired_count = 0

        for item in raw_data_list:
            start_diff = abs(item['x'][0] - ref_x[0])
            end_diff = abs(item['x'][-1] - ref_x[-1])

            if start_diff <= tolerance and end_diff <= tolerance:
                if np.array_equal(item['x'], ref_x):
                    valid_spectra.append(item['y'])
                else:
                    f_interp = interp1d(item['x'], item['y'], kind='linear', bounds_error=False, fill_value="extrapolate")
                    aligned_y = f_interp(ref_x)
                    valid_spectra.append(aligned_y)
                    repaired_count += 1
                valid_filenames.append(item['filename'])
            else:
                skipped_files.append((item['filename'], len(item['x']), item['x'][0], item['x'][-1]))

        if repaired_count > 0:
            print(f"\n🔧 [数据自动对齐] 已将 {repaired_count} 个存在微小漂移的光谱，自动插值重采样至标准波数轴！")

        if skipped_files:
            print(f"\n⚠️ [严重偏移警告] 发现 {len(skipped_files)} 个文件超出了 {tolerance} cm⁻¹ 的允许误差，已跳过！")

        return np.array(valid_spectra), valid_filenames, ref_x, ref_header

    def load_background_data(self, background_path=None):
        if background_path and os.path.isdir(background_path):
            file_paths = []
            for fname in os.listdir(background_path):
                if fname.endswith(".txt"):
                    file_paths.append((os.path.join(background_path, fname), fname))
            
            if file_paths:
                self.background_data, _, _, _ = self._load_and_filter_files(file_paths)
                if self.background_data is not None:
                    print(f"✅ 成功加载 {len(self.background_data)} 条背景数据")
            else:
                self.background_data = None
        else:
            self.background_data = None

    def process_pipeline(self, steps_config):
        print("\n开始光谱数据处理流程...")
        self.processed_data = self.spectra_data.copy()
        self.processed_filenames = self.file_names.copy()

        if steps_config.get('remove_outliers', True) and len(self.processed_data) >= 10:
            print("\n1. 数据筛选 (移除异常值)...")
            outliers, inliers, inlier_filenames = self._remove_outliers(self.processed_data, self.processed_filenames)
            self.processed_data = inliers
            self.processed_filenames = inlier_filenames
            print(f"  移除了 {len(outliers)} 个异常样本，保留 {len(self.processed_data)} 个样本")
        else:
            print("\n1. 跳过数据筛选步骤")

        if steps_config.get('background_subtraction', False) and self.background_data is not None:
            print("\n2. 背景扣除...")
            self.processed_data = self._subtract_background(self.processed_data)
            print(f"  完成背景扣除")
        else:
            print("\n2. 跳过背景扣除步骤")

        if steps_config.get('baseline_correction', True):
            print("\n3. 基线校正...")
            self.processed_data = self._baseline_correction(self.processed_data)
            print(f"  完成基线校正")
        else:
            print("\n3. 跳过基线校正步骤")

        if steps_config.get('denoising', True):
            print("\n4. 去噪处理...")
            self.processed_data, denoise_params = self._denoise(self.processed_data)
            print(f"  去噪参数: 小波层数={denoise_params[0]}, SG窗口={denoise_params[1]}, SG多项式阶数={denoise_params[2]}")
            print(f"  平均SNR: {denoise_params[3]:.2f} dB")
        else:
            print("\n4. 跳过去噪步骤")

        norm_method = steps_config.get('normalization', 'area')
        if norm_method:
            print(f"\n5. 归一化处理 ({norm_method})...")
            if norm_method == 'peak':
                peak_position = steps_config.get('peak_position', 402)
                self.processed_data = self._normalize(self.processed_data, method=norm_method, peak_position=peak_position)
            else:
                self.processed_data = self._normalize(self.processed_data, method=norm_method)
            print(f"  完成 {norm_method} 归一化")
        else:
            print("\n5. 跳过归一化步骤")

        print("\n✅ 光谱数据处理完成！")
        return self.processed_data, self.processed_filenames

    def _remove_outliers(self, data, filenames, confidence=0.95):
        if len(data) < 5: return [], data, filenames
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(data)
        mean = np.mean(reduced, axis=0)
        cov = np.cov(reduced, rowvar=False)
        diff = reduced - mean
        inv_cov = np.linalg.inv(cov)
        md_squared = np.sum(diff @ inv_cov * diff, axis=1)
        threshold = chi2.ppf(confidence, df=2)
        outlier_mask = md_squared > threshold

        outlier_files = [filenames[i] for i in range(len(filenames)) if outlier_mask[i]]
        inlier_data = data[~outlier_mask]
        inlier_filenames = [filenames[i] for i in range(len(filenames)) if not outlier_mask[i]]
        return outlier_files, inlier_data, inlier_filenames

    def _find_peaks_adaptive(self, data, **kwargs):
        window_size, prominence, distance, step_size = kwargs.get('window_size', 200), kwargs.get('prominence', 5.0), kwargs.get('distance', 50), kwargs.get('step_size', 50)
        peaks = []
        for i in range(0, len(data), step_size):
            start, end = max(0, i - window_size // 2), min(len(data), i + window_size // 2)
            if data[i] > np.median(data[start:end]) + 20 * np.std(data[start:end]):
                peaks.append(i)
        peaks, _ = find_peaks(data, height=0, distance=distance, prominence=prominence)
        if len(peaks) > 0:
            widths = peak_widths(data, peaks, rel_height=0.5)[0]
            peak_areas = widths * data[peaks]
            return peaks[peak_areas >= np.mean(peak_areas) * 0.6]
        return peaks

    def _subtract_background(self, spectra):
        if self.background_data is None or len(self.background_data) == 0: return spectra
        background_avg = np.mean(self.background_data, axis=0)
        bg_peaks = self._find_peaks_adaptive(background_avg)
        if len(bg_peaks) == 0: return spectra

        scale_factors = []
        for spectrum in spectra:
            ratios = spectrum[bg_peaks] / background_avg[bg_peaks]
            scale_factors.append(np.median(ratios[ratios > 0]) if np.any(ratios > 0) else 0.0)

        corrected_spectra = []
        for spectrum, scale_factor in zip(spectra, scale_factors):
            corrected = spectrum - scale_factor * background_avg
            corrected[corrected < 0] = 0
            corrected_spectra.append(corrected)
        return np.array(corrected_spectra)

    def _baseline_correction(self, spectra, lam=1e4, porder=0.005, itermax=15):
        if len(spectra) == 0: return spectra
        m = spectra.shape[1]
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(m - 2, m))
        penalty = (lam * (D.transpose().dot(D))).tocsc()

        def airPLS(signal):
            w = np.ones(m)
            w[:10], w[-10:] = 1000, 1000
            for i in range(itermax):
                W = sparse.spdiags(w, 0, m, m).tocsc()
                baseline = spsolve(W + penalty, w * signal)
                diff = signal - baseline
                w_new = np.where(diff >= 0, porder, np.exp(i * diff / np.sum(diff[diff < 0])))
                w_new[w_new < 1e-8] = 1e-8
                if np.linalg.norm(w - w_new) / np.linalg.norm(w) < 1e-3: break
                w = w_new
            return baseline

        return np.array([sp - airPLS(sp) for sp in spectra])

    def _denoise(self, spectra, **kwargs):
        if len(spectra) == 0: return spectra, (0, 0, 0, 0)
        wavelet, snr_threshold = kwargs.get('wavelet', 'sym8'), kwargs.get('snr_threshold', 30)
        from itertools import product
        wavelet_levels, sg_windows, sg_polys = [1, 2, 3], [5, 7, 9, 11, 15, 21], [2, 3, 4]

        def wavelet_denoise(y, level):
            coeffs = pywt.wavedec(y, wavelet=wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(y)))
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            return pywt.waverec(coeffs, wavelet)[:len(y)]

        def get_snr(original, denoised):
            noise = original - denoised
            noise_power = np.mean(noise ** 2)
            if noise_power == 0: return float('inf')
            return 10 * np.log10(np.mean(original ** 2) / noise_power)

        sample_spectra = spectra[np.random.choice(len(spectra), min(5, len(spectra)), replace=False)]
        best_params, best_avg_snr = None, -np.inf

        for level, sg_window, sg_poly in product(wavelet_levels, sg_windows, sg_polys):
            snr_list = []
            for sp in sample_spectra:
                denoised = wavelet_denoise(sp, level)
                if sg_window >= len(denoised) or sg_window % 2 == 0: continue
                snr_list.append(get_snr(sp, savgol_filter(denoised, window_length=sg_window, polyorder=sg_poly)))

            if not snr_list: continue
            avg_snr = np.mean(snr_list)
            if avg_snr > best_avg_snr:
                best_params, best_avg_snr = (level, sg_window, sg_poly), avg_snr
                if avg_snr >= snr_threshold: break 

        if best_params is None: return spectra, (0, 0, 0, 0)
        level, sg_window, sg_poly = best_params
        
        smoothed_all = []
        for sp in spectra:
            denoised = wavelet_denoise(sp, level)
            smoothed_all.append(savgol_filter(denoised, window_length=sg_window, polyorder=sg_poly))

        return np.array(smoothed_all), (level, sg_window, sg_poly, best_avg_snr)

    def _normalize(self, spectra, method='area', peak_position=None):
        if method == 'area':
            return np.array([sp / area if (area := np.trapz(sp)) != 0 and not np.isnan(area) else sp for sp in spectra])
        elif method == 'maxmin':
            return np.array([(sp - min_val) / (max_val - min_val) if (max_val := np.max(sp)) != (min_val := np.min(sp)) and not np.isnan(max_val) else sp for sp in spectra])
        elif method == 'peak':
            idx = np.argmin(np.abs(self.wavenumber_data - peak_position)) if peak_position is not None else None
            return np.array([sp / peak_val if (peak_val := sp[idx] if idx is not None else np.max(sp)) != 0 and not np.isnan(peak_val) else sp for sp in spectra])
        return spectra

    def save_processed_data(self, save_path, keep_structure=False):
        if self.processed_data is None: return False
        saved_count = 0
        for i, spectrum in enumerate(self.processed_data):
            output_data = np.column_stack((self.wavenumber_data, spectrum))
            if i < len(self.processed_filenames):
                original_filename = self.processed_filenames[i]
                if keep_structure and "/" in original_filename:
                    target_dir = os.path.join(save_path, os.path.dirname(original_filename))
                    os.makedirs(target_dir, exist_ok=True)
                    output_file = os.path.join(target_dir, f"processed_{os.path.basename(original_filename)}")
                else:
                    os.makedirs(save_path, exist_ok=True)
                    output_file = os.path.join(save_path, f"processed_{original_filename.replace('/', '_')}")
            else:
                os.makedirs(save_path, exist_ok=True)
                output_file = os.path.join(save_path, f"processed_spectrum_{i + 1:04d}.txt")

            np.savetxt(output_file, output_data, delimiter="\t", fmt="%.6f", header=self.header, comments='')
            saved_count += 1
        print(f"✅ 已保存 {saved_count} 个处理后文件到: {save_path}")
        return True

    def plot_spectra(self, n_samples=10, show_original=True, show_processed=True, save_dir=None):
        """绘制光谱图预览"""
        if show_original and self.spectra_data is not None:
            self._plot_single_set(self.spectra_data, "原始光谱", n_samples, save_dir, prefix="Original")
        if show_processed and self.processed_data is not None:
            self._plot_single_set(self.processed_data, "处理后光谱", n_samples, save_dir, prefix="Processed")

    def _plot_single_set(self, data, title, n_samples, save_dir=None, prefix=""):
        indices = np.linspace(0, data.shape[0] - 1, min(n_samples, data.shape[0]), dtype=int)
        plt.figure(figsize=(10, 6))
        for idx in indices:
            plt.plot(self.wavenumber_data, data[idx], label=f'Sample {idx + 1}')
        plt.xlabel('Raman Shift (cm$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{prefix}_Spectra_Preview.png"), dpi=300, bbox_inches='tight')


# ======================== 全新 2x3 综合分析画图面板 ======================== #
def plot_comprehensive_dashboard(processor, save_dir=None):
    """
    取代原有6个独立画图函数，生成一张高整合度的2x3科研面板图。
    包含：重复性热图、强度热图、平均谱±SD、PCA聚类、PCC矩阵、堆叠谱。
    """
    if processor.processed_data is None:
        return

    spectra = processor.processed_data
    wavenumber = processor.wavenumber_data
    filenames = processor.processed_filenames
    n_spec, n_pts = spectra.shape

    # 1. 自动提取分组标签 (如果通过子文件夹加载，则根据子文件夹名称分组)
    labels = []
    for fname in filenames:
        if "/" in fname:
            labels.append(fname.split("/")[0])
        elif "\\" in fname:
            labels.append(fname.split("\\")[0])
        else:
            labels.append("All")
    labels = np.array(labels)
    unique_groups = list(dict.fromkeys(labels))
    n_groups = len(unique_groups)
    group_means = {g: spectra[labels == g].mean(axis=0) for g in unique_groups}

    # 2. 颜色配置
    raman_cmap = LinearSegmentedColormap.from_list("raman", ["#0d1b2a", "#1f4e79", "#2e86c1", "#27ae60", "#f4d03f", "#e74c3c", "#ffffff"])
    cool_warm = LinearSegmentedColormap.from_list("cool_warm", ["#2980b9", "#ecf0f1", "#c0392b"])
    group_colors = plt.cm.tab10(np.linspace(0, 0.9, max(1, n_groups)))
    group_color_map = dict(zip(unique_groups, group_colors))

    # 3. 创建大画布 (2x3)
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#f7f9fc")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38, left=0.07, right=0.97, top=0.93, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # 注意：imshow 的 extent 设置，自动适应上升或下降的波数
    ext = [wavenumber[0], wavenumber[-1], n_spec, 0]

    # --- 图1: 重复性热图 (归一化后) ---
    spectra_norm = spectra / (spectra.max(axis=1, keepdims=True) + 1e-10)
    im1 = ax1.imshow(spectra_norm, aspect="auto", cmap=raman_cmap, extent=ext, interpolation="nearest")
    ax1.set_xlabel("Raman Shift (cm$^{-1}$)")
    ax1.set_ylabel("Spectrum Index")
    ax1.set_title("Reproducibility Heatmap\n(Normalized)", fontsize=11, fontweight="bold")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Norm. Intensity", fontsize=8)

    # --- 图2: 强度热图 (原始强度，按均值排序) ---
    sort_idx = np.argsort(spectra.mean(axis=1))
    spectra_sorted = spectra[sort_idx]
    im2 = ax2.imshow(spectra_sorted, aspect="auto", cmap="inferno", extent=ext, interpolation="nearest")
    ax2.set_xlabel("Raman Shift (cm$^{-1}$)")
    ax2.set_ylabel("Spectrum (sorted by mean intensity)")
    ax2.set_title("Intensity Heatmap\n(Raw, sorted)", fontsize=11, fontweight="bold")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Intensity (a.u.)", fontsize=8)

    # --- 图3: 平均谱 ± 标准差 ---
    mean_spec, std_spec = spectra.mean(axis=0), spectra.std(axis=0)
    ax3.plot(wavenumber, mean_spec, color="#2c3e50", lw=1.5, label="Mean")
    ax3.fill_between(wavenumber, mean_spec - std_spec, mean_spec + std_spec, alpha=0.25, color="#2980b9", label="±1 SD")
    ax3.set_xlabel("Raman Shift (cm$^{-1}$)")
    ax3.set_ylabel("Intensity (a.u.)")
    ax3.set_title("Mean Spectrum ± SD", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8)

    # --- 图4: PCA 聚类 ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(spectra)
    pca = PCA(n_components=min(5, n_spec, n_pts))
    scores = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    for g in unique_groups:
        mask = labels == g
        ax4.scatter(scores[mask, 0], scores[mask, 1], label=g, color=group_color_map[g], s=60, alpha=0.8, edgecolors="white", linewidths=0.5)
        # 添加 95% 置信椭圆 (均值±2σ)
        if mask.sum() >= 3:
            cx, cy = scores[mask, 0].mean(), scores[mask, 1].mean()
            sx, sy = scores[mask, 0].std() * 2, scores[mask, 1].std() * 2
            theta = np.linspace(0, 2 * np.pi, 100)
            ax4.plot(cx + sx * np.cos(theta), cy + sy * np.sin(theta), color=group_color_map[g], lw=1, ls="--", alpha=0.5)

    ax4.axhline(0, color="gray", lw=0.5, ls=":")
    ax4.axvline(0, color="gray", lw=0.5, ls=":")
    ax4.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax4.set_ylabel(f"PC2 ({explained[1]:.1f}%)" if len(explained) > 1 else "PC2")
    ax4.set_title("PCA Clustering", fontsize=11, fontweight="bold")
    if n_groups > 1: ax4.legend(fontsize=8, markerscale=0.9)

    # --- 图5: PCC 矩阵 ---
    pcc_matrix = np.corrcoef(spectra)
    im5 = ax5.imshow(pcc_matrix, cmap=cool_warm, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    cbar5 = fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cbar5.set_label("Pearson r", fontsize=8)
    if n_spec <= 20:
        for i in range(n_spec):
            for j in range(n_spec):
                ax5.text(j, i, f"{pcc_matrix[i,j]:.2f}", ha="center", va="center", fontsize=6, color="black" if abs(pcc_matrix[i,j]) < 0.7 else "white")
    # 计算剔除自相关(对角线)后的真实平均 PCC 
    mask_off = ~np.eye(n_spec, dtype=bool)
    pcc_vals = pcc_matrix[mask_off]
    
    # 将平均值和标准差显示在 X 轴标签上
    ax5.set_xlabel(f"Spectrum Index\nmean PCC = {pcc_vals.mean():.4f} ± {pcc_vals.std():.4f}", fontsize=10)
    ax5.set_ylabel("Spectrum Index")
    ax5.set_title("Pearson Correlation\nCoefficient Matrix", fontsize=11, fontweight="bold")

    # --- 图6: 平均谱堆叠 (偏移显示) ---
    if n_groups > 1:
        offset_step = max(gm.max() - gm.min() for gm in group_means.values()) * 1.3
        for i, g in enumerate(unique_groups):
            gm = group_means[g]
            offset = i * offset_step
            ax6.plot(wavenumber, gm + offset, color=group_color_map[g], lw=1.5, label=g)
            ax6.fill_between(wavenumber, gm + offset - spectra[labels == g].std(axis=0), gm + offset + spectra[labels == g].std(axis=0), alpha=0.15, color=group_color_map[g])
            ax6.text(max(wavenumber), gm.mean() + offset, f" {g}", fontsize=9, va="center", color=group_color_map[g])
        ax6.set_title("Stacked Mean Spectra\n(by Group)", fontsize=11, fontweight="bold")
    else:
        max_range = (spectra.max(axis=1) - spectra.min(axis=1)).max()
        offset_step = max_range * 1.1
        cmap_stack = plt.cm.viridis(np.linspace(0, 1, n_spec))
        for i, sp in enumerate(spectra):
            ax6.plot(wavenumber, sp + i * offset_step, color=cmap_stack[i], lw=0.8, alpha=0.85)
        ax6.set_title("Stacked Individual Spectra", fontsize=11, fontweight="bold")
    
    ax6.set_xlabel("Raman Shift (cm$^{-1}$)")
    ax6.set_ylabel("Intensity (offset, a.u.)")
    ax6.set_yticks([])

    fig.suptitle(f"Raman Spectra Comprehensive Analysis  ·  {n_spec} Spectra", fontsize=16, fontweight="bold", y=0.98)

    # 4. 导出图表及附加数据
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_svg = os.path.join(save_dir, "Comprehensive_Analysis_Dashboard.svg")
        save_png = os.path.join(save_dir, "Comprehensive_Analysis_Dashboard.png")
        fig.savefig(save_svg, dpi=300, bbox_inches="tight")
        fig.savefig(save_png, dpi=300, bbox_inches="tight")
        
        pcc_save = os.path.join(save_dir, "pcc_matrix.txt")
        np.savetxt(pcc_save, pcc_matrix, fmt="%.6f", header=f"Pearson Correlation Coefficient Matrix ({n_spec}x{n_spec})")
        
        loadings_save = os.path.join(save_dir, "pca_loadings.txt")
        loadings_header = "Wavenumber " + " ".join([f"PC{i+1}({explained[i]:.1f}%)" for i in range(pca.n_components_)])
        np.savetxt(loadings_save, np.column_stack([wavenumber, pca.components_.T]), header=loadings_header, fmt="%.6f")
        
        print(f"  📸 综合面板高清大图及数据矩阵已保存至: {save_dir}")

# ======================== 全自动执行流水线 ======================== #

if __name__ == "__main__":
    processor = RamanDataProcessor()
    
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    candidate_names = ["target", "target.txt"]
    target_file = None
    for base_dir in [desktop_path, current_dir]:
        for name in candidate_names:
            candidate = os.path.join(base_dir, name)
            if os.path.isfile(candidate):
                target_file = candidate
                break
        if target_file: break

    if target_file is None:
        for base_dir in [desktop_path, current_dir]:
            if os.path.isdir(base_dir):
                for fname in os.listdir(base_dir):
                    if fname.lower().startswith("target"):
                        candidate = os.path.join(base_dir, fname)
                        if os.path.isfile(candidate):
                            target_file = candidate
                            break
                if target_file: break
    
    data_folder, save_folder = None, None
    if target_file is not None:
        with open(target_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                data_folder, save_folder = lines[0].strip(), lines[1].strip()
    
    if data_folder is None or save_folder is None:
        print("❌ 未找到 target 文件或文件格式错误。")
        exit()
    
    print(f"🔎 已找到 target 文件: {target_file}")
    
    # ================= 自定义开关区 =================
    has_subfolders = False  # 数据是否有子文件夹
    generate_plots = True   # 画图总开关：True 为画图并保存
    plot_only = True       # 只画图模式：True 为直接加载处理好的数据进行画图
    # ===============================================
    print("\n" + "="*50 + "\n🚀 拉曼光谱自动化处理流水线已启动\n" + "="*50)
    
    if plot_only:
        print("\n--- 只画图模式：从已保存的文件加载处理后数据 ---")
        data_save_dir = os.path.join(save_folder, "Data")
        processor.processed_data, processor.processed_filenames = [], []
        if os.path.exists(data_save_dir):
            for file in os.listdir(data_save_dir):
                if file.startswith("processed_") and file.endswith(".txt"):
                    file_path = os.path.join(data_save_dir, file)
                    data = np.loadtxt(file_path)
                    if data.shape[1] >= 2:
                        if processor.wavenumber_data is None:
                            processor.wavenumber_data = data[:, 0]
                        processor.processed_data.append(data[:, 1])
                        processor.processed_filenames.append(file.replace("processed_", "").replace(".txt", ""))
            if processor.processed_data:
                processor.processed_data = np.array(processor.processed_data)
                if processor.wavenumber_data[0] > processor.wavenumber_data[-1]:
                    processor.wavenumber_data = processor.wavenumber_data[::-1]
                    processor.processed_data = processor.processed_data[:, ::-1]
                
                # === 重采样到均匀波数间隔（1 cm-1）===
                wavenumber_resampled = np.arange(np.round(processor.wavenumber_data.min()), 
                                                 np.round(processor.wavenumber_data.max()) + 1, 1.0)
                
                # 重采样处理后光谱
                processed_spectra_resampled = []
                for spectrum in processor.processed_data:
                    f_interp = interp1d(processor.wavenumber_data, spectrum, kind='linear', bounds_error=False, fill_value="extrapolate")
                    processed_spectra_resampled.append(f_interp(wavenumber_resampled))
                processor.processed_data = np.array(processed_spectra_resampled)
                processor.wavenumber_data = wavenumber_resampled
                
                # 保存处理后用来画图的光谱矩阵（重采样版）
                processed_matrix_path = os.path.join(save_folder, "processed_spectra_matrix.txt")
                processed_matrix = np.column_stack([processor.wavenumber_data] + [processor.processed_data[i] for i in range(len(processor.processed_data))])
                np.savetxt(processed_matrix_path, processed_matrix, delimiter="\t", fmt="%.6f",
                           header=f"Wavenumber(cm-1)\t" + "\t".join([f"Spectrum_{i}" for i in range(len(processor.processed_data))]))
                
                # 计算处理后数据的统计信息
                processed_mean = np.mean(processor.processed_data, axis=0)
                processed_std = np.std(processor.processed_data, axis=0)
                processed_var = np.var(processor.processed_data, axis=0)
                processed_stats_matrix = np.column_stack([processor.wavenumber_data, processed_mean, processed_std, processed_var])
                processed_stats_path = os.path.join(save_folder, "processed_spectra_statistics.txt")
                np.savetxt(processed_stats_path, processed_stats_matrix, delimiter="\t", fmt="%.6f",
                           header="Wavenumber(cm-1)\tMean\tStd_Dev\tVariance")
                
                print(f"💾 处理后光谱矩阵已保存（均匀1cm-1采样）: {processed_matrix_path}")
                print(f"💾 处理后光谱统计信息已保存: {processed_stats_path}")
            else:
                print("❌ 未找到处理后的数据文件"); exit()
        else:
            print(f"❌ 数据保存目录不存在: {data_save_dir}"); exit()
    else:
        try:
            processor.load_spectra_data(data_folder, has_subfolders=has_subfolders)
        except Exception as e:
            print(f"\n❌ 加载数据失败: {e}"); exit()
        
        steps_config = {
            'remove_outliers': True,          
            'background_subtraction': True,  
            'baseline_correction': True,     
            'denoising': True,                
            'normalization': 'area'           
        }
        
        try:
            processor.process_pipeline(steps_config)
        except Exception as e:
            print(f"\n❌ 处理发生错误:\n"); traceback.print_exc(); exit()
        
        data_save_dir = os.path.join(save_folder, "Data")
        processor.save_processed_data(data_save_dir, keep_structure=has_subfolders)
        
        # === 重采样到均匀波数间隔（1 cm-1）===
        wavenumber_resampled = np.arange(np.round(processor.wavenumber_data.min()), 
                                         np.round(processor.wavenumber_data.max()) + 1, 1.0)
        
        # 重采样原始光谱
        raw_spectra_resampled = []
        for spectrum in processor.spectra_data:
            f_interp = interp1d(processor.wavenumber_data, spectrum, kind='linear', bounds_error=False, fill_value="extrapolate")
            raw_spectra_resampled.append(f_interp(wavenumber_resampled))
        raw_spectra_resampled = np.array(raw_spectra_resampled)
        
        # 重采样处理后光谱
        processed_spectra_resampled = []
        for spectrum in processor.processed_data:
            f_interp = interp1d(processor.wavenumber_data, spectrum, kind='linear', bounds_error=False, fill_value="extrapolate")
            processed_spectra_resampled.append(f_interp(wavenumber_resampled))
        processed_spectra_resampled = np.array(processed_spectra_resampled)
        
        # 保存原始读取的光谱矩阵（重采样版）
        raw_matrix_path = os.path.join(save_folder, "raw_spectra_matrix.txt")
        raw_matrix = np.column_stack([wavenumber_resampled] + [raw_spectra_resampled[i] for i in range(len(raw_spectra_resampled))])
        np.savetxt(raw_matrix_path, raw_matrix, delimiter="\t", fmt="%.6f", 
                   header=f"Wavenumber(cm-1)\t" + "\t".join([f"Spectrum_{i}" for i in range(len(raw_spectra_resampled))]))
        
        # 计算原始数据的统计信息
        raw_mean = np.mean(raw_spectra_resampled, axis=0)
        raw_std = np.std(raw_spectra_resampled, axis=0)
        raw_var = np.var(raw_spectra_resampled, axis=0)
        raw_stats_matrix = np.column_stack([wavenumber_resampled, raw_mean, raw_std, raw_var])
        raw_stats_path = os.path.join(save_folder, "raw_spectra_statistics.txt")
        np.savetxt(raw_stats_path, raw_stats_matrix, delimiter="\t", fmt="%.6f",
                   header="Wavenumber(cm-1)\tMean\tStd_Dev\tVariance")
        
        print(f"💾 原始光谱矩阵已保存（均匀1cm-1采样）: {raw_matrix_path}")
        print(f"💾 原始光谱统计信息已保存: {raw_stats_path}")
        
        # 保存处理后用来画图的光谱矩阵（重采样版）
        processed_matrix_path = os.path.join(save_folder, "processed_spectra_matrix.txt")
        processed_matrix = np.column_stack([wavenumber_resampled] + [processed_spectra_resampled[i] for i in range(len(processed_spectra_resampled))])
        np.savetxt(processed_matrix_path, processed_matrix, delimiter="\t", fmt="%.6f",
                   header=f"Wavenumber(cm-1)\t" + "\t".join([f"Spectrum_{i}" for i in range(len(processed_spectra_resampled))]))
        
        # 计算处理后数据的统计信息
        processed_mean = np.mean(processed_spectra_resampled, axis=0)
        processed_std = np.std(processed_spectra_resampled, axis=0)
        processed_var = np.var(processed_spectra_resampled, axis=0)
        processed_stats_matrix = np.column_stack([wavenumber_resampled, processed_mean, processed_std, processed_var])
        processed_stats_path = os.path.join(save_folder, "processed_spectra_statistics.txt")
        np.savetxt(processed_stats_path, processed_stats_matrix, delimiter="\t", fmt="%.6f",
                   header="Wavenumber(cm-1)\tMean\tStd_Dev\tVariance")
        
        print(f"💾 处理后光谱矩阵已保存（均匀1cm-1采样）: {processed_matrix_path}")
        print(f"💾 处理后光谱统计信息已保存: {processed_stats_path}")

    # 6. 生成并保存图表
    if generate_plots:
        print("\n--- 正在生成并保存图表 ---")
        plot_save_dir = os.path.join(save_folder, "Plots")
        
        # 1. 抽样原始光谱预览 (仅完整处理模式下存在)
        show_original_spectra = not plot_only
        processor.plot_spectra(n_samples=5, show_original=show_original_spectra, show_processed=True, save_dir=plot_save_dir)
        
        # 2. 调用全新的 2x3 综合画图面板（取代旧版的散碎拼图）
        plot_comprehensive_dashboard(processor, save_dir=plot_save_dir)
        
        print("📸 图片及分析数据已全部生成并保存！")
    else:
        print("\n--- 🚫 画图开关已关闭 (generate_plots=False) ---")
    
    print(f"\n🎉 全部任务已完成！请前往 {save_folder} 查看结果。")