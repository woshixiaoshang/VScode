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
from tqdm import tqdm
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
        self.outlier_info = []

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
        else:
            self.background_data = None

    def process_pipeline(self, steps_config):
        print("\n开始光谱数据处理流程...")
        self.processed_data = self.spectra_data.copy()
        self.processed_filenames = self.file_names.copy()

        background_subtraction = steps_config.get("background_subtraction", False)
        baseline_correction = steps_config.get("baseline_correction", True)
        remove_outliers = steps_config.get("remove_outliers", True)
        denoising = steps_config.get("denoising", True)
        norm_method = steps_config.get("normalization", "area")

        # 步骤1：背景扣除（移到最前面）
        if background_subtraction and self.background_data is not None:
            print("\n1. 背景扣除...")
            self.processed_data = self._subtract_background(self.processed_data)
            print("  完成背景扣除")
        elif background_subtraction:
            print("\n1. 背景扣除已启用，但未检测到背景数据，跳过该步骤")
        else:
            print("\n1. 跳过背景扣除步骤")

        # 步骤2：基线校正（移到异常值筛选之前）
        if baseline_correction:
            print("\n2. 基线校正...")
            self.processed_data = self._baseline_correction(self.processed_data)
            print("  完成基线校正")
        else:
            print("\n2. 跳过基线校正步骤")

        # 步骤3：异常值筛选（基线校正之后跑，PCC才有意义）
        if remove_outliers and len(self.processed_data) >= 10:
            print("\n3. 数据筛选 (移除异常值)...")
            outliers, inliers, inlier_filenames, outlier_info = self._remove_outliers(
                self.processed_data, self.processed_filenames
            )
            self.processed_data = inliers
            self.processed_filenames = inlier_filenames
            self.outlier_info = outlier_info
            print(
                f"  移除了 {len(outliers)} 个异常样本，保留 {len(self.processed_data)} 个样本"
            )
            if self.outlier_info:
                print("  剔除原因: 依据 PCA Mahalanobis 距离超过阈值")
                for record in self.outlier_info[:5]:
                    print(
                        f"    {record['filename']} | md²={record['md_squared']:.2f} | threshold={record['threshold']:.2f} | PC1={record['pc1']:.2f} | PC2={record['pc2']:.2f}"
                    )
                if len(self.outlier_info) > 5:
                    print(f"    ... 共 {len(self.outlier_info)} 个异常样本，详情已记录")
        else:
            self.outlier_info = []
            print("\n3. 跳过数据筛选步骤")

        # 步骤4：去噪
        if denoising:
            print("\n4. 去噪处理...")
            self.processed_data, denoise_params = self._denoise(self.processed_data)
            try:
                print(
                    f"  去噪参数: 小波层数={denoise_params[0]}, "
                    f"SG窗口={denoise_params[1]}, SG多项式阶数={denoise_params[2]}"
                )
                print(f"  平均SNR: {denoise_params[3]:.2f} dB")
            except Exception:
                print("  去噪完成 (参数不可用)")
        else:
            print("\n4. 跳过去噪步骤")

        # 步骤5：归一化
        if norm_method:
            print(f"\n5. 归一化处理 ({norm_method})...")
            if norm_method == "peak":
                peak_position = steps_config.get("peak_position", 402)
                self.processed_data = self._normalize(
                    self.processed_data, method=norm_method, peak_position=peak_position
                )
            else:
                self.processed_data = self._normalize(
                    self.processed_data, method=norm_method
                )
            print(f"  完成 {norm_method} 归一化")
        else:
            print("\n5. 跳过归一化步骤")

        print("\n✅ 光谱数据处理完成！")
        return self.processed_data, self.processed_filenames

    def _remove_outliers(self, data, filenames, confidence=0.95):
        if len(data) < 5:
            return [], data, filenames, []
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(data)
        mean = np.mean(reduced, axis=0)
        cov = np.cov(reduced, rowvar=False)
        diff = reduced - mean
        inv_cov = np.linalg.inv(cov)
        md_squared = np.sum(diff @ inv_cov * diff, axis=1)
        threshold = chi2.ppf(confidence, df=2)
        outlier_mask = md_squared > threshold

        outlier_info = []
        for i in range(len(filenames)):
            outlier_info.append({
                'filename': filenames[i],
                'md_squared': float(md_squared[i]),
                'threshold': float(threshold),
                'pc1': float(reduced[i, 0]),
                'pc2': float(reduced[i, 1]),
                'removed': bool(outlier_mask[i])
            })

        outlier_files = [filenames[i] for i in range(len(filenames)) if outlier_mask[i]]
        inlier_data = data[~outlier_mask]
        inlier_filenames = [filenames[i] for i in range(len(filenames)) if not outlier_mask[i]]
        removed_info = [record for record in outlier_info if record['removed']]
        return outlier_files, inlier_data, inlier_filenames, removed_info

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

        return np.array([sp - airPLS(sp) for sp in tqdm(spectra, desc="  基线校正进度")])

    def _denoise(self, spectra, **kwargs):
        """Adaptive wavelet + Savitzky-Golay denoising with peak preservation."""
        if len(spectra) == 0:
            return spectra, (0, 0, 0, 0)

        wavelet = kwargs.get('wavelet', 'db8')
        level_candidates = kwargs.get('wavelet_levels', [1, 2, 3, 4])
        sg_windows = kwargs.get('sg_windows', [5, 7, 9, 11, 15, 21])
        sg_poly = kwargs.get('sg_poly', 3)
        threshold_scales = kwargs.get('threshold_scales', [0.65, 0.85, 1.0, 1.2])
        max_peak_loss = kwargs.get('max_peak_loss', 0.12)

        n_points = spectra.shape[1]
        max_level = pywt.dwt_max_level(n_points, pywt.Wavelet(wavelet).dec_len)
        level_candidates = [level for level in level_candidates if 0 < level <= max_level]
        if not level_candidates:
            return spectra, (0, 0, 0, 0)

        def valid_window(window, polyorder):
            window = min(int(window), n_points - 1 if n_points % 2 == 0 else n_points)
            if window % 2 == 0:
                window -= 1
            min_window = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
            return window if window >= min_window else None

        sg_windows = [valid_window(w, sg_poly) for w in sg_windows]
        sg_windows = sorted({w for w in sg_windows if w is not None})
        if not sg_windows:
            return spectra, (0, 0, 0, 0)

        def noise_sigma(y):
            diff = np.diff(y)
            sigma = np.median(np.abs(diff - np.median(diff))) / 0.6745
            return max(float(sigma) / np.sqrt(2), 1e-12)

        def robust_snr(y):
            sigma = noise_sigma(y)
            signal = np.percentile(y, 95) - np.percentile(y, 5)
            if signal <= 0:
                signal = np.std(y)
            return 20 * np.log10((signal + 1e-12) / sigma)

        def wavelet_denoise(y, level, threshold_scale):
            coeffs = pywt.wavedec(y, wavelet=wavelet, level=level, mode='symmetric')
            sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
            threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(y)))
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            return pywt.waverec(coeffs, wavelet, mode='symmetric')[:len(y)]

        def peak_loss(original, candidate):
            sigma = noise_sigma(original)
            peaks, _ = find_peaks(original, prominence=max(sigma * 3, 1e-12))
            if len(peaks) == 0:
                return 0.0
            # 原始峰高减去噪声贡献，更真实地估计峰高损失
            original_heights = np.maximum(original[peaks] - sigma, 1e-12)
            candidate_heights = np.maximum(candidate[peaks], 0)
            loss = (original_heights - candidate_heights) / original_heights
            return float(np.clip(np.median(loss), 0, 1))

        denoised_spectra = []
        selected_params = []
        snr_values = []

        for sp in tqdm(spectra, desc="  去噪进度", unit="样本"):
            original_snr = robust_snr(sp)
            noise_level = noise_sigma(sp)
            dynamic_range = np.percentile(sp, 95) - np.percentile(sp, 5)
            noise_ratio = noise_level / (dynamic_range + 1e-12)

            if noise_ratio > 0.08:
                preferred_levels = sorted(level_candidates, reverse=True)
                preferred_windows = sorted(sg_windows, reverse=True)
                dynamic_peak_loss = 0.22  # 噪声大，放宽峰损失容忍度
            elif noise_ratio > 0.035:
                preferred_levels = level_candidates
                preferred_windows = sg_windows
                dynamic_peak_loss = 0.15  # 中等噪声，使用默认峰损失容忍度
            else:
                preferred_levels = sorted(level_candidates)
                preferred_windows = sorted(sg_windows)
                dynamic_peak_loss = max_peak_loss  # 噪声小，严格保峰

            best_score = -np.inf
            best_smoothed = sp
            best_param = (preferred_levels[0], preferred_windows[0], sg_poly)
            best_snr = original_snr

            for level in preferred_levels:
                for scale in threshold_scales:
                    wavelet_smoothed = wavelet_denoise(sp, level, scale)
                    for window in preferred_windows:
                        smoothed = savgol_filter(wavelet_smoothed, window_length=window, polyorder=sg_poly, mode='interp')
                        after_snr = robust_snr(smoothed)
                        loss = peak_loss(sp, smoothed)
                        residual = sp - smoothed
                        residual_ratio = np.std(residual) / (np.std(sp) + 1e-12)

                        if loss > dynamic_peak_loss:
                            continue

                        score = (after_snr - original_snr) - 18 * loss - 2.5 * abs(residual_ratio - min(noise_ratio * 4, 0.35))
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

    def _normalize(self, spectra, method='area', peak_position=None):
        if method == 'area':
            return np.array([sp / area if (area := np.trapz(sp)) != 0 
                         and not np.isnan(area) else sp for sp in spectra])
        elif method == 'maxmin':
            return np.array([(sp - min_val) / (max_val - min_val) 
                         if (max_val := np.max(sp)) != (min_val := np.min(sp)) 
                         and not np.isnan(max_val) else sp for sp in spectra])
        elif method == 'peak':
            if peak_position is not None:
            # 在peak_position ± 15 cm⁻¹窗口内找真实峰顶，而不是固定索引取值
                center_idx = np.argmin(np.abs(self.wavenumber_data - peak_position))
                window = 15  # cm⁻¹
                step = self.wavenumber_data[1] - self.wavenumber_data[0]
                half_win = max(1, int(window / abs(step)))
                lo = max(0, center_idx - half_win)
                hi = min(len(self.wavenumber_data), center_idx + half_win + 1)
            
                normalized = []
                for sp in spectra:
                    peak_val = np.max(sp[lo:hi])  # 局部最大值
                    if peak_val != 0 and not np.isnan(peak_val):
                        normalized.append(sp / peak_val)
                    else:
                        normalized.append(sp)
                return np.array(normalized)
            else:
                return np.array([sp / np.max(sp) if np.max(sp) != 0 else sp for sp in spectra])
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
        if save_dir is not None:
            self.plot_outliers(save_dir=save_dir)

    def plot_outliers(self, save_dir=None, max_samples=10):
        if not self.outlier_info:
            return

        outlier_filenames = [record['filename'] for record in self.outlier_info]
        outlier_spectra = []
        outlier_labels = []
        for fname in outlier_filenames:
            try:
                idx = self.file_names.index(fname)
            except ValueError:
                continue
            outlier_spectra.append(self.spectra_data[idx])
            outlier_labels.append(fname)

        if not outlier_spectra:
            return

        outlier_spectra = np.array(outlier_spectra)
        save_dir = save_dir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)

        fig = plt.figure(figsize=(10, 6))
        for i, sp in enumerate(outlier_spectra[:max_samples]):
            plt.plot(self.wavenumber_data, sp, label=f"Outlier {i + 1}: {outlier_labels[i]}")
        plt.xlabel('Raman Shift (cm$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Removed Outlier Spectra')
        plt.legend(fontsize=8, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        outlier_plot_path = os.path.join(save_dir, 'Removed_Outlier_Spectra.png')
        fig.savefig(outlier_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        report_path = os.path.join(save_dir, 'outlier_removal_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('filename\tmd_squared\tthreshold\tpc1\tpc2\n')
            for record in self.outlier_info:
                f.write(f"{record['filename']}\t{record['md_squared']:.6f}\t{record['threshold']:.6f}\t{record['pc1']:.6f}\t{record['pc2']:.6f}\n")

        print(f"  🔍 已生成异常样本展示图: {outlier_plot_path}")
        print(f"  📄 已生成异常样本剔除报告: {report_path}")

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
    task_label = None
    if target_file is not None:
        with open(target_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) >= 3:
                task_label = lines[0].strip()
                data_folder = lines[1].strip()
                save_folder = lines[2].strip()
            elif len(lines) == 2:
                # 向后兼容：如果只有 2 行，则为旧格式（不带标签）
                data_folder, save_folder = lines[0].strip(), lines[1].strip()
                task_label = "拉曼预处理"
    
    if data_folder is None or save_folder is None:
        print("❌ 未找到 target 文件或文件格式错误。")
        print("   预期格式：")
        print("   第一行: 任务标签 (如 '预处理')")
        print("   第二行: 数据文件夹路径")
        print("   第三行: 保存文件夹路径")
        exit()
    
    print(f"🔎 已找到 target 文件: {target_file}")
    print(f"   任务标签: {task_label}")
    print(f"   数据路径: {data_folder}")
    print(f"   保存路径: {save_folder}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 🎛️  统一配置中心 - 所有处理选项在此集中管理
    # ═══════════════════════════════════════════════════════════════════
    config = {
        # ━━━ 数据加载选项 ━━━
        'has_subfolders': False,              # 数据是否有子文件夹结构
        
        # ━━━ 处理流程选项 ━━━
        'plot_only': False,                   # 只画图模式：True = 直接加载已处理数据并画图
        'generate_plots': True,               # 画图总开关：True = 生成并保存图表
        
        # ━━━ 处理步骤开关 ━━━
        'background_subtraction': True,       # 是否扣除背景光谱
        'baseline_correction': True,          # 是否进行基线校正 (airPLS 算法)
        'remove_outliers': False,              # 是否剔除异常样本 (PCA Mahalanobis)
        'denoising': True,                    # 是否进行自适应降噪 (小波 + SG 滤波)
        
        # ━━━ 归一化选项 ━━━
        'normalization': 'peak',              # 归一化方法: 'area' | 'maxmin' | 'peak' | None
        'peak_position': 400,                 # 峰值归一化时的目标波数 (cm⁻¹)，仅当 normalization='peak' 时生效
    }
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n" + "="*50 + "\n🚀 拉曼光谱自动化处理流水线已启动\n" + "="*50)
    print(f"✓ 配置已加载:")
    print(f"  - 子文件夹支持: {config['has_subfolders']}")
    print(f"  - 只画图模式: {config['plot_only']}")
    print(f"  - 生成图表: {config['generate_plots']}")
    print(f"  - 处理步骤: 背景扣除={config['background_subtraction']}, 基线校正={config['baseline_correction']}, "
          f"剔除异常={config['remove_outliers']}, 降噪={config['denoising']}")
    print(f"  - 归一化方法: {config['normalization']}" + (f" (目标波数: {config['peak_position']} cm⁻¹)" if config['normalization'] == 'peak' else ""))
    
    if config['plot_only']:
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
                # 保存处理后用来画图的光谱矩阵
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
                
                print(f"💾 处理后光谱矩阵已保存: {processed_matrix_path}")
                print(f"💾 处理后光谱统计信息已保存: {processed_stats_path}")
            else:
                print("❌ 未找到处理后的数据文件"); exit()
        else:
            print(f"❌ 数据保存目录不存在: {data_save_dir}"); exit()
    else:
        try:
            processor.load_spectra_data(data_folder, has_subfolders=config['has_subfolders'])
        except Exception as e:
            print(f"\n❌ 加载数据失败: {e}"); exit()
        
        # 构建处理流程配置
        steps_config = {
            'background_subtraction': config['background_subtraction'],
            'baseline_correction': config['baseline_correction'],
            'remove_outliers': config['remove_outliers'],
            'denoising': config['denoising'],
            'normalization': config['normalization'],
            'peak_position': config['peak_position']
        }
        
        try:
            processor.process_pipeline(steps_config)
        except Exception as e:
            print(f"\n❌ 处理发生错误:\n"); traceback.print_exc(); exit()
        
        data_save_dir = os.path.join(save_folder, "Data")
        processor.save_processed_data(data_save_dir, keep_structure=config['has_subfolders'])
        
        # 保存原始读取的光谱矩阵
        raw_matrix_path = os.path.join(save_folder, "raw_spectra_matrix.txt")
        raw_matrix = np.column_stack([processor.wavenumber_data] + [processor.spectra_data[i] for i in range(len(processor.spectra_data))])
        np.savetxt(raw_matrix_path, raw_matrix, delimiter="\t", fmt="%.6f", 
                   header=f"Wavenumber(cm-1)\t" + "\t".join([f"Spectrum_{i}" for i in range(len(processor.spectra_data))]))
        
        # 计算原始数据的统计信息
        raw_mean = np.mean(processor.spectra_data, axis=0)
        raw_std = np.std(processor.spectra_data, axis=0)
        raw_var = np.var(processor.spectra_data, axis=0)
        raw_stats_matrix = np.column_stack([processor.wavenumber_data, raw_mean, raw_std, raw_var])
        raw_stats_path = os.path.join(save_folder, "raw_spectra_statistics.txt")
        np.savetxt(raw_stats_path, raw_stats_matrix, delimiter="\t", fmt="%.6f",
                   header="Wavenumber(cm-1)\tMean\tStd_Dev\tVariance")
        
        print(f"💾 原始光谱矩阵已保存: {raw_matrix_path}")
        print(f"💾 原始光谱统计信息已保存: {raw_stats_path}")
        
        # 保存处理后用来画图的光谱矩阵
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
        
        print(f"💾 处理后光谱矩阵已保存: {processed_matrix_path}")
        print(f"💾 处理后光谱统计信息已保存: {processed_stats_path}")

    # 6. 生成并保存图表
    if config['generate_plots']:
        print("\n--- 正在生成并保存图表 ---")
        plot_save_dir = os.path.join(save_folder, "Plots")
        
        # 1. 抽样原始光谱预览 (仅完整处理模式下存在)
        show_original_spectra = not config['plot_only']
        processor.plot_spectra(n_samples=5, show_original=show_original_spectra, show_processed=True, save_dir=plot_save_dir)
        
        # 2. 调用全新的 2x3 综合画图面板（取代旧版的散碎拼图）
        plot_comprehensive_dashboard(processor, save_dir=plot_save_dir)
        
        print("📸 图片及分析数据已全部生成并保存！")
    else:
        print("\n--- 🚫 画图开关已关闭 (generate_plots=False) ---")
    
    print(f"\n🎉 全部任务已完成！请前往 {save_folder} 查看结果。")
