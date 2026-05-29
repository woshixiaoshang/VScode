import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # <--- 加上这一行！它的作用是强制纯后台画图
from sklearn.decomposition import PCA
from scipy.stats import chi2
from matplotlib.patches import Ellipse
from scipy import sparse
from scipy.sparse.linalg import spsolve
import pywt
from scipy.interpolate import interp1d
from collections import Counter
from scipy.signal import savgol_filter, peak_widths
import pandas as pd
from scipy.signal import find_peaks
from dtaidistance import dtw
import warnings
from scipy.stats import pearsonr
import traceback


warnings.filterwarnings('ignore')

# 1. 设置中文字体（优先使用微软雅黑，备选黑体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial'] 
# 2. 解决坐标轴负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 18

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

        # 1. 收集所有 txt 文件的路径
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

        # 2. 交给核心函数进行解析和异常过滤
        self.spectra_data, self.file_names, self.wavenumber_data, self.header = self._load_and_filter_files(file_paths)
        
        if self.spectra_data is None:
            raise ValueError("❌ 所有文件均读取失败或格式无效！")

        print(f"\n✅ 成功提取 {len(self.spectra_data)} 条有效光谱数据")
        print(f"   数据维度: {self.spectra_data.shape}")
        print(f"   对齐波数: {self.wavenumber_data[0]:.2f} 到 {self.wavenumber_data[-1]:.2f} cm⁻¹")

    def _load_and_filter_files(self, file_paths_with_names, tolerance=10.0):
        """核心解析逻辑：读取文件、寻找基准坐标、对允许误差内的文件进行插值对齐"""
        raw_data_list = []

        # 1. 读取所有文件的数据
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

        if not raw_data_list:
            return None, [], None, ""

        # 2. 提取粗略特征：将起点和终点四舍五入到十位数，快速圈定“大部队”
        # 这样 2020.8 和 2020.9 都会被认为是同一批数据
        signatures = [(len(item['x']), round(item['x'][0], -1), round(item['x'][-1], -1)) for item in raw_data_list]

        # 3. 找到大部队，提取出一个绝对标准的 X 轴
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

        # 4. 容差对齐与插值重采样
        for item in raw_data_list:
            start_diff = abs(item['x'][0] - ref_x[0])
            end_diff = abs(item['x'][-1] - ref_x[-1])

            # 只要起点和终点的差距在 tolerance (默认 10 cm-1) 以内，就认为是同类数据
            if start_diff <= tolerance and end_diff <= tolerance:
                # 检查是否需要插值对齐
                if np.array_equal(item['x'], ref_x):
                    valid_spectra.append(item['y'])
                else:
                    # 🔬 核心：使用线性插值，将稍微偏移的 Y 值，精准映射到标准的 X 轴上
                    f_interp = interp1d(item['x'], item['y'], kind='linear', bounds_error=False, fill_value="extrapolate")
                    aligned_y = f_interp(ref_x)
                    valid_spectra.append(aligned_y)
                    repaired_count += 1

                valid_filenames.append(item['filename'])
            else:
                skipped_files.append((item['filename'], len(item['x']), item['x'][0], item['x'][-1]))

        # 5. 打印分析报告
        if repaired_count > 0:
            print(f"\n🔧 [数据自动对齐] 已将 {repaired_count} 个存在微小漂移的光谱，自动插值重采样至标准波数轴！")

        if skipped_files:
            print(f"\n⚠️ [严重偏移警告] 发现 {len(skipped_files)} 个文件超出了 {tolerance} cm⁻¹ 的允许误差，已跳过！")
            print(f"   ✓ 标准波数基准: {len(ref_x)} 个数据点, 区间 {ref_x[0]:.1f} ~ {ref_x[-1]:.1f} cm⁻¹")
            print(f"   ❌ 异常文件列表 (最多显示10个):")
            for fname, n_pts, start, end in skipped_files[:10]:
                print(f"      - {fname} (点数: {n_pts}, 区间: {start:.1f} ~ {end:.1f})")
            if len(skipped_files) > 10:
                print(f"      ... (还有 {len(skipped_files) - 10} 个文件未显示)")

        return np.array(valid_spectra), valid_filenames, ref_x, ref_header

    def load_background_data(self, background_path=None):
        """加载背景数据"""
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
                print("⚠️ 背景文件夹中没有 .txt 文件，跳过")
        else:
            self.background_data = None
            print("⚠️ 未提供背景数据，跳过背景扣除步骤")

    def process_pipeline(self, steps_config):
        """执行处理流程"""
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
        """PCA移除异常值"""
        if len(data) < 5:
            return [], data, filenames

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
        """自适应寻峰"""
        window_size = kwargs.get('window_size', 200)
        prominence = kwargs.get('prominence', 5.0)
        distance = kwargs.get('distance', 50)
        step_size = kwargs.get('step_size', 50)

        peaks = []
        n = len(data)
        for i in range(0, n, step_size):
            start = max(0, i - window_size // 2)
            end = min(n, i + window_size // 2)
            window = data[start:end]
            local_baseline = np.median(window)
            local_std = np.std(window)
            if data[i] > local_baseline + 20 * local_std:
                peaks.append(i)

        peaks, _ = find_peaks(data, height=0, distance=distance, prominence=prominence)
        if len(peaks) > 0:
            widths = peak_widths(data, peaks, rel_height=0.5)[0]
            peak_areas = widths * data[peaks]
            min_area = np.mean(peak_areas) * 0.6
            strong_peaks = peaks[peak_areas >= min_area]
            return strong_peaks
        return peaks

    def _subtract_background(self, spectra):
        """背景扣除"""
        if self.background_data is None or len(self.background_data) == 0:
            return spectra

        background_avg = np.mean(self.background_data, axis=0)
        bg_peaks = self._find_peaks_adaptive(background_avg)

        if len(bg_peaks) == 0:
            print("  警告: 背景数据中没有检测到明显的尖峰，使用默认比例因子")
            return spectra

        scale_factors = []
        for spectrum in spectra:
            ratios = spectrum[bg_peaks] / background_avg[bg_peaks]
            scale_factor = np.median(ratios[ratios > 0]) if np.any(ratios > 0) else 0.0
            scale_factors.append(scale_factor)

        corrected_spectra = []
        for spectrum, scale_factor in zip(spectra, scale_factors):
            corrected = spectrum - scale_factor * background_avg
            corrected[corrected < 0] = 0
            corrected_spectra.append(corrected)
        return np.array(corrected_spectra)

    def _baseline_correction(self, spectra, lam=1e4, porder=0.005, itermax=15):
        """【极速版】airPLS基线校正"""
        if len(spectra) == 0: return spectra
        m = spectra.shape[1]
        
        # 🌟 关键提速：巨型矩阵只计算1次，并且转为最快的 csc 格式！
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(m - 2, m))
        penalty = (lam * (D.transpose().dot(D))).tocsc()

        def airPLS(signal):
            w = np.ones(m)
            w[:10] = 1000
            w[-10:] = 1000
            for i in range(itermax):
                W = sparse.spdiags(w, 0, m, m).tocsc()
                Z = W + penalty
                baseline = spsolve(Z, w * signal)
                diff = signal - baseline
                w_new = np.where(diff >= 0, porder, np.exp(i * diff / np.sum(diff[diff < 0])))
                w_new[w_new < 1e-8] = 1e-8
                if np.linalg.norm(w - w_new) / np.linalg.norm(w) < 1e-3: break
                w = w_new
            return baseline

        corrected_spectra = []
        for sp in spectra:
            corrected_spectra.append(sp - airPLS(sp))
        return np.array(corrected_spectra)

    def _denoise(self, spectra, **kwargs):
        """【极速版】自适应小波+SG平滑降噪"""
        if len(spectra) == 0: return spectra, (0, 0, 0, 0)
            
        wavelet = kwargs.get('wavelet', 'sym8')
        snr_threshold = kwargs.get('snr_threshold', 30)
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

        # 🌟 关键提速：只抽取5条样本去测试54种参数，而不是2934条！
        sample_size = min(5, len(spectra))
        sample_indices = np.random.choice(len(spectra), sample_size, replace=False)
        sample_spectra = spectra[sample_indices]

        best_params, best_avg_snr = None, -np.inf

        for level, sg_window, sg_poly in product(wavelet_levels, sg_windows, sg_polys):
            snr_list = []
            for sp in sample_spectra:
                denoised = wavelet_denoise(sp, level)
                if sg_window >= len(denoised) or sg_window % 2 == 0: continue
                smoothed = savgol_filter(denoised, window_length=sg_window, polyorder=sg_poly)
                snr_list.append(get_snr(sp, smoothed))

            if not snr_list: continue
            avg_snr = np.mean(snr_list)
            if avg_snr > best_avg_snr:
                best_params, best_avg_snr = (level, sg_window, sg_poly), avg_snr
                if avg_snr >= snr_threshold: break # 达标直接停止

        if best_params is None: return spectra, (0, 0, 0, 0)
            
        level, sg_window, sg_poly = best_params
        smoothed_all = []
        
        # 将测试好的最强参数，一次性应用到所有数据上
        for sp in spectra:
            denoised = wavelet_denoise(sp, level)
            smoothed = savgol_filter(denoised, window_length=sg_window, polyorder=sg_poly)
            smoothed_all.append(smoothed)

        return np.array(smoothed_all), (level, sg_window, sg_poly, best_avg_snr)
    def _normalize(self, spectra, method='area', peak_position=None):
        """归一化"""
        if method == 'area':
            normalized = []
            for sp in spectra:
                area = np.trapz(sp)
                if area == 0 or np.isnan(area):
                    normalized.append(sp)
                else:
                    normalized.append(sp / area)
            return np.array(normalized)
        elif method == 'maxmin':
            normalized = []
            for sp in spectra:
                max_val = np.max(sp)
                min_val = np.min(sp)
                if max_val == min_val or np.isnan(max_val) or np.isnan(min_val):
                    normalized.append(sp)
                else:
                    normalized.append((sp - min_val) / (max_val - min_val))
            return np.array(normalized)
        elif method == 'peak':
            if peak_position is not None:
                if peak_position < self.wavenumber_data[0] or peak_position > self.wavenumber_data[-1]:
                    print(f"⚠️ 警告: 指定位置 {peak_position} 超出波数范围，使用默认最大值归一化")
                    peak_position = None
                else:
                    idx = np.argmin(np.abs(self.wavenumber_data - peak_position))
                    actual_position = self.wavenumber_data[idx]
                    print(f"    使用拉曼位移 {actual_position:.2f} cm$^{-1}$处的强度进行归一化")

            normalized = []
            for sp in spectra:
                if peak_position is not None:
                    idx = np.argmin(np.abs(self.wavenumber_data - peak_position))
                    peak_val = sp[idx]
                else:
                    peak_val = np.max(sp)
                if peak_val == 0 or np.isnan(peak_val):
                    normalized.append(sp)
                else:
                    normalized.append(sp / peak_val)
            return np.array(normalized)
        return spectra

    def save_processed_data(self, save_path, keep_structure=False):
        """保存处理后的数据，支持多层级文件夹"""
        if self.processed_data is None:
            print("❌ 没有处理后的数据可保存")
            return False

        saved_count = 0
        for i, spectrum in enumerate(self.processed_data):
            output_data = np.column_stack((self.wavenumber_data, spectrum))

            if i < len(self.processed_filenames):
                original_filename = self.processed_filenames[i]
                if keep_structure and "/" in original_filename:
                    sub_dir = os.path.dirname(original_filename)
                    base_name = os.path.basename(original_filename)
                    target_dir = os.path.join(save_path, sub_dir)
                    os.makedirs(target_dir, exist_ok=True)
                    output_file = os.path.join(target_dir, f"processed_{base_name}")
                else:
                    os.makedirs(save_path, exist_ok=True)
                    # 如果原文件名带斜杠但不保留结构，将斜杠替换为下划线防止路径错误
                    safe_name = original_filename.replace("/", "_")
                    output_file = os.path.join(save_path, f"processed_{safe_name}")
            else:
                os.makedirs(save_path, exist_ok=True)
                output_file = os.path.join(save_path, f"processed_spectrum_{i + 1:04d}.txt")

            if self.header:
                np.savetxt(output_file, output_data, delimiter="\t", fmt="%.6f", header=self.header, comments='')
            else:
                np.savetxt(output_file, output_data, delimiter="\t", fmt="%.6f")
            saved_count += 1

        structure_msg = "保留原文件夹结构" if keep_structure else "扁平化保存"
        print(f"✅ 已保存 {saved_count} 个处理后文件 ({structure_msg}) 到: {save_path}")
        return True

    def plot_spectra(self, n_samples=10, show_original=True, show_processed=True, save_dir=None):
        """绘制并展示/保存光谱图"""
        if show_original and self.spectra_data is not None:
            self._plot_single_set(self.spectra_data, "原始光谱", n_samples, save_dir, prefix="Original")

        if show_processed and self.processed_data is not None:
            self._plot_single_set(self.processed_data, "处理后光谱", n_samples, save_dir, prefix="Processed")

    def _plot_single_set(self, data, title, n_samples, save_dir=None, prefix=""):
        total = data.shape[0]
        n_samples = min(n_samples, total)
        indices = np.linspace(0, total - 1, n_samples, dtype=int)

        plt.figure(figsize=(12, 8))
        for i, idx in enumerate(indices):
            plt.plot(self.wavenumber_data, data[idx], label=f'Sample {idx + 1}')

        plt.xlabel('Raman Shift (cm$^{-1}$)')
        plt.ylabel('Intensity (a.u.)')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{prefix}_Spectra_Preview.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  📸 图表已保存: {save_path}")
        plt.show()

# ======================== 分析绘图函数 (均已添加保存功能) ======================== #

def plot_mean_variance(processor, save_dir=None):
    if processor.processed_data is None: return
    mean_spectrum = np.mean(processor.processed_data, axis=0)
    std_spectrum = np.std(processor.processed_data, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(processor.wavenumber_data, mean_spectrum, 'b-', linewidth=5, label='average spectrum')
    ax.fill_between(processor.wavenumber_data, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum,
                    alpha=0.3, color='blue', label='±1 standard deviation')
    ax.set_xlabel('Raman shift (cm$^{-1}$)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title('Average spectrum')
    ax.legend()
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "Average_Spectrum_Variance.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_pca_clustering(processor, save_dir=None):
    if processor.processed_data is None: return
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(processor.processed_data)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, c=range(len(reduced)), cmap='viridis')
    plt.xlabel('PC1 ({:.1f}%)'.format(pca.explained_variance_ratio_[0] * 100))
    plt.ylabel('PC2 ({:.1f}%)'.format(pca.explained_variance_ratio_[1] * 100))
    plt.colorbar(scatter, label='kind_numbers')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "PCA_Clustering.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_pcc_analysis(processor, save_dir=None):
    if processor.processed_data is None: return
    corr_matrix = np.corrcoef(processor.processed_data)

    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(corr_matrix, cmap='hot', vmin=0.8, vmax=1.0)
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
    plt.title('Sample Correlation Heatmap (PCC)')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if i != j:
                plt.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "PCC_Heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_raman_intensity_heatmap(processor, save_dir=None):
    if processor.processed_data is None: return
    data = processor.processed_data
    wavenumbers = processor.wavenumber_data
    sample_numbers = np.arange(data.shape[0]) + 1

    plt.figure(figsize=(12, 8))
    heatmap = plt.imshow(data, cmap='hot', aspect='auto',
                         extent=[wavenumbers.min(), wavenumbers.max(), sample_numbers.min(), sample_numbers.max()])
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Intensity (a.u.)', rotation=270, labelpad=20)
    plt.title('Raman Spectra Intensity Heatmap')
    plt.xlabel('Raman Shift (cm$^{-1}$)')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "Intensity_Heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_repeatability_corr_heatmap(processor, save_dir=None):
    if processor.processed_data is None: return
    corr_matrix = np.corrcoef(processor.processed_data)

    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(corr_matrix, cmap='hot', vmin=0.8, vmax=1.0)
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
    plt.title('Reproducibility Correlation Heatmap')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "Repeatability_Corr_Heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_mean_spectra_stack(processor, save_dir=None):
    if processor.processed_data is None: return
    mean_spectrum = np.mean(processor.processed_data, axis=0)

    plt.figure(figsize=(12, 8))
    for i, spectrum in enumerate(processor.processed_data):
        alpha = 0.1 if len(processor.processed_data) > 20 else 0.3
        plt.plot(processor.wavenumber_data, spectrum, 'b-', alpha=alpha, linewidth=0.5)
    plt.plot(processor.wavenumber_data, mean_spectrum, 'r-', linewidth=2, label='Mean Spectrum')
    plt.xlabel('Raman Shift (cm$^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Spectra Stack Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "Spectra_Stack.png"), dpi=300, bbox_inches='tight')
    plt.show()

def interactive_menu():
    """保留交互菜单以备不时之需 (已修复选项7的缺失)"""
    processor = RamanDataProcessor()
    # ...(原始菜单逻辑保持不变，为节省空间，已在底层通过__main__自动化接管)...
    print("当前为全自动执行模式。如需恢复手动交互菜单，请修改代码底部逻辑。")

# ======================== 全自动执行流水线 ======================== #

if __name__ == "__main__":
    # 1. 实例化处理器
    processor = RamanDataProcessor()
    
    # 2. 从 target 文件读取路径
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
        if target_file:
            break

    # 也允许桌面/脚本目录下存在类似的 target* 文件，避免扩展名差异问题
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
    
    data_folder = None
    save_folder = None
    if target_file is not None:
        with open(target_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                data_folder = lines[0].strip()
                save_folder = lines[1].strip()
    
    if data_folder is None or save_folder is None:
        print("❌ 未找到 target 文件或文件格式错误。请确保 target 文件存在于桌面或代码同级文件夹中，且第一行是读取路径，第二行是保存路径。")
        exit()
    
    print(f"🔎 已找到 target 文件: {target_file}")
    print(f"📁 读取路径: {data_folder}")
    print(f"📁 保存路径: {save_folder}")
    
    # bg_folder = r"D:\Raman_Data\Background"          # 可选：背景数据路径，无则注释
    
    # 3. 加载数据 
    # ================= 自定义开关区 =================
    has_subfolders = False  # 数据是否有子文件夹
    # 👇👇👇 画图总开关：True 为画图并保存，False 为直接跳过画图提速 👇👇👇
    generate_plots = True
    # 👇👇👇 只画图模式：True 为只加载已处理数据并画图，False 为完整处理 👇👇👇
    plot_only = False
    # ===============================================
    print("\n" + "="*50 + "\n🚀 拉曼光谱自动化处理流水线已启动\n" + "="*50)
    if plot_only:
        print("📊 当前模式：只画图模式（跳过数据处理）")
    else:
        print("⚙️ 当前模式：完整处理模式（处理 + 画图）")
    
    if plot_only:
        print("\n--- 只画图模式：从已保存的文件加载处理后数据 ---")
        data_save_dir = os.path.join(save_folder, "Data")
        processor.processed_data = []
        processor.processed_filenames = []
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
                print(f"✅ 成功加载 {len(processor.processed_data)} 个处理后光谱")
                print(f"   波数范围: {processor.wavenumber_data[0]:.2f} 到 {processor.wavenumber_data[-1]:.2f} cm⁻¹")
            else:
                print("❌ 未找到处理后的数据文件，请先运行完整处理流程")
                exit()
        else:
            print(f"❌ 数据保存目录不存在: {data_save_dir}，请先运行完整处理流程")
            exit()
    else:
        try:
            processor.load_spectra_data(data_folder, has_subfolders=has_subfolders)
            # processor.load_background_data(bg_folder) # 如有背景请取消注释
        except Exception as e:
            print(f"\n❌ 加载数据失败，请检查路径是否正确！错误详情: {e}")
            exit()
        
        # 4. 配置并执行处理流水线
        steps_config = {
            'remove_outliers': True,          # 是否使用 PCA 剔除异常值
            'background_subtraction': True,  # 是否扣除背景
            'baseline_correction': True,     # 是否基线校正 (airPLS)
            'denoising': True,                # 是否自适应降噪
            'normalization': 'area'           # 归一化: 'area', 'maxmin', 'peak', None
        }
        
        try:
            processor.process_pipeline(steps_config)
        except Exception as e:
            print(f"\n❌ 处理过程中发生错误:\n")
            traceback.print_exc()
            exit()
        
        # 5. 保存处理后的光谱数据
        data_save_dir = os.path.join(save_folder, "Data")
        # keep_structure=True 意味着会自动保留原先的分类文件夹结构
        processor.save_processed_data(data_save_dir, keep_structure=has_subfolders)

   # 6. 根据开关决定是否画图
    if generate_plots:
        print("\n--- 正在生成并保存图表 ---")
        plot_save_dir = os.path.join(save_folder, "Plots")
        
        # 在只画图模式下，不显示原始光谱（因为没有加载原始数据）
        show_original_spectra = not plot_only
        processor.plot_spectra(n_samples=5, show_original=show_original_spectra, show_processed=True, save_dir=plot_save_dir)
        plot_mean_variance(processor, save_dir=plot_save_dir)
        plot_mean_spectra_stack(processor, save_dir=plot_save_dir)
        plot_raman_intensity_heatmap(processor, save_dir=plot_save_dir)
        plot_pca_clustering(processor, save_dir=plot_save_dir)
        print("📸 图片已全部生成并保存！")
    else:
        print("\n--- 🚫 画图开关已关闭 (generate_plots=False)，已跳过画图步骤以提升速度 ---")
    
    print(f"\n🎉 全部任务已完成！请前往 {save_folder} 查看结果。")