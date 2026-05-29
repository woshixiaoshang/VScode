
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import chi2
from matplotlib.patches import Ellipse
from scipy import sparse
from scipy.sparse.linalg import spsolve
import pywt
from scipy.signal import savgol_filter, peak_widths
import pandas as pd
from scipy.signal import find_peaks
from dtaidistance import dtw
import warnings
from scipy.stats import pearsonr
import traceback

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'
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
        """
        加载光谱数据

        参数:
            folder_path: 文件夹路径
            has_subfolders: 是否有子文件夹（第二种数据格式）
        """
        print(f"正在加载数据从: {folder_path}")

        if has_subfolders:
            # 第二种数据格式：包含子文件夹
            all_data = []
            all_filenames = []
            all_wavenumbers = []
            all_headers = []
            subfolder_names = []

            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    print(f"  处理子文件夹: {subfolder}")
                    spectra, filenames, wavenumber, header = self._load_single_folder(subfolder_path)

                    if spectra is not None and len(spectra) > 0:
                        all_data.append(spectra)
                        all_filenames.append([f"{subfolder}/{f}" for f in filenames])
                        all_wavenumbers.append(wavenumber)
                        all_headers.append(header)
                        subfolder_names.append(subfolder)

            if not all_data:
                raise ValueError("未找到任何有效的光谱数据")

            # 合并所有数据
            self.spectra_data = np.vstack(all_data)
            self.file_names = [item for sublist in all_filenames for item in sublist]

            # 使用第一个子文件夹的波数和头信息
            self.wavenumber_data = all_wavenumbers[0]
            self.header = all_headers[0] if all_headers[0] else "#Wave\t#Intensity"

        else:
            # 第一种数据格式：直接包含txt文件
            self.spectra_data, self.file_names, self.wavenumber_data, self.header = self._load_single_folder(
                folder_path)

        print(f"✅ 成功加载 {len(self.spectra_data)} 条光谱数据")
        print(f"   数据维度: {self.spectra_data.shape}")
        print(f"   波数范围: {self.wavenumber_data[0]:.2f} 到 {self.wavenumber_data[-1]:.2f}")

    def _load_single_folder(self, folder_path):
        """加载单个文件夹中的光谱数据"""
        spectra = []
        filenames = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                try:
                    data = np.loadtxt(file_path)
                    if data.ndim == 2 and data.shape[1] >= 2:
                        y_values = data[:, 1]
                        spectra.append(y_values)
                        filenames.append(filename)
                except Exception as e:
                    print(f"  警告: 文件 {filename} 读取失败: {e}")
                    continue

        if not spectra:
            return None, [], None, ""

        # 统一数据长度
        max_length = max(len(sp) for sp in spectra)
        unified_spectra = []
        for sp in spectra:
            if len(sp) < max_length:
                padded_sp = np.pad(sp, (0, max_length - len(sp)), 'constant')
                unified_spectra.append(padded_sp)
            else:
                unified_spectra.append(sp)

        # 获取波数信息
        try:
            sample_file = os.path.join(folder_path, filenames[0])
            sample_data = np.loadtxt(sample_file)
            wavenumber_data = sample_data[:, 0] if sample_data.ndim == 2 and sample_data.shape[1] >= 2 else np.arange(
                len(unified_spectra[0]))
        except:
            wavenumber_data = np.arange(len(unified_spectra[0]))

        # 获取头信息
        try:
            with open(sample_file, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                if any(keyword in first_line.lower() for keyword in ['raman', 'wave', 'intensity', 'shift']):
                    header = first_line
                else:
                    header = "#Wave\t#Intensity"
        except:
            header = "#Wave\t#Intensity"

        return np.array(unified_spectra), filenames, wavenumber_data, header

    def load_background_data(self, background_path=None):
        """加载背景数据（可选）"""
        if background_path and os.path.isdir(background_path):
            self.background_data, _, _, _ = self._load_single_folder(background_path)
            print(f"✅ 成功加载 {len(self.background_data)} 条背景数据")
        else:
            self.background_data = None
            print("⚠️  未提供背景数据，跳过背景扣除步骤")

    def process_pipeline(self, steps_config):
        """
        执行处理流程

        参数:
            steps_config: 处理步骤配置方案
        """
        print("\n开始光谱数据处理流程...")

        # 初始化处理数据
        self.processed_data = self.spectra_data.copy()
        self.processed_filenames = self.file_names.copy()

        # 步骤1: 移除异常值
        if steps_config.get('remove_outliers', True) and len(self.processed_data) >= 10:
            print("\n1. 数据筛选 (移除异常值)...")
            outliers, inliers, inlier_filenames = self._remove_outliers(self.processed_data, self.processed_filenames)
            self.processed_data = inliers
            self.processed_filenames = inlier_filenames
            print(f"  移除了 {len(outliers)} 个异常样本，保留 {len(self.processed_data)} 个样本")
        else:
            print("\n1. 跳过数据筛选步骤")

        # 步骤2: 背景扣除
        if steps_config.get('background_subtraction', False) and self.background_data is not None:
            print("\n2. 背景扣除...")
            self.processed_data = self._subtract_background(self.processed_data)
            print(f"  完成背景扣除")
        else:
            print("\n2. 跳过背景扣除步骤")

        # 步骤3: 基线校正
        if steps_config.get('baseline_correction', True):
            print("\n3. 基线校正...")
            self.processed_data = self._baseline_correction(self.processed_data)
            print(f"  完成基线校正")
        else:
            print("\n3. 跳过基线校正步骤")

        # 步骤4: 去噪
        if steps_config.get('denoising', True):
            print("\n4. 去噪处理...")
            self.processed_data, denoise_params = self._denoise(self.processed_data)
            print(
                f"  去噪参数: 小波层数={denoise_params[0]}, SG窗口={denoise_params[1]}, SG多项式阶数={denoise_params[2]}")
            print(f"  平均SNR: {denoise_params[3]:.2f} dB")
        else:
            print("\n4. 跳过去噪步骤")

        # 步骤5: 归一化
        norm_method = steps_config.get('normalization', 'area')
        if norm_method:
            print(f"\n5. 归一化处理 ({norm_method})...")

            # 如果是peak归一化，需要获取peak_position
            if norm_method == 'peak':
                if 'peak_position' in steps_config:
                    peak_position = steps_config['peak_position']
                else:
                    # 如果未指定，让用户交互式输入
                    print(f"当前波数范围: {self.wavenumber_data[0]:.1f} - {self.wavenumber_data[-1]:.1f} cm⁻¹")
                    try:
                        peak_position = float(input("请输入用于归一化的拉曼位移(cm⁻¹): "))
                    except:
                        print("⚠️  输入无效，使用默认最大值归一化")
                        peak_position = None

                self.processed_data = self._normalize(
                    self.processed_data,
                    method=norm_method,
                    peak_position=peak_position
                )
            else:
                self.processed_data = self._normalize(
                    self.processed_data,
                    method=norm_method
                )
            print(f"  完成 {norm_method} 归一化")
        else:
            print("\n5. 跳过归一化步骤")

        print("\n✅ 光谱数据处理完成！")
        return self.processed_data, self.processed_filenames

    def _remove_outliers(self, data, filenames, confidence=0.95):
        """使用PCA和置信椭圆移除异常值"""
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

        # 返回异常文件名、inliers数据和inlier_filenames
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
        """扣除背景"""
        if self.background_data is None or len(self.background_data) == 0:
            return spectra

        background_avg = np.mean(self.background_data, axis=0)
        bg_peaks = self._find_peaks_adaptive(background_avg)

        if len(bg_peaks) == 0:
            print("  警告: 背景数据中没有检测到明显的尖峰，使用默认比例因子")
            return spectra

        # 计算比例因子
        scale_factors = []
        for spectrum in spectra:
            ratios = spectrum[bg_peaks] / background_avg[bg_peaks]
            scale_factor = np.median(ratios[ratios > 0]) if np.any(ratios > 0) else 0.0
            scale_factors.append(scale_factor)

        # 应用背景扣除
        corrected_spectra = []
        for spectrum, scale_factor in zip(spectra, scale_factors):
            corrected = spectrum - scale_factor * background_avg
            corrected[corrected < 0] = 0
            corrected_spectra.append(corrected)

        return np.array(corrected_spectra)

    def _baseline_correction(self, spectra, lam=1e4, porder=0.005, itermax=15):
        """基线校正"""

        def airPLS(signal, lam=lam, porder=porder, itermax=itermax):
            m = len(signal)
            D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(m - 2, m))
            penalty = lam * (D.transpose().dot(D))

            w = np.ones(m)
            w[:10] = 1000
            w[-10:] = 1000

            for i in range(itermax):
                W = sparse.spdiags(w, 0, m, m)
                Z = W + penalty
                baseline = spsolve(Z, w * signal)
                diff = signal - baseline
                w_new = np.where(diff >= 0, porder, np.exp(i * diff / np.sum(diff[diff < 0])))
                w_new[w_new < 1e-8] = 1e-8
                if np.linalg.norm(w - w_new) / np.linalg.norm(w) < 1e-3:
                    break
                w = w_new

            return baseline

        corrected_spectra = []
        for sp in spectra:
            baseline = airPLS(sp)
            corrected_spectra.append(sp - baseline)

        return np.array(corrected_spectra)

    def _denoise(self, spectra, **kwargs):
        """自动去噪"""
        wavelet = kwargs.get('wavelet', 'sym8')
        snr_threshold = kwargs.get('snr_threshold', 30)

        from itertools import product

        wavelet_levels = [1, 2, 3]
        sg_windows = [5, 7, 9, 11, 15, 21]
        sg_polys = [2, 3, 4]
        best_result = None
        best_avg_snr = -np.inf

        def wavelet_denoise(y, level):
            coeffs = pywt.wavedec(y, wavelet=wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(y)))
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            return pywt.waverec(coeffs, wavelet)[:len(y)]

        def snr(original, denoised):
            noise = original - denoised
            signal_power = np.mean(original ** 2)
            noise_power = np.mean(noise ** 2)
            if noise_power == 0:
                return float('inf'), 0
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
            mse_val = np.mean(noise ** 2)
            return snr_db, mse_val

        for level, sg_window, sg_poly in product(wavelet_levels, sg_windows, sg_polys):
            snr_list, smoothed_all = [], []

            for sp in spectra:
                denoised = wavelet_denoise(sp, level)
                window = sg_window
                if window >= len(denoised) or window % 2 == 0:
                    continue

                smoothed = savgol_filter(denoised, window_length=window, polyorder=sg_poly)
                s, _ = snr(sp, smoothed)
                snr_list.append(s)
                smoothed_all.append(smoothed)

            if not snr_list:
                continue

            avg_snr = np.mean(snr_list)

            if avg_snr >= snr_threshold:
                return np.array(smoothed_all), (level, sg_window, sg_poly, avg_snr)

            if avg_snr > best_avg_snr:
                best_result = (np.array(smoothed_all), (level, sg_window, sg_poly, avg_snr))
                best_avg_snr = avg_snr

        if best_result is not None:
            return best_result

        return spectra, (0, 0, 0, 0)

    def _normalize(self, spectra, method='area', peak_position=None):
        """
        归一化处理

        参数:
            spectra: 输入光谱数据
            method: 归一化方法 ('area', 'maxmin', 'peak')
            peak_position: 指定用于归一化的拉曼位移位置(cm⁻¹)，仅当method='peak'时有效

        返回:
            归一化后的光谱数据
        """
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
                # 验证peak_position是否在波数范围内
                if peak_position < self.wavenumber_data[0] or peak_position > self.wavenumber_data[-1]:
                    print(f"⚠️  警告: 指定位置 {peak_position} 超出波数范围")
                    print(f"    使用默认最大值归一化")
                    peak_position = None
                else:
                    # 找到最接近用户指定拉曼位移的索引
                    idx = np.argmin(np.abs(self.wavenumber_data - peak_position))
                    actual_position = self.wavenumber_data[idx]
                    print(f"    使用拉曼位移 {actual_position:.2f} cm⁻¹处的强度进行归一化")

            normalized = []
            for sp in spectra:
                if peak_position is not None:
                    idx = np.argmin(np.abs(self.wavenumber_data - peak_position))
                    peak_val = sp[idx]
                else:
                    # 默认使用最大值
                    peak_val = np.max(sp)

                if peak_val == 0 or np.isnan(peak_val):
                    normalized.append(sp)
                else:
                    normalized.append(sp / peak_val)
            return np.array(normalized)

        else:
            return spectra

    def save_processed_data(self, save_path):
        """保存处理后的数据"""
        if self.processed_data is None:
            print("❌ 没有处理后的数据可保存")
            return False

        os.makedirs(save_path, exist_ok=True)

        saved_count = 0
        for i, spectrum in enumerate(self.processed_data):
            output_data = np.column_stack((self.wavenumber_data, spectrum))

            if i < len(self.processed_filenames):
                original_filename = self.processed_filenames[i]
                output_file = os.path.join(save_path, f"processed_{os.path.basename(original_filename)}")
            else:
                output_file = os.path.join(save_path, f"processed_spectrum_{i + 1:04d}.txt")

            if self.header:
                np.savetxt(output_file, output_data, delimiter="\t", fmt="%.6f", header=self.header, comments='')
            else:
                np.savetxt(output_file, output_data, delimiter="\t", fmt="%.6f")
            saved_count += 1

        print(f"✅ 已保存 {saved_count} 个处理后文件到: {save_path}")
        return True

    def plot_spectra(self, n_samples=10, show_original=True, show_processed=True):
        """绘制光谱图"""
        if show_original and self.spectra_data is not None:
            self._plot_single_set(self.spectra_data, "原始光谱", n_samples)

        if show_processed and self.processed_data is not None:
            self._plot_single_set(self.processed_data, "处理后光谱", n_samples)

    def _plot_single_set(self, data, title, n_samples):
        """绘制单个数据集"""
        total = data.shape[0]
        n_samples = min(n_samples, total)
        indices = np.linspace(0, total - 1, n_samples, dtype=int)

        plt.figure(figsize=(12, 8))
        for i, idx in enumerate(indices):
            plt.plot(self.wavenumber_data, data[idx], label=f'样本 {idx + 1}')

        plt.xlabel('拉曼位移 (cm⁻¹)')
        plt.ylabel('强度')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def interactive_menu():
    """交互式菜单"""
    processor = RamanDataProcessor()

    print("\n" + "=" * 60)
    print("拉曼光谱数据处理系统")
    print("=" * 60)

    while True:
        print("-" * 60)
        print("1. 加载数据")
        print("2. 设计处理步骤")
        print("3. 执行处理流程")
        print("4. 查看光谱")
        print("5. 保存结果")
        print("6. 光谱分析工具")
        print("7. 退出")
        print("-" * 60)

        choice = input("请选择操作 (1-7): ").strip()

        if choice == "1":
            folder = input("请输入原始数据路径: ").strip('"').strip()
            if not os.path.isdir(folder):
                print("❌ 路径无效！")
                continue

            has_subfolders = input("数据是否有子文件夹结构? (y/N): ").strip().lower() == 'y'
            processor.load_spectra_data(folder, has_subfolders)

            background_path = input("请输入背景数据路径（可选，直接回车跳过）: ").strip('"').strip()
            if background_path and os.path.isdir(background_path):
                processor.load_background_data(background_path)

        elif choice == "2":
            if processor.spectra_data is None:
                print("❌ 请先加载数据！")
                continue

            print("\n配置处理步骤:")
            print("-" * 40)

            steps_config = {}

            # 数据筛选
            print("异常值移除建议光谱数据量大于30条")
            if len(processor.spectra_data) >= 10:
                remove_outliers = input("是否移除异常值? (Y/n): ").strip().lower()
                steps_config['remove_outliers'] = remove_outliers != 'n'
            else:
                print("数据量少于10条，跳过异常值检测")
                steps_config['remove_outliers'] = False

            # 背景扣除
            if processor.background_data is not None:
                bg_sub = input("是否扣除背景? (Y/n): ").strip().lower()
                steps_config['background_subtraction'] = bg_sub != 'n'
            else:
                steps_config['background_subtraction'] = False

            # 基线校正
            baseline = input("是否进行基线校正? (Y/n): ").strip().lower()
            steps_config['baseline_correction'] = baseline != 'n'

            # 去噪
            denoise = input("是否进行去噪? (Y/n): ").strip().lower()
            steps_config['denoising'] = denoise != 'n'

            # 归一化
            print("\n归一化方法选择:")
            print("  a. 面积归一化 (area)")
            print("  b. 最大最小归一化 (maxmin)")
            print("  c. 峰强度归一化 (peak)")
            print("  d. 不归一化")

            norm_choice = input("请选择归一化方法 (a/b/c/d): ").strip().lower()
            if norm_choice == 'a':
                steps_config['normalization'] = 'area'
            elif norm_choice == 'b':
                steps_config['normalization'] = 'maxmin'
            elif norm_choice == 'c':
                steps_config['normalization'] = 'peak'
                # 如果选择peak归一化，询问峰位置
                print(f"\n当前波数范围: {processor.wavenumber_data[0]:.1f} - {processor.wavenumber_data[-1]:.1f} cm⁻¹")
                peak_input = input("请输入用于归一化的拉曼位移(cm⁻¹)(直接回车使用最大值归一化): ").strip()
                if peak_input:
                    try:
                        steps_config['peak_position'] = float(peak_input)
                    except:
                        print("⚠️  输入无效，将使用最大值归一化")
            else:
                steps_config['normalization'] = None

            processor.steps_config = steps_config
            print("\n✅ 处理步骤配置完成!")

        elif choice == "3":
            if processor.spectra_data is None:
                print("❌ 请先加载数据！")
                continue

            if not hasattr(processor, 'steps_config'):
                print("⚠️  使用默认配置进行处理")
                processor.steps_config = {
                    'remove_outliers': True,
                    'background_subtraction': False,
                    'baseline_correction': True,
                    'denoising': True,
                    'normalization': 'maxmin'
                }

            try:
                processor.process_pipeline(processor.steps_config)
            except Exception as e:
                print(f"❌ 处理过程中发生错误: {e}")
                traceback.print_exc()

        elif choice == "4":
            if processor.spectra_data is None:
                print("❌ 请先加载数据！")
                continue

            show_original = input("显示原始光谱? (Y/n): ").strip().lower() != 'n'
            show_processed = False
            if processor.processed_data is not None:
                show_processed = input("显示处理后光谱? (Y/n): ").strip().lower() != 'n'

            n_samples = input("显示多少条光谱? (默认10): ").strip()
            n_samples = int(n_samples) if n_samples.isdigit() else 10

            processor.plot_spectra(n_samples=n_samples,
                                   show_original=show_original,
                                   show_processed=show_processed)

        elif choice == "5":
            if processor.processed_data is None:
                print("❌ 没有处理后的数据可保存，请先执行处理流程！")
                continue

            save_path = input("请输入保存路径: ").strip('"').strip()
            processor.save_processed_data(save_path)

        elif choice == "6":
            if processor.processed_data is None:
                print("❌ 请先处理数据！")
                continue

            print("\n光谱分析工具:")
            print("-" * 40)
            print("1. 平均谱和方差图")
            print("2. PCC分析")
            print("3. 光谱强度热图")
            print("4. 平均谱堆叠图")
            print("5. PCA聚类可视化")
            print("6. 重复性相关系数热图")
            print("7. PLS-LDA机器学习+结果用SHAP展示")
            print("-" * 40)

            analysis_choice = input("请选择分析类型 (1-6): ").strip()

            if analysis_choice == "1":
                plot_mean_variance(processor)
            elif analysis_choice == "2":
                plot_pcc_analysis(processor)
            elif analysis_choice == "3":
                plot_raman_intensity_heatmap(processor)
            elif analysis_choice == "4":
                plot_mean_spectra_stack(processor)
            elif analysis_choice == "5":
                plot_pca_clustering(processor)
            elif analysis_choice == "6":
                plot_repeatability_corr_heatmap(processor)
            else:
                print("❌ 无效选择")

        elif choice == "7":
            print("感谢使用，再见！")
            break

        else:
            print("❌ 无效选择，请重新输入！")


def plot_mean_variance(processor):
    """绘制平均谱和方差图"""
    if processor.processed_data is None:
        return

    mean_spectrum = np.mean(processor.processed_data, axis=0)
    std_spectrum = np.std(processor.processed_data, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.plot(processor.wavenumber_data, mean_spectrum, 'b-', linewidth=5, label='average spectrum')
    ax.fill_between(processor.wavenumber_data,
                    mean_spectrum - std_spectrum,
                    mean_spectrum + std_spectrum,
                    alpha=0.3, color='blue', label='±1 standard deviation')
    ax.set_xlabel('Raman Shift (cm⁻¹)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title('Average spectrum')
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_pca_clustering(processor):
    """绘制PCA聚类"""
    if processor.processed_data is None:
        return
    kind_number = 0
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(processor.processed_data)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, c=range(len(reduced)), cmap='viridis')
    plt.xlabel('PC1 ({:.1f}%)'.format(pca.explained_variance_ratio_[0] * 100))
    plt.ylabel('PC2 ({:.1f}%)'.format(pca.explained_variance_ratio_[1] * 100))
    # plt.title('PCA')
    plt.colorbar(scatter, label='kind_numbers')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pcc_analysis(processor):
    """绘制PCC分析"""
    if processor.processed_data is None:
        return

    # 计算相关系数矩阵
    corr_matrix = np.corrcoef(processor.processed_data)

    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(corr_matrix,
                         cmap='hot',
                         vmin=0.8,
                         vmax=1.0)

    cbar = plt.colorbar(heatmap)
    cbar.set_label('相关系数', rotation=270, labelpad=20)
    plt.title('样本间相关系数热图 (PCC)')
    plt.xlabel('样本编号')
    plt.ylabel('样本编号')

    # 添加相关系数值
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if i != j:  # 不显示对角线的1.0
                plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                         ha='center', va='center',
                         color='white', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_raman_intensity_heatmap(processor):
    """
    绘制拉曼光谱强度热图
    横轴: 拉曼位移，纵轴: 光谱样本，颜色: 拉曼强度
    """
    if processor.processed_data is None:
        print("❌ 没有处理过的数据可绘制")
        return

    data = processor.processed_data
    wavenumbers = processor.wavenumber_data
    sample_numbers = np.arange(data.shape[0]) + 1

    plt.figure(figsize=(12, 8))

    # 注意: extent参数格式为 [xmin, xmax, ymin, ymax]
    heatmap = plt.imshow(data,
                         cmap='hot',
                         aspect='auto',
                         extent=[wavenumbers.min(), wavenumbers.max(),
                                 sample_numbers.max(), sample_numbers.min()])

    cbar = plt.colorbar(heatmap)
    cbar.set_label('拉曼强度', rotation=270, labelpad=20)
    plt.title('拉曼光谱强度热图')
    plt.xlabel('拉曼位移 (cm⁻¹)')
    plt.ylabel('样本编号')
    plt.tight_layout()
    plt.show()


def plot_repeatability_corr_heatmap(processor):
    """
    绘制重复性相关系数热图
    显示样本间的相关系数矩阵
    """
    if processor.processed_data is None:
        print("❌ 没有处理过的数据可绘制")
        return

    corr_matrix = np.corrcoef(processor.processed_data)

    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(corr_matrix,
                         cmap='hot',
                         vmin=0.8,
                         vmax=1.0)

    cbar = plt.colorbar(heatmap)
    cbar.set_label('相关系数', rotation=270, labelpad=20)
    plt.title('重复性相关系数热图')
    plt.xlabel('样本编号')
    plt.ylabel('样本编号')
    plt.tight_layout()
    plt.show()


def plot_mean_spectra_stack(processor):
    """绘制平均谱堆叠图"""
    if processor.processed_data is None:
        return

    mean_spectrum = np.mean(processor.processed_data, axis=0)
    std_spectrum = np.std(processor.processed_data, axis=0)

    plt.figure(figsize=(12, 8))

    # 绘制所有光谱的堆叠
    for i, spectrum in enumerate(processor.processed_data):
        alpha = 0.1 if len(processor.processed_data) > 20 else 0.3
        plt.plot(processor.wavenumber_data, spectrum, 'b-', alpha=alpha, linewidth=0.5)

    # 绘制平均谱
    plt.plot(processor.wavenumber_data, mean_spectrum, 'r-', linewidth=2, label='平均谱')

    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('光谱堆叠图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. 实例化处理器
    processor = RamanDataProcessor()
    
    # 2. 修改为你的实际本地数据路径 (路径前加 r 防止转义字符报错)
    data_folder = r"D:\aaaSCNU\0data\Raman\20260317\1052" 
    # bg_folder = r"D:\Raman_Data\Background" # 如果有背景数据就填路径，没有就注释掉
    
    # 3. 加载数据 (如果你的文件夹里没有子文件夹，has_subfolders 改为 False)
    print("--- 正在加载数据 ---")
    processor.load_spectra_data(data_folder, has_subfolders=False)
    # processor.load_background_data(bg_folder) # 如果有背景数据取消这行注释
    
    # 4. 一键配置你的处理流水线
    steps_config = {
        'remove_outliers': True,          # 剔除异常值
        'background_subtraction': False,  # 是否扣除背景
        'baseline_correction': True,      # 基线校正 (airPLS)
        'denoising': True,                # 小波+SG平滑降噪
        'normalization': 'maxmin'         # 归一化方式: 'area', 'maxmin', 'peak' 或 None
    }
    
    # 5. 执行处理
    print("\n--- 正在执行处理流程 ---")
    processor.process_pipeline(steps_config)
    
    # 6. 一键出图 (把你想看的图取消注释即可)
    print("\n--- 正在生成可视化图表 ---")
    # 对比原始光谱与处理后的光谱 (默认抽样展示5条)
    processor.plot_spectra(n_samples=5, show_original=True, show_processed=True)
    
    # 调用外部的分析画图函数
    plot_mean_variance(processor)         # 均值方差图
    plot_mean_spectra_stack(processor)    # 光谱堆叠图
    plot_raman_intensity_heatmap(processor) # 拉曼强度热图
    
    # 7. 保存处理结果 (取消注释并修改路径即可保存)
    save_folder = r"D:\aaaSCNU\0data\Raman\20260317\1052processed"
    processor.save_processed_data(save_folder)