import os
import csv
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
# 只读取并处理 2800~3000 cm⁻¹ 区域；先筛查异常，再按该区域总面积归一化
# ============================================================
WAVENUMBER_MIN = 2800   # 截取起始波数 cm⁻¹
WAVENUMBER_MAX = 3000   # 截取终止波数 cm⁻¹
AREA_EPS = 1e-12
OUTLIER_PCA_SIGMA = 3.5
OUTLIER_PCC_SIGMA = 3.0
OUTLIER_AREA_SIGMA = 4.0


class RamanHighWavenumberProcessor:
    """高波数区域拉曼光谱处理（2800~3000 cm⁻¹，去噪、面积归一化、异常值筛选）"""

    def __init__(self):
        self.spectra_data = None
        self.file_names = None
        self.wavenumber_data = None
        self.header = ""
        self.processed_data = None
        self.processed_filenames = None
        self.area_factors = None
        self.outlier_report = None
        self.removed_outliers = None

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
    # 异常值筛选与面积归一化
    # ──────────────────────────────────────────────────────────
    def _robust_z(self, values):
        values = np.asarray(values, dtype=float)
        med = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - med))
        if not np.isfinite(mad) or mad < AREA_EPS:
            std = np.nanstd(values)
            return np.zeros_like(values) if std < AREA_EPS else (values - np.nanmean(values)) / std
        return 0.6745 * (values - med) / mad
    
    def _local_baseline_correct(self, spectra, anchor_lo=2800, anchor_hi=3000):
        wn = self.wavenumber_data
        lo_idx = np.argmin(np.abs(wn - anchor_lo))
        hi_idx = np.argmin(np.abs(wn - anchor_hi))

        # 左端单点，右端5点均值
        hi_range = slice(max(hi_idx - 4, 0), hi_idx + 1)

        corrected = []
       
        for sp in spectra:
            y_lo = sp[lo_idx]                # 左端单点
            y_hi = sp[hi_range].mean()       # 右端5点均值
            baseline = y_lo + (y_hi - y_lo) * (wn - wn[lo_idx]) / (wn[hi_idx] - wn[lo_idx] + 1e-12)
            corrected.append(sp - baseline)

        return np.array(corrected)

    def _peak_area_normalize(self, spectra):
        areas = np.trapz(spectra, self.wavenumber_data, axis=1)
        safe_areas = np.where(areas <= AREA_EPS, np.nan, areas)
        normalized = spectra / safe_areas[:, None]
        return normalized, areas

    def _filter_outliers_before_normalization(self, spectra, filenames):
        n_spec = len(spectra)
        reject = np.zeros(n_spec, dtype=bool)
        reasons = np.full(n_spec, "", dtype=object)

        def mark(mask, reason):
            for idx in np.where(mask)[0]:
                if not reject[idx]:
                    reasons[idx] = reason
                reject[idx] = True

        areas = np.trapz(spectra, self.wavenumber_data, axis=1)
        max_intensity = np.nanmax(spectra, axis=1)
        area_z = self._robust_z(areas)
        max_z = self._robust_z(max_intensity)

        invalid = (~np.isfinite(spectra).all(axis=1)) | (~np.isfinite(areas)) | (areas <= AREA_EPS)
        mark(invalid, "invalid_or_zero_area")
        mark(np.abs(area_z) > OUTLIER_AREA_SIGMA, f"raw_area_z>{OUTLIER_AREA_SIGMA}")
        mark(np.abs(max_z) > OUTLIER_AREA_SIGMA, f"raw_max_z>{OUTLIER_AREA_SIGMA}")

        report = []
        for i in range(n_spec):
            report.append({
                "index": i,
                "filename": filenames[i],
                "raw_area": areas[i],
                "raw_area_z": area_z[i],
                "raw_max": max_intensity[i],
                "raw_max_z": max_z[i],
                "pcc_to_median": np.nan,
                "pcc_z": np.nan,
                "pca_distance": np.nan,
                "pca_distance_z": np.nan,
                "status": "rejected" if reject[i] else "kept",
                "reason": reasons[i],
            })

        return ~reject, report

    def _filter_outliers_after_normalization(self, spectra_norm, keep_mask, report):
        kept_idx = np.where(keep_mask)[0]
        if len(kept_idx) < 5:
            return keep_mask, report

        spectra_kept = spectra_norm[keep_mask]
        median_spec = np.median(spectra_kept, axis=0)
        pcc = np.array([
            np.corrcoef(sp, median_spec)[0, 1] if np.std(sp) > AREA_EPS and np.std(median_spec) > AREA_EPS else np.nan
            for sp in spectra_kept
        ])
        pcc_z = self._robust_z(pcc)
        shape_reject = np.isfinite(pcc_z) & (pcc_z < -OUTLIER_PCC_SIGMA) & (pcc < 0.80) & (pcc < 0.80)

        pca_dist = np.full(len(kept_idx), np.nan)
        pca_z = np.full(len(kept_idx), np.nan)
        if len(kept_idx) >= 5 and spectra_kept.shape[1] >= 2:
            n_comp = min(5, len(kept_idx) - 1, spectra_kept.shape[1])
            X_scaled = StandardScaler().fit_transform(spectra_kept)
            scores = PCA(n_components=n_comp).fit_transform(X_scaled)
            center = np.median(scores, axis=0)
            spread = np.median(np.abs(scores - center), axis=0) / 0.6745
            spread = np.where(spread < AREA_EPS, 1.0, spread)
            pca_dist = np.sqrt(np.sum(((scores - center) / spread) ** 2, axis=1))
            pca_z = self._robust_z(pca_dist)
            shape_reject |= np.isfinite(pca_z) & (pca_z > OUTLIER_PCA_SIGMA)
            
            print(f"  PCC均值={np.nanmean(pcc):.4f}, 标准差={np.nanstd(pcc):.4f}, 最小值={np.nanmin(pcc):.4f}")

        for local_i, original_i in enumerate(kept_idx):
            report[original_i]["pcc_to_median"] = pcc[local_i]
            report[original_i]["pcc_z"] = pcc_z[local_i]
            report[original_i]["pca_distance"] = pca_dist[local_i]
            report[original_i]["pca_distance_z"] = pca_z[local_i]
            if shape_reject[local_i]:
                reason_parts = []
                if np.isfinite(pcc_z[local_i]) and pcc_z[local_i] < -OUTLIER_PCC_SIGMA:
                    reason_parts.append(f"normalized_pcc_z<-{OUTLIER_PCC_SIGMA}")
                if np.isfinite(pca_z[local_i]) and pca_z[local_i] > OUTLIER_PCA_SIGMA:
                    reason_parts.append(f"normalized_pca_z>{OUTLIER_PCA_SIGMA}")
                report[original_i]["status"] = "rejected"
                report[original_i]["reason"] = ";".join(reason_parts)
                keep_mask[original_i] = False

        return keep_mask, report

    def _apply_outlier_filter_and_normalization(self, spectra, filenames, outlier_filter=True):
        if outlier_filter:
            keep_mask, report = self._filter_outliers_before_normalization(spectra, filenames)
            
            n_before_norm = keep_mask.sum()
            print(f"  第一轮筛选后剩余: {n_before_norm} 条")
        else:
            keep_mask = np.ones(len(spectra), dtype=bool)
            report = [{
                "index": i,
                "filename": filenames[i],
                "raw_area": np.trapz(spectra[i], self.wavenumber_data),
                "raw_area_z": np.nan,
                "raw_max": np.nanmax(spectra[i]),
                "raw_max_z": np.nan,
                "pcc_to_median": np.nan,
                "pcc_z": np.nan,
                "pca_distance": np.nan,
                "pca_distance_z": np.nan,
                "status": "kept",
                "reason": "",
            } for i in range(len(spectra))]

        normalized_all, areas = self._peak_area_normalize(spectra)
        self.area_factors = areas

        if outlier_filter:
            keep_mask, report = self._filter_outliers_after_normalization(normalized_all, keep_mask, report)
            n_after_norm = keep_mask.sum()
            print(f"  第二轮筛选后剩余: {n_after_norm} 条")
        
            

        if keep_mask.sum() == 0:
            print("  ⚠️ 异常值筛选会剔除全部光谱，已自动保留全部光谱；请检查阈值或原始数据")
            keep_mask[:] = True
            for item in report:
                item["status"] = "kept"
                item["reason"] = "kept_by_safety_fallback"

        self.outlier_report = report
        self.removed_outliers = [report[i] for i in range(len(report)) if not keep_mask[i]]
        return normalized_all[keep_mask], [filenames[i] for i in np.where(keep_mask)[0]]

    # ──────────────────────────────────────────────────────────
    # 主流程
    # ──────────────────────────────────────────────────────────
    def process_pipeline(self, denoising=True, outlier_filter=True, area_normalization=True):
        print("\n开始高波数区域处理流程...")
        self.processed_data = self.spectra_data.copy()
        self.processed_filenames = self.file_names.copy()

        print("\n  ✓ 跳过全谱基线校正（水峰包区域不适合airPLS）")
        print("  ✓ 将在去噪后做局部线性基线校正（2800~3000 cm⁻¹）")

        if denoising:
            print("\n1. 去噪处理...")
            self.processed_data, denoise_params = self._denoise(self.processed_data)
            print(f"  去噪参数: 小波层数={denoise_params[0]:.1f}, SG窗口={denoise_params[1]}, 阶数={denoise_params[2]}")
            print(f"  平均SNR: {denoise_params[3]:.2f} dB")
        else:
            print("\n1. 跳过去噪步骤")

        # 局部线性基线校正（2800~3000 cm⁻¹两端锚点）
        #print("\n2. 局部基线校正（2800~3000 cm⁻¹线性基线）...")
        #self.processed_data = self._local_baseline_correct(self.processed_data, anchor_lo=2800, anchor_hi=3000)
        #print("  ✓ 完成，消除缓慢上升背景，峰形不变")

        if area_normalization:
            print("\n2. 异常值筛选与峰面积归一化...")
            n_before = len(self.processed_data)
            self.processed_data, self.processed_filenames = self._apply_outlier_filter_and_normalization(
                self.processed_data,
                self.processed_filenames,
                outlier_filter=outlier_filter
            )
            n_after = len(self.processed_data)
            print(f"  异常值筛选: 保留 {n_after}/{n_before} 条，剔除 {n_before - n_after} 条")
            print("  归一化方法: 每条谱除以 2800~3000 cm⁻¹ 区域总面积")
        else:
            print("\n2. 跳过面积归一化")

    
        

        def peak_area_ratio(wavenumber, spectrum,
                    peak1=(2835, 2870),
                    peak2=(2910, 2960)):
            m1 = (wavenumber >= peak1[0]) & (wavenumber <= peak1[1])
            m2 = (wavenumber >= peak2[0]) & (wavenumber <= peak2[1])
            area1 = np.trapz(spectrum[m1], wavenumber[m1])
            area2 = np.trapz(spectrum[m2], wavenumber[m2])
            return area1 / (area2 + 1e-12)

        ratios = np.array([peak_area_ratio(self.wavenumber_data, sp) 
                            for sp in self.processed_data])

        print(f"  均值:   {ratios.mean():.4f}")
        print(f"  中位数: {np.median(ratios):.4f}")
        print(f"  标准差: {ratios.std():.4f}")
        print(f"  范围:   {ratios.min():.4f} ~ {ratios.max():.4f}")
        print(f"  CV:     {ratios.std()/ratios.mean()*100:.1f}%")
        
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

    def save_outlier_report(self, save_path):
        if self.outlier_report is None:
            return False
        os.makedirs(save_path, exist_ok=True)
        report_file = os.path.join(save_path, "hw_outlier_report.csv")
        fields = [
            "index", "filename", "status", "reason",
            "raw_area", "raw_area_z", "raw_max", "raw_max_z",
            "pcc_to_median", "pcc_z", "pca_distance", "pca_distance_z"
        ]
        with open(report_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.outlier_report:
                writer.writerow({field: row.get(field, "") for field in fields})
        print(f"✅ 异常值筛选报告已保存至: {report_file}")
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
        ax1.set_title("Reproducibility Heatmap\n(Max-scaled display)", fontsize=11, fontweight="bold")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).set_label("Display scale", fontsize=8)

        # 图2：强度热图（原始）
        sort_idx = np.argsort(spectra.mean(axis=1))
        im2 = ax2.imshow(spectra[sort_idx], aspect="auto", cmap="inferno",
                         extent=ext, interpolation="nearest")
        ax2.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax2.set_ylabel("Spectrum (sorted by mean intensity)")
        ax2.set_title("Area-normalized Heatmap\n(sorted)", fontsize=11, fontweight="bold")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label("Area-normalized intensity", fontsize=8)

        # 图3：平均谱 ± SD
        mean_spec = spectra.mean(axis=0)
        std_spec = spectra.std(axis=0)
        ax3.plot(wavenumber, mean_spec, color="#2c3e50", lw=1.5, label="Mean")
        ax3.fill_between(wavenumber, mean_spec - std_spec, mean_spec + std_spec,
                         alpha=0.25, color="#2980b9", label="±1 SD")
        ax3.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax3.set_ylabel("Area-normalized intensity")
        ax3.set_title("Mean Spectrum ± SD\n(2800~3000 cm⁻¹)", fontsize=11, fontweight="bold")
        ax3.legend(fontsize=8)

        # 图4：抽样叠加谱（显示重复性）
        n_show = min(30, n_spec)
        idx_show = np.linspace(0, n_spec - 1, n_show, dtype=int)
        cmap_lines = plt.cm.viridis(np.linspace(0, 1, n_show))
        for ii, idx in enumerate(idx_show):
            ax4.plot(wavenumber, spectra[idx], color=cmap_lines[ii], lw=0.8, alpha=0.7)
        ax4.set_xlabel("Raman Shift (cm$^{-1}$)")
        ax4.set_ylabel("Area-normalized intensity")
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
        'outlier_filter': True,
        'area_normalization': True,
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
        processor.process_pipeline(
            denoising=config['denoising'],
            outlier_filter=config['outlier_filter'],
            area_normalization=config['area_normalization']
        )
    except Exception as e:
        print(f"\n❌ 处理失败:\n")
        traceback.print_exc()
        exit()

    data_save_dir = os.path.join(save_folder, "Data")
    processor.save_processed_data(data_save_dir)

    # outlier report保存到Data同层（save_folder下），不放进Data文件夹
    processor.save_outlier_report(save_folder)

    # ── 保存峰面积比csv，也放在Data同层 ──────────────────────
    print("\n--- 计算并保存2850/2930峰面积比 ---")
    wn = processor.wavenumber_data
    m1 = (wn >= 2835) & (wn <= 2870)
    m2 = (wn >= 2910) & (wn <= 2960)

    ratio_rows = []
    for i, sp in enumerate(processor.processed_data):
        area1 = float(np.trapz(sp[m1], wn[m1]))
        area2 = float(np.trapz(sp[m2], wn[m2]))
        ratio = area1 / (area2 + 1e-12)
        ratio_rows.append({
            "index": i,
            "filename": processor.processed_filenames[i],
            "ratio_2850_2930": round(ratio, 6),
            "area_2850": round(area1, 8),
            "area_2930": round(area2, 8),
        })

    ratio_csv_path = os.path.join(save_folder, "peak_area_ratio_2850_2930.csv")
    import csv as _csv
    with open(ratio_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = _csv.DictWriter(f, fieldnames=["index", "filename", "ratio_2850_2930", "area_2850", "area_2930"])
        writer.writeheader()
        writer.writerows(ratio_rows)

    ratio_arr = np.array([r["ratio_2850_2930"] for r in ratio_rows])
    print(f"  均值:   {ratio_arr.mean():.4f}")
    print(f"  中位数: {np.median(ratio_arr):.4f}")
    print(f"  标准差: {ratio_arr.std():.4f}")
    print(f"  CV:     {ratio_arr.std()/ratio_arr.mean()*100:.1f}%")
    print(f"✅ 峰面积比已保存至: {ratio_csv_path}")

    # ── 保存统计摘要 ───────────────────────────────────────────
    summary_path = os.path.join(save_folder, "peak_area_ratio_summary.txt")
    with open(summary_path, "w", encoding="utf-8-sig") as f:
        f.write("2850/2930 峰面积比统计摘要\n")
        f.write("=" * 35 + "\n")
        f.write(f"数据来源:   {data_folder}\n")
        f.write(f"有效光谱数: {len(ratio_rows)}\n")
        f.write(f"峰区间:     2835~2870 / 2910~2960 cm⁻¹\n")
        f.write("-" * 35 + "\n")
        f.write(f"均值:       {ratio_arr.mean():.6f}\n")
        f.write(f"中位数:     {np.median(ratio_arr):.6f}\n")
        f.write(f"标准差:     {ratio_arr.std():.6f}\n")
        f.write(f"SEM:        {ratio_arr.std()/np.sqrt(len(ratio_arr)):.6f}\n")
        f.write(f"CV:         {ratio_arr.std()/ratio_arr.mean()*100:.2f}%\n")
        f.write(f"最小值:     {ratio_arr.min():.6f}\n")
        f.write(f"最大值:     {ratio_arr.max():.6f}\n")
        f.write(f"25%分位数:  {np.percentile(ratio_arr, 25):.6f}\n")
        f.write(f"75%分位数:  {np.percentile(ratio_arr, 75):.6f}\n")
    print(f"✅ 统计摘要已保存至: {summary_path}")

    if config['generate_plots']:
        print("\n--- 正在生成图表 ---")
        plot_save_dir = os.path.join(save_folder, "Plots")
        processor.plot_dashboard(save_dir=plot_save_dir)
        print("📸 图表已保存")

    print(f"\n🎉 全部完成！结果保存在: {save_folder}")