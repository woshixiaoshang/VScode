import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # <-- 引入坐标轴格式化工具
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

WAVENUMBER_MIN = 2800
WAVENUMBER_MAX = 3000
AREA_EPS = 1e-12
OUTLIER_PCA_SIGMA = 3.5
OUTLIER_PCC_SIGMA = 3.0
OUTLIER_AREA_SIGMA = 4.0


class RamanHighWavenumberProcessor:

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
        self.spectra_data, self.file_names, self.wavenumber_data, self.header = self._load_and_align(file_paths)
        if self.spectra_data is None:
            raise ValueError("❌ 所有文件均读取失败或格式无效！")
        print(f"\n✅ 成功提取 {len(self.spectra_data)} 条有效光谱")
        print(f"   全谱波数范围: {self.wavenumber_data[0]:.2f} ~ {self.wavenumber_data[-1]:.2f} cm⁻¹")
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
        mask = (self.wavenumber_data >= wn_min) & (self.wavenumber_data <= wn_max)
        if mask.sum() == 0:
            raise ValueError(f"❌ 波数范围 {wn_min}~{wn_max} cm⁻¹ 在数据中不存在！")
        self.wavenumber_data = self.wavenumber_data[mask]
        self.spectra_data = self.spectra_data[:, mask]

    def load_processed_data(self, data_dir):
        """分析模式：直接读取已处理好的Data文件夹"""
        print(f"\n📂 分析模式：读取已处理数据从 {data_dir}")
        if not os.path.isdir(data_dir):
            raise ValueError(f"❌ Data文件夹不存在：{data_dir}")
        txt_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")])
        if not txt_files:
            raise ValueError("❌ Data文件夹内没有.txt文件")
        spectra, filenames, wavenumber = [], [], None
        for fname in txt_files:
            fpath = os.path.join(data_dir, fname)
            try:
                data = np.loadtxt(fpath)
                if data.ndim == 2 and data.shape[1] >= 2:
                    if wavenumber is None:
                        wavenumber = data[:, 0]
                    spectra.append(data[:, 1])
                    filenames.append(fname)
            except Exception:
                continue
        if not spectra:
            raise ValueError("❌ 没有成功读取任何光谱文件")
        self.processed_data = np.array(spectra)
        self.processed_filenames = filenames
        self.wavenumber_data = wavenumber
        print(f"✅ 读取 {len(spectra)} 条已处理光谱")
        print(f"   波数范围: {wavenumber[0]:.2f} ~ {wavenumber[-1]:.2f} cm⁻¹")

    def _denoise(self, spectra):
        if len(spectra) == 0:
            return spectra, (0, 0, 0, 0)
        wavelet = 'db8'
        level_candidates = [1, 2]
        sg_windows = [7, 11, 15, 21, 27]
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
            r = np.corrcoef(original, smoothed)[0, 1]
            return float(np.clip(1 - r, 0, 1))

        denoised_spectra, selected_params, snr_values = [], [], []
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
                        smoothed = savgol_filter(wavelet_smoothed, window_length=window, polyorder=sg_poly, mode='interp')
                        after_snr = robust_snr(smoothed)
                        sloss = shape_loss(sp, smoothed)
                        if sloss > 0.05:
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
        summary = (float(np.mean(levels)), Counter(windows).most_common(1)[0][0], Counter(polys).most_common(1)[0][0], float(np.mean(snr_values)))
        return np.array(denoised_spectra), summary

    def _robust_z(self, values):
        values = np.asarray(values, dtype=float)
        med = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - med))
        if not np.isfinite(mad) or mad < AREA_EPS:
            std = np.nanstd(values)
            return np.zeros_like(values) if std < AREA_EPS else (values - np.nanmean(values)) / std
        return 0.6745 * (values - med) / mad

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
            report.append({"index": i, "filename": filenames[i], "raw_area": areas[i], "raw_area_z": area_z[i], "raw_max": max_intensity[i], "raw_max_z": max_z[i], "pcc_to_median": np.nan, "pcc_z": np.nan, "pca_distance": np.nan, "pca_distance_z": np.nan, "status": "rejected" if reject[i] else "kept", "reason": reasons[i]})
        return ~reject, report

    def _filter_outliers_after_normalization(self, spectra_norm, keep_mask, report):
        kept_idx = np.where(keep_mask)[0]
        if len(kept_idx) < 5:
            return keep_mask, report
        spectra_kept = spectra_norm[keep_mask]
        median_spec = np.median(spectra_kept, axis=0)
        pcc = np.array([np.corrcoef(sp, median_spec)[0, 1] if np.std(sp) > AREA_EPS and np.std(median_spec) > AREA_EPS else np.nan for sp in spectra_kept])
        pcc_z = self._robust_z(pcc)
        shape_reject = np.isfinite(pcc_z) & (pcc_z < -OUTLIER_PCC_SIGMA) & (pcc < 0.80)
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
            print(f"  第一轮筛选后剩余: {keep_mask.sum()} 条")
        else:
            keep_mask = np.ones(len(spectra), dtype=bool)
            report = [{"index": i, "filename": filenames[i], "raw_area": np.trapz(spectra[i], self.wavenumber_data), "raw_area_z": np.nan, "raw_max": np.nanmax(spectra[i]), "raw_max_z": np.nan, "pcc_to_median": np.nan, "pcc_z": np.nan, "pca_distance": np.nan, "pca_distance_z": np.nan, "status": "kept", "reason": ""} for i in range(len(spectra))]
        normalized_all, areas = self._peak_area_normalize(spectra)
        self.area_factors = areas
        if outlier_filter:
            keep_mask, report = self._filter_outliers_after_normalization(normalized_all, keep_mask, report)
            print(f"  第二轮筛选后剩余: {keep_mask.sum()} 条")
        if keep_mask.sum() == 0:
            print("  ⚠️ 异常值筛选会剔除全部光谱，已自动保留全部；请检查阈值或原始数据")
            keep_mask[:] = True
            for item in report:
                item["status"] = "kept"
                item["reason"] = "kept_by_safety_fallback"
        self.outlier_report = report
        self.removed_outliers = [report[i] for i in range(len(report)) if not keep_mask[i]]
        return normalized_all[keep_mask], [filenames[i] for i in np.where(keep_mask)[0]]

    def process_pipeline(self, denoising=True, outlier_filter=True, area_normalization=True):
        print("\n开始高波数区域处理流程...")
        print("  ✓ 跳过全谱基线校正（水峰包区域不适合airPLS）")
        self.processed_data = self.spectra_data.copy()
        self.processed_filenames = self.file_names.copy()

        if denoising:
            print("\n1. 去噪处理...")
            self.processed_data, denoise_params = self._denoise(self.processed_data)
        else:
            print("\n1. 跳过去噪步骤")

        if area_normalization:
            print("\n2. 异常值筛选与峰面积归一化...")
            self.processed_data, self.processed_filenames = self._apply_outlier_filter_and_normalization(
                self.processed_data, self.processed_filenames, outlier_filter=outlier_filter)
        else:
            print("\n2. 跳过面积归一化")

        print("\n✅ 高波数区域处理完成！")
        return self.processed_data, self.processed_filenames

    def compute_peak_areas_and_ratios(self, peak1, peak2, peak3):
        """计算三个峰的面积以及峰面积比"""
        wn = self.wavenumber_data
        m1 = (wn >= peak1[0]) & (wn <= peak1[1])
        m2 = (wn >= peak2[0]) & (wn <= peak2[1])
        m3 = (wn >= peak3[0]) & (wn <= peak3[1])
        
        ratio_rows = []
        for i, sp in enumerate(self.processed_data):
            area1 = float(np.trapz(sp[m1], wn[m1]))
            area2 = float(np.trapz(sp[m2], wn[m2]))
            area3 = float(np.trapz(sp[m3], wn[m3]))
            
            # 峰面积比：P2/P1, P1/P3, P2/P3
            ratio_p2_p1 = area2 / (area1 + 1e-12)
            ratio_p1_p3 = area1 / (area3 + 1e-12)
            ratio_p2_p3 = area2 / (area3 + 1e-12)
            
            ratio_rows.append({
                "index": i, 
                "filename": self.processed_filenames[i], 
                "ratio_p2_p1": round(ratio_p2_p1, 6), 
                "ratio_p1_p3": round(ratio_p1_p3, 6),
                "ratio_p2_p3": round(ratio_p2_p3, 6),
                "area_p1": round(area1, 8), 
                "area_p2": round(area2, 8),
                "area_p3": round(area3, 8)
            })
            
        r21_arr = np.array([r["ratio_p2_p1"] for r in ratio_rows])
        r13_arr = np.array([r["ratio_p1_p3"] for r in ratio_rows])
        r23_arr = np.array([r["ratio_p2_p3"] for r in ratio_rows])
        return ratio_rows, r21_arr, r13_arr, r23_arr

    def save_peak_ratio(self, save_folder, ratio_rows, r21_arr, r13_arr, r23_arr, peak1, peak2, peak3):
        """保存峰面积和比值结果及摘要"""
        import csv as _csv
        ratio_csv_path = os.path.join(save_folder, "peak_areas_and_ratios.csv")
        with open(ratio_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = _csv.DictWriter(f, fieldnames=[
                "index", "filename", "ratio_p2_p1", "ratio_p1_p3", "ratio_p2_p3",
                "area_p1", "area_p2", "area_p3"
            ])
            writer.writeheader()
            writer.writerows(ratio_rows)

        summary_path = os.path.join(save_folder, "peak_area_ratio_summary.txt")
        with open(summary_path, "w", encoding="utf-8-sig") as f:
            f.write("高波数区域 峰面积与比值 统计摘要\n")
            f.write("=" * 45 + "\n")
            f.write(f"Peak 1 积分区间: {peak1[0]} ~ {peak1[1]} cm⁻¹\n")
            f.write(f"Peak 2 积分区间: {peak2[0]} ~ {peak2[1]} cm⁻¹\n")
            f.write(f"Peak 3 积分区间: {peak3[0]} ~ {peak3[1]} cm⁻¹\n")
            f.write(f"有效光谱数: {len(ratio_rows)}\n")
            f.write("-" * 45 + "\n")
            
            def write_stats(name, arr):
                f.write(f"【{name}】\n")
                f.write(f"  均值:       {arr.mean():.6f}\n")
                f.write(f"  中位数:     {np.median(arr):.6f}\n")
                f.write(f"  标准差(SD): {arr.std():.6f}\n")
                f.write(f"  变异系数(CV): {arr.std()/arr.mean()*100:.2f}%\n\n")
                
            write_stats("Ratio: Peak 2 / Peak 1", r21_arr)
            write_stats("Ratio: Peak 1 / Peak 3", r13_arr)
            write_stats("Ratio: Peak 2 / Peak 3", r23_arr)

        print(f"✅ 峰面积及比值已保存至: {ratio_csv_path}")
        print(f"✅ 统计摘要已保存至: {summary_path}")

    def plot_peak_boundaries(self, save_dir, peak1, peak2, peak3):
        """绘制平均光谱及峰边界以验证分割是否合理"""
        if self.processed_data is None or len(self.processed_data) == 0:
            return
            
        fig, ax = plt.subplots(figsize=(10, 5))
        mean_spec = np.mean(self.processed_data, axis=0)
        std_spec = np.std(self.processed_data, axis=0)
        
        # 绘制均值与误差带
        ax.plot(self.wavenumber_data, mean_spec, color='black', lw=1.5, label='Mean Spectrum')
        ax.fill_between(self.wavenumber_data, mean_spec - std_spec, mean_spec + std_spec, 
                        color='gray', alpha=0.3, label='±1 SD')
                        
        # 绘制砍峰边界
        colors = ['#e74c3c', '#2ecc71', '#3498db']
        peaks = [peak1, peak2, peak3]
        labels = [f"Peak 1 ({peak1[0]}-{peak1[1]})", 
                  f"Peak 2 ({peak2[0]}-{peak2[1]})", 
                  f"Peak 3 ({peak3[0]}-{peak3[1]})"]
                  
        for i, (p, color, label) in enumerate(zip(peaks, colors, labels)):
            ax.axvspan(p[0], p[1], color=color, alpha=0.2, label=label)
            # 添加垂直虚线强调边界
            ax.axvline(p[0], color=color, linestyle='--', lw=1)
            ax.axvline(p[1], color=color, linestyle='--', lw=1)
            
        ax.set_title("Peak Boundaries Segmentation Verification", fontsize=12, fontweight='bold')
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Normalized Intensity")
        ax.legend(loc='upper right', fontsize=9)
        
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, "Peak_Boundaries_Check.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ 砍峰边界检验图已保存至: {plot_path}")

    def save_processed_data(self, save_path):
        if self.processed_data is None:
            return False
        os.makedirs(save_path, exist_ok=True)
        saved_count = 0
        for i, spectrum in enumerate(self.processed_data):
            output_data = np.column_stack((self.wavenumber_data, spectrum))
            fname = self.processed_filenames[i].replace('/', '_') if i < len(self.processed_filenames) else f"hw_spectrum_{i+1:04d}.txt"
            np.savetxt(os.path.join(save_path, f"hw_{fname}"), output_data, delimiter="\t", fmt="%.6f", header=self.header, comments='')
            saved_count += 1
        return True

    def save_outlier_report(self, save_path):
        if self.outlier_report is None:
            return False
        os.makedirs(save_path, exist_ok=True)
        report_file = os.path.join(save_path, "hw_outlier_report.csv")
        fields = ["index", "filename", "status", "reason", "raw_area", "raw_area_z", "raw_max", "raw_max_z", "pcc_to_median", "pcc_z", "pca_distance", "pca_distance_z"]
        with open(report_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.outlier_report:
                writer.writerow({field: row.get(field, "") for field in fields})
        return True

    def plot_dashboard(self, save_dir=None):
        if self.processed_data is None:
            return
            
        spectra = self.processed_data
        wavenumber = self.wavenumber_data
        n_spec = len(spectra)
        
        # 1. 创建更宽的画布[cite: 2]
        fig = plt.figure(figsize=(15, 10))
        fig.patch.set_facecolor("#f7f9fc")
        
        # 2. 主网格：将画布分为左右两部分，左边占 1.2 份，右边占 2 份[cite: 2]
        outer_gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 2], 
                                     left=0.05, right=0.97, top=0.92, bottom=0.08, wspace=0.25)
        
        # ==============================================================
        # 左半部分：平均谱(上) + 热图(下)，无缝贴合[cite: 2]
        # ==============================================================
        left_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[0], 
                                                   height_ratios=[1, 1.2], width_ratios=[30, 1], 
                                                   hspace=0.0, wspace=0.05)
        
        ax_mean = fig.add_subplot(left_gs[0, 0])
        ax_heat = fig.add_subplot(left_gs[1, 0], sharex=ax_mean)
        cax_heat = fig.add_subplot(left_gs[1, 1])  # 左侧热图专属 Colorbar[cite: 2]
        
        # --- 左上：平均谱与误差带 ---
        mean_spec = spectra.mean(axis=0)
        std_spec = spectra.std(axis=0)
        ax_mean.plot(wavenumber, mean_spec, color="#2c3e50", lw=2, label="Mean")
        ax_mean.fill_between(wavenumber, mean_spec - std_spec, mean_spec + std_spec, 
                             alpha=0.25, color="#2980b9", label="±1 SD")
        
        y_max = np.max(mean_spec + std_spec)
        ax_mean.set_ylim(bottom=np.min(mean_spec - std_spec), top=y_max * 1.15)
        
        # 峰值检测 (自适应高波数区域)[cite: 2]
        def _detect_peaks_hw(spectrum, prominence_factor=0.05, distance_factor=0.02):
            prom_val = (spectrum.max() - spectrum.min()) * prominence_factor
            dist_val = max(1, int(len(spectrum) * distance_factor))
            peaks, _ = find_peaks(spectrum, prominence=prom_val, distance=dist_val)
            return peaks

        peak_indices = _detect_peaks_hw(mean_spec)
        peak_color = "#e74c3c"
        if len(peak_indices) > 0:
            for idx in peak_indices:
                pw, pi = wavenumber[idx], mean_spec[idx]
                ax_mean.scatter(pw, pi, color=peak_color, s=80, marker="^", zorder=5, edgecolors="darkred", linewidths=1.2)
                ax_mean.vlines(pw, 0, pi, colors=peak_color, linewidth=0.8, linestyles="--", alpha=0.5)
                ax_mean.text(pw, pi + (y_max * 0.03), f"{pw:.0f}", ha="center", va="bottom", 
                             fontsize=9, color=peak_color, fontweight="bold")
                
        ax_mean.set_ylabel("Normalized Intensity", fontsize=10)
        # 使用 FormatStrFormatter 强制去除 0.000 这类冗余的小数位[cite: 2]
        ax_mean.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax_mean.legend(fontsize=9, loc='upper left')
        ax_mean.grid(True, alpha=0.2)
        ax_mean.set_xlim(wavenumber[0], wavenumber[-1])
        ax_mean.margins(x=0)
        
        # 隐藏下边框实现无缝贴合[cite: 2]
        ax_mean.tick_params(axis='x', bottom=False, labelbottom=False)
        ax_mean.spines['bottom'].set_visible(False)
        
        # --- 左下：强度排序热图 ---
        # 从小到大排序。因为热图绘制从上往下，这样最亮的就能排在下方[cite: 2]
        sort_idx = np.argsort(spectra.max(axis=1))
        spectra_sorted = spectra[sort_idx]
        ext = [wavenumber[0], wavenumber[-1], n_spec, 0]
        
        heatmap_vmin = float(np.percentile(spectra_sorted, 1))
        heatmap_vmax = float(np.percentile(spectra_sorted, 99))
        im_heat = ax_heat.imshow(spectra_sorted, aspect="auto", cmap="inferno", 
                                 vmin=heatmap_vmin, vmax=heatmap_vmax, extent=ext, interpolation="nearest")
        
        ax_heat.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=10)
        ax_heat.set_ylabel("Spectrum (sorted by max intensity)", fontsize=10)
        ax_heat.yaxis.set_major_locator(MaxNLocator(prune='lower'))
        # 同步给热图设置最简坐标格式，确保两个 "0" 完全一致[cite: 2]
        ax_heat.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        
        ax_heat.set_xlim(wavenumber[0], wavenumber[-1])
        ax_heat.margins(x=0)
        
        cbar_heat = fig.colorbar(im_heat, cax=cax_heat)
        cbar_heat.set_label("Area-normalized intensity", fontsize=9)
        cbar_heat.ax.tick_params(labelsize=8)

        # ==============================================================
        # 右半部分：其余 4 张图 (2x2 网格)[cite: 2]
        # ==============================================================
        right_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[1], hspace=0.35, wspace=0.3)
        ax_repro = fig.add_subplot(right_gs[0, 0])
        ax_over = fig.add_subplot(right_gs[0, 1])
        ax_pcc = fig.add_subplot(right_gs[1, 0])
        ax_pca = fig.add_subplot(right_gs[1, 1])
        
        # 3. 重复性热图 (Max-scaled)
        spectra_max_norm = spectra / (spectra.max(axis=1, keepdims=True) + 1e-10)
        raman_cmap = LinearSegmentedColormap.from_list("raman", ["#0d1b2a", "#1f4e79", "#2e86c1", "#27ae60", "#f4d03f", "#e74c3c", "#ffffff"])
        im_repro = ax_repro.imshow(spectra_max_norm, aspect="auto", cmap=raman_cmap, extent=ext, interpolation="nearest")
        ax_repro.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=9)
        ax_repro.set_ylabel("Spectrum Index", fontsize=9)
        ax_repro.set_title("Reproducibility Heatmap", fontsize=11, fontweight="bold")
        fig.colorbar(im_repro, ax=ax_repro, fraction=0.046, pad=0.04).set_label("Max-scaled", fontsize=8)
        
        # 4. 叠加图
        n_show = min(30, n_spec)
        idx_show = np.linspace(0, n_spec - 1, n_show, dtype=int)
        cmap_lines = plt.cm.viridis(np.linspace(0, 1, n_show))
        for ii, idx in enumerate(idx_show):
            ax_over.plot(wavenumber, spectra[idx], color=cmap_lines[ii], lw=0.8, alpha=0.7)
        ax_over.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=9)
        ax_over.set_ylabel("Area-normalized intensity", fontsize=9)
        ax_over.set_title(f"Overlay (n={n_show} sampled)", fontsize=11, fontweight="bold")
        
        # 5. PCC矩阵
        pcc_matrix = np.corrcoef(spectra)
        cool_warm = LinearSegmentedColormap.from_list("cw", ["#2980b9", "#ecf0f1", "#c0392b"])
        im_pcc = ax_pcc.imshow(pcc_matrix, cmap=cool_warm, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
        fig.colorbar(im_pcc, ax=ax_pcc, fraction=0.046, pad=0.04).set_label("Pearson r", fontsize=8)
        mask_off = ~np.eye(n_spec, dtype=bool)
        pcc_vals = pcc_matrix[mask_off]
        ax_pcc.set_xlabel(f"mean PCC = {pcc_vals.mean():.4f} ± {pcc_vals.std():.4f}", fontsize=9)
        ax_pcc.set_ylabel("Spectrum Index", fontsize=9)
        ax_pcc.set_title("Pearson Correlation", fontsize=11, fontweight="bold")
        
        # 6. PCA
        X_scaled = StandardScaler().fit_transform(spectra)
        pca = PCA(n_components=min(2, n_spec, spectra.shape[1]))
        scores = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_ * 100
        ax_pca.scatter(scores[:, 0], scores[:, 1] if scores.shape[1] > 1 else np.zeros(n_spec), 
                       s=40, alpha=0.7, color="#2980b9", edgecolors="white", linewidths=0.5)
        ax_pca.axhline(0, color="gray", lw=0.5, ls=":")
        ax_pca.axvline(0, color="gray", lw=0.5, ls=":")
        ax_pca.set_xlabel(f"PC1 ({explained[0]:.1f}%)", fontsize=9)
        ax_pca.set_ylabel(f"PC2 ({explained[1]:.1f}%)" if len(explained) > 1 else "PC2", fontsize=9)
        ax_pca.set_title("PCA Score Plot", fontsize=11, fontweight="bold")
        
        fig.suptitle(f"High-Wavenumber Analysis  ·  {n_spec} Spectra  ·  {WAVENUMBER_MIN}~{WAVENUMBER_MAX} cm⁻¹", 
                     fontsize=15, fontweight="bold", y=0.98)
        
        # 保存图表及数据[cite: 2]
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, "HighWavenumber_Dashboard.svg"), dpi=300, bbox_inches="tight")
            fig.savefig(os.path.join(save_dir, "HighWavenumber_Dashboard.png"), dpi=300, bbox_inches="tight")
            
            matrix = np.column_stack([wavenumber, spectra.T])
            np.savetxt(os.path.join(save_dir, "hw_spectra_matrix.txt"), matrix, delimiter="\t", fmt="%.6f", 
                       header="Wavenumber(cm-1)\t" + "\t".join(f"Spectrum_{i}" for i in range(n_spec)))
            stats = np.column_stack([wavenumber, mean_spec, std_spec, spectra.var(axis=0)])
            np.savetxt(os.path.join(save_dir, "hw_spectra_statistics.txt"), stats, delimiter="\t", fmt="%.6f", 
                       header="Wavenumber(cm-1)\tMean\tStd_Dev\tVariance")
            print(f"  📸 高波数分析面板已保存至: {save_dir}")


# ============================================================
# 主程序入口
# target.txt 格式：
#   高波数
#   /原始数据文件夹路径
#   /保存文件夹路径
# ============================================================
if __name__ == "__main__":

    # ── 配置区：每次只需改这里 ────────────────────────────────
    config = {
        'full_pipeline': True,
        'denoising':          True,
        'outlier_filter':     True,
        'area_normalization': True,
        'compute_peak_ratio': True,

        # ==== 积分区间自定义 ====
        'peak1': (2828, 2870),   # 例如：2850峰边界
        'peak2': (2870, 2900),   # 例如：2885峰边界
        'peak3': (2910, 2965),   # 例如：2935峰边界

        'has_subfolders': False,
        'generate_plots': True,
    }
    # ─────────────────────────────────────────────────────────

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
        exit()

    print("=" * 55)
    print("🚀  Raman 高波数处理流水线")
    print("=" * 55)

    processor = RamanHighWavenumberProcessor()

    if config['full_pipeline']:
        processor.load_spectra_data(data_folder, has_subfolders=config['has_subfolders'])
        processor.process_pipeline(
            denoising=config['denoising'],
            outlier_filter=config['outlier_filter'],
            area_normalization=config['area_normalization']
        )
        data_save_dir = os.path.join(save_folder, "Data")
        processor.save_processed_data(data_save_dir)
        processor.save_outlier_report(save_folder)

        if config['generate_plots']:
            plot_dir = os.path.join(save_folder, "Plots")
            processor.plot_dashboard(save_dir=plot_dir)
    else:
        data_save_dir = os.path.join(save_folder, "Data")
        processor.load_processed_data(data_save_dir)

    # ── 峰面积与比值计算 ─────────────────────────────
    if config['compute_peak_ratio']:
        print(f"\n--- 计算峰面积及比值 ---")
        ratio_rows, r21_arr, r13_arr, r23_arr = processor.compute_peak_areas_and_ratios(
            config['peak1'], config['peak2'], config['peak3']
        )
        processor.save_peak_ratio(
            save_folder, ratio_rows, r21_arr, r13_arr, r23_arr, 
            config['peak1'], config['peak2'], config['peak3']
        )
        
        # 生成砍峰边界检查图
        plot_dir = os.path.join(save_folder, "Plots")
        processor.plot_peak_boundaries(plot_dir, config['peak1'], config['peak2'], config['peak3'])

    print(f"\n🎉 全部完成！结果保存在: {save_folder}")