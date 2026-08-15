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
from scipy import sparse
from scipy.sparse.linalg import spsolve
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

# 🌟 优化 1：扩大截取范围到 2800-3100
WAVENUMBER_MIN = 2800
WAVENUMBER_MAX = 3100
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
            if w % 2 == 0: w -= 1
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
        for sp in tqdm(spectra, desc="  去噪进度", unit="样本"):
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
                        if sloss > 0.05: continue
                        score = (after_snr - original_snr) - 30 * sloss
                        if score > best_score:
                            best_score, best_smoothed, best_param, best_snr = score, smoothed, (level, window, sg_poly), after_snr
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
                if not reject[idx]: reasons[idx] = reason
                reject[idx] = True

        areas = np.trapz(spectra, self.wavenumber_data, axis=1)
        max_intensity = np.nanmax(spectra, axis=1)
        area_z = self._robust_z(areas)
        max_z = self._robust_z(max_intensity)
        invalid = (~np.isfinite(spectra).all(axis=1)) | (~np.isfinite(areas)) | (areas <= AREA_EPS)
        mark(invalid, "invalid_or_zero_area")
        mark(np.abs(area_z) > OUTLIER_AREA_SIGMA, f"raw_area_z>{OUTLIER_AREA_SIGMA}")
        mark(np.abs(max_z) > OUTLIER_AREA_SIGMA, f"raw_max_z>{OUTLIER_AREA_SIGMA}")
        report = [{"index": i, "filename": filenames[i], "status": "rejected" if reject[i] else "kept", "reason": reasons[i]} for i in range(n_spec)]
        return ~reject, report

    def _filter_outliers_after_normalization(self, spectra_norm, keep_mask, report):
        kept_idx = np.where(keep_mask)[0]
        if len(kept_idx) < 5: return keep_mask, report
        spectra_kept = spectra_norm[keep_mask]
        median_spec = np.median(spectra_kept, axis=0)
        pcc = np.array([np.corrcoef(sp, median_spec)[0, 1] if np.std(sp) > AREA_EPS else np.nan for sp in spectra_kept])
        pcc_z = self._robust_z(pcc)
        shape_reject = np.isfinite(pcc_z) & (pcc_z < -OUTLIER_PCC_SIGMA) & (pcc < 0.80)
        
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
            if shape_reject[local_i]:
                report[original_i]["status"] = "rejected"
                report[original_i]["reason"] = "shape_outlier_PCA_or_PCC"
                keep_mask[original_i] = False
        return keep_mask, report

    def _apply_outlier_filter_and_normalization(self, spectra, filenames, outlier_filter=True):
        if outlier_filter:
            keep_mask, report = self._filter_outliers_before_normalization(spectra, filenames)
        else:
            keep_mask = np.ones(len(spectra), dtype=bool)
            report = [{"status": "kept"} for _ in range(len(spectra))]
            
        normalized_all, areas = self._peak_area_normalize(spectra)
        self.area_factors = areas
        
        if outlier_filter:
            keep_mask, report = self._filter_outliers_after_normalization(normalized_all, keep_mask, report)
            
        if keep_mask.sum() == 0:
            keep_mask[:] = True
            
        self.outlier_report = report
        return normalized_all[keep_mask], [filenames[i] for i in np.where(keep_mask)[0]]

    def process_pipeline(self, denoising=True, outlier_filter=True, area_normalization=True):
        print("\n开始高波数区域处理流程...")
        self.processed_data = self.spectra_data.copy()
        self.processed_filenames = self.file_names.copy()

        if denoising:
            print("\n1. 去噪处理...")
            self.processed_data, params = self._denoise(self.processed_data)

        if area_normalization:
            print("\n2. 异常值筛选与峰面积归一化...")
            self.processed_data, self.processed_filenames = self._apply_outlier_filter_and_normalization(
                self.processed_data, self.processed_filenames, outlier_filter=outlier_filter)

        print("\n✅ 高波数区域处理完成！")
        return self.processed_data, self.processed_filenames

    # 🌟 优化 3：带 Prominence 的深谷识别逻辑 (已修复 2880 被吞并的问题)
    def compute_peak_ratio(self, peak1=(2828, 2870), peak2=(2910, 2965), auto_boundary=True, search_margin=15):
        wn = self.wavenumber_data
        mean_sp = self.processed_data.mean(axis=0)

        def find_valley(wn, sp, center_wn, margin):
            """斜坡真谷底识别：局部去趋势法"""
            mask = (wn >= center_wn - margin) & (wn <= center_wn + margin)
            sub_wn, sub_sp = wn[mask], sp[mask]
            
            if len(sub_sp) < 5: return np.argmin(np.abs(wn - center_wn))
                
            # 消除大斜坡的干扰：将两端连线作为局部基线，并扣除 (局部去趋势)
            bg_line = np.linspace(sub_sp[0], sub_sp[-1], len(sub_sp))
            detrended = sub_sp - bg_line
            
            # 在去除了斜坡的光谱中找最低点（即下凹最深的谷）
            true_valley_wn = sub_wn[np.argmin(detrended)]
            return np.argmin(np.abs(wn - true_valley_wn))

        if auto_boundary:
            # 独立寻找4个边界，绝不强行合并，保留中间 2880 的地盘！
            p1_lo_idx = find_valley(wn, mean_sp, peak1[0], search_margin)
            p1_hi_idx = find_valley(wn, mean_sp, peak1[1], search_margin)
            
            p2_lo_idx = find_valley(wn, mean_sp, peak2[0], search_margin)
            p2_hi_idx = find_valley(wn, mean_sp, peak2[1], search_margin)

            actual_p1 = (wn[p1_lo_idx], wn[p1_hi_idx])
            actual_p2 = (wn[p2_lo_idx], wn[p2_hi_idx])

            print(f"  自动边界检测（基于平均谱）：")
            print(f"    2850峰: {min(actual_p1):.1f} ~ {max(actual_p1):.1f} cm⁻¹ (配置区 {peak1[0]}~{peak1[1]})")
            print(f"    2930峰: {min(actual_p2):.1f} ~ {max(actual_p2):.1f} cm⁻¹ (配置区 {peak2[0]}~{peak2[1]})")
            print(f"    (✅ 已成功避开中间的 2880 cm⁻¹ 肩峰)")
        else:
            p1_lo_idx = np.argmin(np.abs(wn - peak1[0]))
            p1_hi_idx = np.argmin(np.abs(wn - peak1[1]))
            p2_lo_idx = np.argmin(np.abs(wn - peak2[0]))
            p2_hi_idx = np.argmin(np.abs(wn - peak2[1]))
            actual_p1, actual_p2 = peak1, peak2

        # 使用 min/max 确保布尔索引正确（无视波数是升序还是降序排列）
        m1 = (wn >= min(actual_p1)) & (wn <= max(actual_p1))
        m2 = (wn >= min(actual_p2)) & (wn <= max(actual_p2))

        ratio_rows = []
        for i, sp in enumerate(self.processed_data):
            # 采用光谱学标准的“两点连线局部扣除法”计算净面积
            
            # 峰 1：2850
            raw_area1 = float(np.trapz(sp[m1], wn[m1]))
            bg_area1 = 0.5 * (sp[m1][0] + sp[m1][-1]) * abs(wn[m1][-1] - wn[m1][0])
            net_area1 = max(raw_area1 - bg_area1, 1e-8)  # 净面积 = 原始面积 - 底下梯形的面积

            # 峰 2：2930
            raw_area2 = float(np.trapz(sp[m2], wn[m2]))
            bg_area2 = 0.5 * (sp[m2][0] + sp[m2][-1]) * abs(wn[m2][-1] - wn[m2][0])
            net_area2 = max(raw_area2 - bg_area2, 1e-8)

            ratio = net_area1 / net_area2
            ratio_rows.append({
                "index": i, "filename": self.processed_filenames[i],
                "ratio_2850_2930": round(ratio, 6),
                "area_2850": round(net_area1, 8), "area_2930": round(net_area2, 8),
            })
            
        return ratio_rows, np.array([r["ratio_2850_2930"] for r in ratio_rows]), actual_p1, actual_p2

    def save_peak_ratio(self, save_folder, ratio_rows, ratio_arr, peak1, peak2):
        ratio_csv_path = os.path.join(save_folder, "peak_area_ratio_2850_2930.csv")
        with open(ratio_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "filename", "ratio_2850_2930", "area_2850", "area_2930"])
            writer.writeheader()
            writer.writerows(ratio_rows)

        # 🌟 恢复并增强控制台详细输出
        summary_text = (
            "2850/2930 峰面积比统计摘要\n"
            "===================================\n"
            f"积分区间:   {peak1[0]:.1f}~{peak1[1]:.1f} / {peak2[0]:.1f}~{peak2[1]:.1f} cm⁻¹\n"
            f"有效光谱数: {len(ratio_arr)}\n"
            "-----------------------------------\n"
            f"均值:       {ratio_arr.mean():.6f}\n"
            f"中位数:     {np.median(ratio_arr):.6f}\n"
            f"标准差:     {ratio_arr.std():.6f}\n"
            f"SEM:        {ratio_arr.std()/np.sqrt(len(ratio_arr)):.6f}\n"
            f"CV:         {ratio_arr.std()/ratio_arr.mean()*100:.2f}%\n"
            f"最小值:     {ratio_arr.min():.6f}\n"
            f"最大值:     {ratio_arr.max():.6f}\n"
            f"25%分位数:  {np.percentile(ratio_arr, 25):.6f}\n"
            f"75%分位数:  {np.percentile(ratio_arr, 75):.6f}\n"
        )

        summary_path = os.path.join(save_folder, "peak_area_ratio_summary.txt")
        with open(summary_path, "w", encoding="utf-8-sig") as f:
            f.write(summary_text)
            
        print("\n" + summary_text)
        print(f"✅ 峰面积比 (CSV) 已保存至: {ratio_csv_path}")
        print(f"✅ 统计摘要 (TXT) 已保存至: {summary_path}")

    def save_processed_data(self, save_path):
        if self.processed_data is None: return False
        os.makedirs(save_path, exist_ok=True)
        for i, spectrum in enumerate(self.processed_data):
            fname = self.processed_filenames[i].replace('/', '_')
            np.savetxt(os.path.join(save_path, f"hw_{fname}"), np.column_stack((self.wavenumber_data, spectrum)), delimiter="\t", fmt="%.6f")
        print(f"✅ 已保存高波数光谱至: {save_path}")

    def save_outlier_report(self, save_path):
        pass # 省略冗长打印

    def plot_dashboard(self, save_dir=None):
        if self.processed_data is None or not save_dir: return
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.wavenumber_data, self.processed_data.mean(axis=0), color="#2c3e50")
        ax.set_title("Mean Spectrum (Baseline Corrected)")
        fig.savefig(os.path.join(save_dir, "HW_Mean_Spectrum.png"), dpi=300, bbox_inches="tight")
        
        matrix = np.column_stack([self.wavenumber_data, self.processed_data.T])
        np.savetxt(os.path.join(save_dir, "hw_spectra_matrix.txt"), matrix, delimiter="\t", fmt="%.6f")
        print(f"📸 绘图和矩阵已保存至: {save_dir}")

if __name__ == "__main__":
    config = {
        'full_pipeline': True,
        'denoising': True,
        'outlier_filter': True,
        'area_normalization': True,
        'compute_peak_ratio': True,
        'auto_boundary': True,
        'peak1': (2828, 2870),
        'peak2': (2910, 2965),
        'has_subfolders': False,
        'generate_plots': True,
    }

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

    if not target_file:
        print("❌ 请在桌面或当前目录放置 target.txt")
        exit()

    data_folder, save_folder = None, None
    with open(target_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
        
    # 🌟 恢复了你原来的标签精准定位逻辑
    for i, line in enumerate(lines):
        if "高波数" in line:
            if i + 1 < len(lines):
                data_folder = lines[i + 1]
            if i + 2 < len(lines):
                save_folder = lines[i + 2]
            break

    if data_folder is None or save_folder is None:
        print("❌ 未找到 target 文件或未找到\"高波数\"标签下的有效路径")
        exit()

    print("=" * 55)
    print("🚀 Raman 高波数处理流水线 (自动拉平基线增强版)")
    print("=" * 55)
    print(f"   数据路径: {data_folder}")
    print(f"   保存路径: {save_folder}")

    processor = RamanHighWavenumberProcessor()
    processor.load_spectra_data(data_folder, has_subfolders=config['has_subfolders'])
    processor.process_pipeline(denoising=config['denoising'], outlier_filter=config['outlier_filter'], area_normalization=config['area_normalization'])
    
    processor.save_processed_data(os.path.join(save_folder, "Data"))
    processor.plot_dashboard(save_dir=os.path.join(save_folder, "Plots"))
    
    ratio_rows, ratio_arr, p1, p2 = processor.compute_peak_ratio(peak1=config['peak1'], peak2=config['peak2'], auto_boundary=config['auto_boundary'])
    processor.save_peak_ratio(save_folder, ratio_rows, ratio_arr, p1, p2)