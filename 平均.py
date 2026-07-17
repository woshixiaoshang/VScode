import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import re
import time

try:
    import openpyxl  # noqa: F401
except ImportError:
    openpyxl = None


def clean_filename(filename):
    """清理文件名中的特殊字符和坐标信息"""
    cleaned = re.sub(r'__X[^_]+__Y[^_]+__ElapsedTime_\d+', '', filename)
    cleaned = re.sub(r'[^\w.-]', '', cleaned)
    return cleaned.split('.txt')[0] + '.txt' if cleaned.endswith('.txt') else cleaned

def extract_wxd_id(filename):
    """
    提取 processed_ 和 .tmp 之间的 wxd 编号，例如：
    processed_wxd22DD.tmp_1__X...
    返回 22DD
    也支持 wxd 后面只有 2 位或 3 位的情况。
    """
    m = re.search(r'processed_(.*?)\.tmp', filename, re.IGNORECASE)
    if m:
        segment = m.group(1)
        inner = re.search(r'wxd([A-Za-z0-9]{2,4})', segment, re.IGNORECASE)
        if inner:
            return inner.group(1).upper()

    m = re.search(r'wxd([A-Za-z0-9]{2,4})', filename, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def get_sample_name(sample_path, root_folder):
    """根据样本路径生成样本名：有文件夹按文件夹名，没有文件夹按文件名"""
    if os.path.isfile(sample_path):
        return os.path.splitext(os.path.basename(sample_path))[0]

    if os.path.isdir(sample_path):
        return os.path.basename(sample_path)

    return os.path.basename(sample_path)


def process_sample(sample_path, root_folder, global_data_collector, sample_name=None):

    # ---------- 新增：文件列表 ----------
    if isinstance(sample_path, list):

        file_paths = sorted(sample_path)

        if sample_name is None:
            sample_name = extract_wxd_id(os.path.basename(file_paths[0]))
        if sample_name is None:
            sample_name = os.path.splitext(os.path.basename(file_paths[0]))[0]

        print(f"  样本 {sample_name} 包含 {len(file_paths)} 个txt")

    elif os.path.isdir(sample_path):

        if sample_name is None:
            sample_name = os.path.basename(sample_path)

        txt_files = sorted(f for f in os.listdir(sample_path) if f.lower().endswith(".txt"))

        if not txt_files:
            print(f"跳过空文件夹: {sample_path}")
            return None

        file_paths = [os.path.join(sample_path, f) for f in txt_files]

        print(f"  样本 {sample_name} 包含 {len(file_paths)} 个txt")

    else:

        if sample_name is None:
            sample_name = os.path.splitext(os.path.basename(sample_path))[0]

        file_paths = [sample_path]

        print(f"  样本 {sample_name} 包含1个txt")

    raman_shifts, counts_list = [], []

    for idx, file_path in enumerate(file_paths, 1):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

                data_lines = []
                for line in lines:
                    parts = re.split(r'[\s,;|]+', line)
                    if len(parts) >= 2 and all(re.match(r'^-?\d*\.?\d+$', p) for p in parts[:2]):
                        data_lines.append(parts[:2])

            if len(data_lines) < 2:
                continue

            df = pd.DataFrame(data_lines, columns=['Raman shift', 'Counts']).astype(float)
            raman_shifts.append(df['Raman shift'].values)
            counts_list.append(df['Counts'].values)

        except Exception as e:
            print(f"文件 {clean_filename(os.path.basename(file_path))} 读取失败: {str(e)[:100]}")
            continue

        if idx % 500 == 0 or idx == len(file_paths):
            print(f"    已处理 {idx}/{len(file_paths)} 个文件")

    if not counts_list:
        print(f"警告：样本 {sample_name} 无有效数据")
        return None

    try:
        all_shifts = np.concatenate(raman_shifts)
        shift_min = max(400, np.floor(np.min(all_shifts)))
        shift_max = min(2000, np.ceil(np.max(all_shifts)))
        shift_uniform = np.linspace(shift_min, shift_max, int(shift_max - shift_min + 1))
    except Exception as e:
        print(f"计算Raman shift范围失败: {e}")
        return None

    counts_interpolated = []
    for shift, counts in zip(raman_shifts, counts_list):
        try:
            interp_func = interp1d(shift, counts, kind='nearest', fill_value='extrapolate')
            counts_interpolated.append(interp_func(shift_uniform))
        except Exception:
            continue

    if not counts_interpolated:
        return None

    sample_mean_counts = np.mean(np.vstack(counts_interpolated), axis=0)
    global_data_collector.append(pd.DataFrame({
        'Raman shift': shift_uniform,
        'Counts': sample_mean_counts,
        'Source': sample_name
    }))

    return shift_uniform


def generate_global_output(global_data, output_root):
    """生成全局平均值文件和图表"""
    if not global_data:
        print("没有有效数据可处理")
        return

    all_data = pd.concat(global_data, ignore_index=True)

    global_avg = all_data.groupby('Raman shift')['Counts'].mean().reset_index()
    global_std = all_data.groupby('Raman shift')['Counts'].std().reset_index()

    print("样本均值计算样例 (前5个):")
    print(global_avg.head())
    print("平均值数据类型:", global_avg['Counts'].dtype)
    print("标准差数据类型:", global_std['Counts'].dtype)

    output_path = os.path.join(output_root, 'global_average.txt')
    np.savetxt(
        output_path,
        np.column_stack((global_avg['Raman shift'], global_avg['Counts'], global_std['Counts'])),
        fmt='%.6f',
        delimiter='\t',
        header='Raman shift\tAverage\tStd'
    )
    print(f"全局平均值已保存到: {output_path}")

    plt.figure(figsize=(12, 7))
    plt.plot(global_avg['Raman shift'], global_avg['Counts'],
             'b-', linewidth=2, label='Global Average')
    plt.fill_between(global_avg['Raman shift'],
                    global_avg['Counts'] - global_std['Counts'],
                    global_avg['Counts'] + global_std['Counts'],
                    color='blue', alpha=0.2, label='±1 Std')

    plt.xlabel('Raman Shift (cm$^{-1}$)', fontsize=14)
    plt.ylabel('Intensity (a.u.)', fontsize=14)
    plt.title('Global Average Raman Spectrum', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_root, 'global_average_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"全局平均图表已保存到: {plot_path}")

    data_path = os.path.join(output_root, 'all_spectra_data.csv')
    all_data.to_csv(data_path, index=False)
    print(f"所有样本均值数据已保存到: {data_path}")

    excel_path = os.path.join(output_root, 'all_sample_mean_spectra.xlsx')
    try:
        pivot_df = all_data.pivot(index='Source', columns='Raman shift', values='Counts')
        pivot_df = pivot_df.transpose()
        pivot_df.index.name = 'Raman shift'
        pivot_df.columns.name = 'Source'
        pivot_df.to_excel(excel_path)
        print(f"样本均值谱汇总已保存到: {excel_path}")
    except Exception as e:
        print(f"保存 Excel 失败: {e}")


def list_txt_files(folder_path, recursive=False):
    """列出文件夹中的 txt 文件；数字样本文件夹可递归收集。"""
    txt_files = []

    if recursive:
        for current_root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.lower().endswith(".txt"):
                    txt_files.append(os.path.join(current_root, filename))
    else:
        for filename in os.listdir(folder_path):
            full_path = os.path.join(folder_path, filename)
            if os.path.isfile(full_path) and filename.lower().endswith(".txt"):
                txt_files.append(full_path)

    return sorted(txt_files)


def build_sample_paths(root_folder, output_root):
    """
    递归识别样本：
    1. 纯数字命名的文件夹作为一个样本，收集该文件夹下所有 txt。
    2. 非数字文件夹中的直接 txt 按 wxdXXXX 分组；没有 wxd 时按文件名分组。
    """
    sample_groups = {}
    output_root_abs = os.path.abspath(output_root)

    for current_root, dirnames, filenames in os.walk(root_folder):
        current_root_abs = os.path.abspath(current_root)

        if current_root_abs == output_root_abs or current_root_abs.startswith(output_root_abs + os.sep):
            dirnames[:] = []
            continue

        dirnames[:] = sorted(
            d for d in dirnames
            if not os.path.abspath(os.path.join(current_root, d)).startswith(output_root_abs + os.sep)
        )

        folder_name = os.path.basename(current_root)

        if re.fullmatch(r"\d+", folder_name):
            txt_files = list_txt_files(current_root, recursive=True)
            if txt_files:
                sample_groups.setdefault(folder_name, []).extend(txt_files)

            # 数字文件夹已经作为一个样本处理，内部 txt 不再按 wxd 重复分组。
            dirnames[:] = []
            continue

        for filename in sorted(filenames):
            if not filename.lower().endswith(".txt"):
                continue

            full_path = os.path.join(current_root, filename)
            wxd_id = extract_wxd_id(filename)
            sample_name = wxd_id if wxd_id is not None else os.path.splitext(filename)[0]
            sample_groups.setdefault(sample_name, []).append(full_path)

    return [(name, sorted(paths)) for name, paths in sorted(sample_groups.items())]


def process_all_folders(root_folder, output_root):
    """主处理函数"""
    os.makedirs(output_root, exist_ok=True)

    global_data_collector = []

    sample_paths = build_sample_paths(root_folder, output_root)

    total_samples = len(sample_paths)
    print(f"找到 {total_samples} 个样本，开始处理...")

    for i, (name, sample_path) in enumerate(sample_paths, 1):
        print(f"处理样本 {i}/{total_samples}: {name}")
        process_sample(sample_path, root_folder, global_data_collector, sample_name=name)

    generate_global_output(global_data_collector, output_root)


if __name__ == "__main__":
    start_time = time.time()
    input_folder = r"D:\aaaSCNU\0data\Amyloid王惊华\process\PB-new\AL治疗前PB_pre"
    output_folder = os.path.join(input_folder, "全局平均结果")

    process_all_folders(input_folder, output_folder)

    end_time = time.time()
    print(f"\n总耗时: {end_time - start_time:.2f} 秒")
