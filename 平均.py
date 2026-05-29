import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import re
import time  # 添加时间模块

def clean_filename(filename):
    """清理文件名中的特殊字符和坐标信息"""
    cleaned = re.sub(r'__X[^_]+__Y[^_]+__ElapsedTime_\d+', '', filename)
    cleaned = re.sub(r'[^\w.-]', '', cleaned)
    return cleaned.split('.txt')[0] + '.txt' if cleaned.endswith('.txt') else cleaned

def process_subfolder(subfolder_path, global_data_collector):
    """处理单个子文件夹并收集数据"""
    folder_name = os.path.basename(subfolder_path)
    all_files = [f for f in os.listdir(subfolder_path) if f.endswith('.txt')]
    
    if not all_files:
        print(f"跳过空文件夹: {subfolder_path}")
        return None

    total_files = len(all_files)
    print(f"  文件夹 {folder_name} 包含 {total_files} 个.txt文件")
    
    raman_shifts, counts_list = [], []
    
    for idx, file in enumerate(all_files, 1):
        file_path = os.path.join(subfolder_path, file)
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
            print(f"文件 {clean_filename(file)} 读取失败: {str(e)[:100]}")
            continue
        
        # 每处理500个文件或最后一个文件时打印进度
        if idx % 500 == 0 or idx == total_files:
            print(f"    已处理 {idx}/{total_files} 个文件")

    if not counts_list:
        print(f"警告：{folder_name} 无有效数据")
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
        except:
            continue

    if not counts_interpolated:
        return None

    # 将数据添加到全局收集器
    for shift, counts in zip([shift_uniform]*len(counts_interpolated), counts_interpolated):
        global_data_collector.append(pd.DataFrame({
            'Raman shift': shift,
            'Counts': counts,
            'Source': folder_name
        }))

    return shift_uniform

def generate_global_output(global_data, output_root):
    """生成全局平均值文件和图表"""
    if not global_data:
        print("没有有效数据可处理")
        return
    
    # 合并所有数据
    all_data = pd.concat(global_data, ignore_index=True)
    
    # 计算全局平均值和标准差
    global_avg = all_data.groupby('Raman shift')['Counts'].mean().reset_index()
    global_std = all_data.groupby('Raman shift')['Counts'].std().reset_index()
    
    print("平均值计算样例 (前5个):")
    print(global_avg.head())
    print("平均值数据类型:", global_avg['Counts'].dtype)
    print("标准差数据类型:", global_std['Counts'].dtype)
    
    # 保存全局平均值文件
    output_path = os.path.join(output_root, 'global_average.txt')
    np.savetxt(
        output_path,
        np.column_stack((global_avg['Raman shift'], global_avg['Counts'], global_std['Counts'])),
        fmt='%.6f',  # 改为6位小数以保留更多精度
        delimiter='\t',
        header='Raman shift\tAverage\tStd'
    )
    print(f"全局平均值已保存到: {output_path}")

    # 绘制全局平均图表
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

    # 保存图表
    plot_path = os.path.join(output_root, 'global_average_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"全局平均图表已保存到: {plot_path}")

    # 保存原始数据供后续分析
    data_path = os.path.join(output_root, 'all_spectra_data.csv')
    all_data.to_csv(data_path, index=False)
    print(f"所有光谱数据已保存到: {data_path}")

def process_all_folders(root_folder, output_root):
    """主处理函数"""
    os.makedirs(output_root, exist_ok=True)
    
    # 全局数据收集器 (存储所有数据点)
    global_data_collector = []
    
    # 收集所有包含.txt文件的子文件夹
    subfolders = []
    for foldername, _, filenames in os.walk(root_folder):
        if any(f.endswith('.txt') for f in filenames):
            subfolders.append(foldername)
    
    total_folders = len(subfolders)
    print(f"找到 {total_folders} 个包含.txt文件的文件夹，开始处理...")
    
    # 处理所有子文件夹
    for i, foldername in enumerate(subfolders, 1):
        print(f"处理文件夹 {i}/{total_folders}: {os.path.basename(foldername)}")
        process_subfolder(foldername, global_data_collector)
    
    # 生成全局输出
    generate_global_output(global_data_collector, output_root)

if __name__ == "__main__":
    start_time = time.time()  # 开始计时
    input_folder = r"D:\aaaSCNU\0data\Amyloid王惊华\process\BM\AL-BM治疗前"
    output_folder = os.path.join(input_folder, "全局平均结果")
    
    process_all_folders(input_folder, output_folder)
    
    end_time = time.time()  # 结束计时
    print(f"\n总耗时: {end_time - start_time:.2f} 秒")