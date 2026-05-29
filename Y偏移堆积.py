import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def create_raman_offset_plot():
    # 读取Excel数据 - 使用原始字符串
    excel_file = r'\aaaSCNU\0data\Raman\data.xlsx'  # 在字符串前加r
    
    try:
        # 读取数据
        df = pd.read_excel(excel_file, sheet_name='Sheet1')  # 添加sheet_name参数
        print(f"成功读取数据，数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {excel_file}")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 第一列是Raman Shift（X轴）
    raman_shift = df.iloc[:, 0]
    print(f"Raman Shift范围: {raman_shift.min():.1f} - {raman_shift.max():.1f}")
    
    # 计算数据组数（从第2列开始，每2列为一组：平均值、标准差）
    num_columns = df.shape[1] - 1  # 减去第一列
    num_groups = num_columns // 2
    
    print(f"共检测到 {num_groups} 组数据")
    
    # 存储所有组的平均值和标准差
    all_global_avg = []
    all_global_std = []
    group_names = ['bg', 'cell']
    
    # 遍历每组数据
    for i in range(num_groups):
        start_col = 1 + i * 2  # 每组数据的起始列索引
        
        # 提取当前组的数据
        group_avg = df.iloc[:, start_col]      # 平均值
        group_std = df.iloc[:, start_col + 1]  # 标准差
        
        # 将当前组的数据添加到列表中
        all_global_avg.append(group_avg)
        all_global_std.append(group_std)
        group_names.append(f'Group {i+1}')
        
        print(f"第{i+1}组 - 平均值范围: {group_avg.min():.2f} - {group_avg.max():.2f}")
        print(f"第{i+1}组 - 标准差范围: {group_std.min():.2f} - {group_std.max():.2f}")
    
    # 创建输出目录
    output_dir = 'output_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建Y偏移堆积图
    plt.figure(figsize=(16, 10))
    
    # 计算合适的偏移步长
    max_intensity = max([group_avg.max() for group_avg in all_global_avg])
    min_intensity = min([group_avg.min() for group_avg in all_global_avg])
    offset_step = (max_intensity - min_intensity) * 1.5
    
    # 设置Y偏移量
    y_offset = 0
    
    # 为每组数据绘制偏移后的曲线和误差带
    colors = plt.cm.tab10(np.linspace(0, 1, num_groups))
    
    for i in range(num_groups):
        group_avg = all_global_avg[i]
        group_std = all_global_std[i]
        
        # 绘制平均值曲线
        plt.plot(raman_shift, group_avg + y_offset, 
                color=colors[i], linewidth=2, alpha=0.9, label=f'{group_names[i]} Avg')
        
        # 绘制误差带 - 为每一组都添加误差带
        plt.fill_between(raman_shift,
                        group_avg + y_offset - group_std,
                        group_avg + y_offset + group_std,
                        color=colors[i], alpha=0.3, label=f'{group_names[i]} ±1 Std')
        
        y_offset += offset_step
    
    # 设置图表属性
    plot_title = 'Raman Spectra - Mac 532 1200'
    plt.xlabel('Raman Shift (cm$^{-1}$)', fontsize=20, fontweight='bold')
    plt.ylabel('Intensity (a.u.)', fontsize=20, fontweight='bold')
    plt.title(plot_title, fontsize=24, fontweight='bold')

    x_min = 275  # 最小Raman shift
    x_max = 2019 # 最大Raman shift
    plt.xlim(x_min, x_max)

    # 坐标轴刻度加粗
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')

    # 设置坐标轴区域背景透明
    ax = plt.gca()
    ax.set_facecolor('none')  # 坐标轴区域背景透明
    
    # 图例管理
    if num_groups <= 4:
        plt.legend(fontsize=20, loc='upper right', 
                  bbox_to_anchor=(1.15, 1.0),  # 向右移动
                  frameon=False,  # 去掉框线
                  facecolor='none',  # 背景透明
                  edgecolor='none')  # 边框颜色透明
    else:
        # 如果组数太多，简化图例（只显示平均值，不显示误差带图例）
        handles, labels = plt.gca().get_legend_handles_labels()
        # 只保留每个组的平均值图例
        unique_handles = []
        unique_labels = []
        seen_groups = set()
        for handle, label in zip(handles, labels):
            if 'Avg' in label and label.replace(' Avg', '') not in seen_groups:
                unique_handles.append(handle)
                unique_labels.append(label.replace(' Avg', ''))
                seen_groups.add(label.replace(' Avg', ''))
        plt.legend(unique_handles, unique_labels, fontsize=20, 
                  loc='upper right', 
                  bbox_to_anchor=(1.15, 1.0),  # 向右移动
                  frameon=False,  # 去掉框线
                  facecolor='none',  # 背景透明
                  edgecolor='none')  # 边框颜色透明
    
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    
     # 保存图表 - 使用标题作为文件名
    # 清理文件名中的特殊字符
    clean_title = "".join(c for c in plot_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    file_name = f"{clean_title}.png"
    plot_path = os.path.join(output_dir, file_name)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                transparent=True,  # 确保这个参数为True
                facecolor='none',   # 添加这个参数
                edgecolor='none')   # 添加这个参数
    plt.show()
    print(f"Y偏移堆积图已保存到: {plot_path}")

# 运行函数
if __name__ == "__main__":
    create_raman_offset_plot()