# -*- coding: utf-8 -*-
"""
基于拉曼光谱批量成像（2900-3000 cm^-1）：
- 绿色热力图模式 (Raw Intensity Map)
- 修正坐标镜像问题 (通过 FLIP_Y / FLIP_X 控制)
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# 1. 基本参数配置
# ==========================

# 存放所有 txt 光谱文件的文件夹 (自动从 target.txt 读取)
DATA_DIR = None

# 输出图片所在文件夹
OUTPUT_DIR = "output"

# 拉曼峰区间（单位：cm^-1）
X_MIN = 2900.0
X_MAX = 2980.0

# ==========================
# [关键修改] 镜像控制开关
# ==========================
# 如果发现图是倒着的（上下颠倒），设为 True
FLIP_Y = False  

# 如果发现图是左右反的（左右镜像），设为 True
FLIP_X = False 


# 文件名中提取 X/Y 的正则表达式
FNAME_XY_PATTERN = re.compile(r"__X_([-\d\.]+)__Y_([-\d\.]+)__")


# ==========================
# 2. 工具函数
# ==========================

def read_data_dir_from_target(target_file="target.txt"):
    """从 target.txt 读取数据路径"""
    cwd_path = os.path.join(os.getcwd(), target_file)
    desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
    desktop_path = os.path.join(desktop_dir, target_file)

    for path in [cwd_path, desktop_path]:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                dir_path = f.readline().strip()
            if not os.path.isdir(dir_path):
                raise NotADirectoryError(f"路径无效：{dir_path}")
            return dir_path
    raise FileNotFoundError(f"未找到 {target_file}")

def find_all_txt_files(data_dir):
    """递归查找 txt 文件"""
    txt_files = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.lower().endswith(".txt"):
                txt_files.append(os.path.join(root, name))
    txt_files.sort()
    return txt_files

def parse_xy_from_filename(filepath):
    """解析文件名坐标"""
    filename = os.path.basename(filepath)
    m = FNAME_XY_PATTERN.search(filename)
    if not m:
        return None, None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None, None

def load_spectrum(txt_path):
    """读取光谱数据"""
    try:
        data = np.loadtxt(txt_path)
    except Exception:
        return None, None
    if data.ndim != 2 or data.shape[1] < 2:
        return None, None
    x = data[:, 0]
    y = data[:, 1]
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y

def extract_intensity_in_range(x, y, x_min, x_max):
    """计算特征强度"""
    mask = (x >= x_min) & (x <= x_max)
    if not np.any(mask):
        return 0.0
    return float(y[mask].max())

def build_intensity_map(file_paths):
    """构建 2D 强度矩阵"""
    coords = []
    values = []
    
    total = len(file_paths)
    print(f"[信息] 开始处理 {total} 个文件...")
    
    for i, path in enumerate(file_paths, start=1):
        x_coord, y_coord = parse_xy_from_filename(path)
        if x_coord is None: continue

        x, y = load_spectrum(path)
        if x is None: continue

        val = extract_intensity_in_range(x, y, X_MIN, X_MAX)
        coords.append((x_coord, y_coord))
        values.append(val)

        if i % 500 == 0:
            print(f"  - 进度: {i}/{total}")

    if not coords:
        raise RuntimeError("无有效数据")

    coords = np.array(coords)
    values = np.array(values)

    xs = coords[:, 0]
    ys = coords[:, 1]
    
    xs_sorted = np.sort(np.unique(xs))
    ys_sorted = np.sort(np.unique(ys))
    
    nx = len(xs_sorted)
    ny = len(ys_sorted)
    
    x2i = {v: i for i, v in enumerate(xs_sorted)}
    y2i = {v: i for i, v in enumerate(ys_sorted)}

    intensity_map = np.zeros((ny, nx), dtype=np.float32)

    for (xc, yc), val in zip(coords, values):
        c = x2i[xc]
        r = y2i[yc]
        intensity_map[r, c] = val

    return intensity_map, xs_sorted, ys_sorted


# ==========================
# 3. 主流程
# ==========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    global DATA_DIR
    if DATA_DIR is None:
        try:
            DATA_DIR = read_data_dir_from_target("target.txt")
        except Exception as e:
            print(f"[错误] {e}")
            return

    file_paths = find_all_txt_files(DATA_DIR)
    if not file_paths: return

    # 1. 构建原始矩阵
    intensity_map, xs, ys = build_intensity_map(file_paths)
    print(f"[信息] 矩阵尺寸: {intensity_map.shape}")

    # 2. 处理镜像/翻转 (关键步骤)
    # 也就是处理你觉得“位置反了”的问题
    if FLIP_Y:
        print("[操作] 执行垂直翻转 (Flip Y)")
        intensity_map = np.flipud(intensity_map)
    
    if FLIP_X:
        print("[操作] 执行水平翻转 (Flip X)")
        intensity_map = np.fliplr(intensity_map)

    # 3. 自动对比度增强
    vmin = np.percentile(intensity_map, 5)
    vmax = np.percentile(intensity_map, 90)
    img_clipped = np.clip(intensity_map, vmin, vmax)

    # 4. 归一化 (0.0 - 1.0)
    img_norm = (img_clipped - vmin) / (vmax - vmin + 1e-6)
    img_norm = np.clip(img_norm, 0.0, 1.0)
    
    # 5. 生成绿色热力图 RGBA
    ny, nx = img_norm.shape
    rgba_img = np.zeros((ny, nx, 4), dtype=np.float32)
    rgba_img[..., 1] = img_norm  # Green Channel
    rgba_img[..., 3] = 1.0       # Alpha Channel

    # 6. 保存图片
    # 注意：这里去掉了 origin='lower'，改用默认的 'upper' (图像坐标系)
    # 配合前面的 np.flipud，你应该能得到和软件一致的视角
    save_path = os.path.join(OUTPUT_DIR, "raman_green_fixed.png")
    plt.imsave(save_path, rgba_img)
    print(f"[成功] 图片已保存: {save_path}")

    # 7. 保存带坐标轴参考图
    plot_path = os.path.join(OUTPUT_DIR, "raman_plot_axis_fixed.png")
    plt.figure(figsize=(10, 8), dpi=150)
    
    # 根据翻转情况调整坐标轴显示
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    if FLIP_Y:
        # 如果翻转了数据，画图时也要让 Y 轴刻度倒过来对应
        extent = [xs.min(), xs.max(), ys.max(), ys.min()]

    plt.imshow(img_norm, cmap='Greens', interpolation='nearest', aspect='auto', extent=extent)
    plt.colorbar(label='Normalized Intensity')
    plt.title(f"Raman Map ({X_MIN}-{X_MAX} cm$^-$$^1$)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.savefig(plot_path)
    plt.close()
    print(f"[成功] 参考图已保存: {plot_path}")

if __name__ == "__main__":
    main()