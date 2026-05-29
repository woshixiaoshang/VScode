#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
升级版 PCC 采样分析 (最终版)：
1. 自动读取 target.txt 寻找数据路径
2. 支持直接读取文件夹或单文件
3. 【新功能】分析结果图自动保存到源数据所在的文件夹
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import glob

# =====================
#  智能读取 target.txt
# =====================
def read_data_path_from_target(target_filename="target.txt"):
    candidates = [
        Path.cwd() / target_filename,
        Path.home() / "Desktop" / target_filename
    ]
    
    found_target_path = None
    for path in candidates:
        if path.exists():
            found_target_path = path
            print(f"[配置] 找到配置文件: {found_target_path}")
            break
    
    if found_target_path is None:
        msg = f"\n[错误] 未找到 '{target_filename}'。请将其放在脚本目录或桌面。"
        raise FileNotFoundError(msg)

    try:
        with open(found_target_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
    except Exception as e:
        raise ValueError(f"无法读取文件: {e}")

    if len(line) == 0:
        raise ValueError(f"配置文件为空！")

    clean_line = line.replace('"', '').replace("'", "")
    data_path = Path(clean_line)

    if not data_path.exists():
        raise FileNotFoundError(f"指向的数据路径不存在：\n{data_path}")

    return data_path


# ======================================
#  读光谱 (支持文件或文件夹)
# ======================================
def load_spectra(path_obj):
    path_obj = Path(path_obj)

    # --- 情况 A: 输入是一个文件夹 ---
    if path_obj.is_dir():
        print(f"[加载] 检测到输入是文件夹，正在搜索 .txt 文件...")
        files = sorted(list(path_obj.glob("*.txt")))
        
        if not files:
            raise ValueError("该文件夹下没有找到 .txt 文件！")
        
        total_files = len(files)
        print(f"[加载] 找到 {total_files} 个文件，正在合并数据...")

        try:
            first_data = np.loadtxt(files[0])
            if first_data.ndim == 1:
                p_points = first_data.shape[0]
                use_col = None 
            else:
                p_points = first_data.shape[0]
                use_col = 1 
        except Exception as e:
            raise ValueError(f"读取第一个文件失败: {files[0]}\n错误: {e}")

        X = np.zeros((total_files, p_points))

        for i, f in enumerate(files):
            try:
                if use_col is not None:
                    data = np.loadtxt(f, usecols=(use_col,))
                else:
                    data = np.loadtxt(f)
                
                if data.shape[0] != p_points:
                    continue
                
                X[i, :] = data
                
                if (i + 1) % 100 == 0:
                    print(f"  - 已读取 {i + 1}/{total_files}")
                    
            except Exception as e:
                print(f"[错误] 读取 {f.name} 失败: {e}")

        return X

    # --- 情况 B: 输入是一个单独的矩阵文件 ---
    else:
        print(f"[加载] 检测到输入是单个文件，正在读取...")
        try:
            X = np.loadtxt(path_obj)
            if X.ndim == 1:
                X = X[None, :]
            return X
        except Exception as e:
            raise ValueError(f"读取矩阵文件失败: {e}")


# ======================================
#  PCC 计算逻辑
# ======================================
def pearson_corr_1d(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
    if denom == 0: return np.nan
    return float(np.dot(a_c, b_c) / denom)

def corr_distribution_for_n(X, n, n_repeat=200):
    N, p = X.shape
    rng = np.random.default_rng()
    mu_full = X.mean(axis=0)
    r_vals = np.empty(n_repeat)

    for i in range(n_repeat):
        idx = rng.choice(N, size=n, replace=False)
        mu_sub = X[idx].mean(axis=0)
        r_vals[i] = pearson_corr_1d(mu_full, mu_sub)
    return r_vals

def corr_summary_over_ns(X, n_list, n_repeat=200):
    results = {}
    for n in n_list:
        r_vals = corr_distribution_for_n(X, n, n_repeat=n_repeat)
        results[n] = {
            "median": float(np.median(r_vals)),
            "p5": float(np.percentile(r_vals, 5)),
            "p95": float(np.percentile(r_vals, 95)),
        }
    return results


# ======================================
#  画图 & 保存
# ======================================
def plot_results(results, target_r=None, save_path=None):
    ns = sorted(results.keys())
    med = [results[n]["median"] for n in ns]
    p5 = [results[n]["p5"] for n in ns]
    p95 = [results[n]["p95"] for n in ns]

    plt.figure(figsize=(10, 7), dpi=150) # 稍微调大了尺寸和清晰度
    plt.plot(ns, med, marker="o", linewidth=2, label="Median PCC", color='#1f77b4')
    plt.fill_between(ns, p5, p95, alpha=0.2, color='#1f77b4', label="5%-95% Range (Reliability)")

    if target_r is not None:
        plt.axhline(target_r, linestyle="--", color='r', label=f"Target PCC = {target_r}")

    plt.xlabel("Number of spectra (n)", fontsize=12)
    plt.ylabel("PCC with Ground Truth (Full Average)", fontsize=12)
    plt.title("How many scans do you need?", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # 保存图片 (如果在 show 之前保存，防止空白)
    if save_path:
        try:
            plt.savefig(save_path)
            print(f"\n[保存] ✅ 结果图已保存至:\n      {save_path}")
        except Exception as e:
            print(f"\n[警告] 保存图片失败: {e}")

    plt.show()


# ======================================
#  主程序
# ======================================
def main():
    try:
        # 1. 获取数据路径
        data_path = read_data_path_from_target("target.txt")
        print(f"[配置] 数据路径: {data_path}")

        # 2. 确定结果图的保存位置
        # 如果是文件夹，就存在该文件夹内
        # 如果是文件，就存在该文件所在的目录
        if data_path.is_dir():
            save_dir = data_path
        else:
            save_dir = data_path.parent
        
        # 定义输出文件名
        output_image_path = save_dir / "PCC_Analysis_Result.svg"


        # 3. 加载数据
        X = load_spectra(data_path)
        
        # 简单清洗全0行
        if np.all(X[-1] == 0): pass

        N, p = X.shape
        print(f"[成功] 矩阵构建完成: 共 {N} 条光谱")

       # ==========================================
        # 修改开始：自定义 n 的取值列表
        # ==========================================
        
        # 1. 预设一些常用的采样点
        potential_ns = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        
        # 2. 筛选出比 N 小的数 (比如 39 以内的)
        n_list = [n for n in potential_ns if n < N]

        # 3. 【关键】强制把最大值 N (比如 39) 加进去
        # 这样你就能看到 n=39 时的结果（理论上 PCC 应该是 1.0）
        n_list.append(N)
        
        # 4. 去重并排序 (防止 N 恰好等于列表里的数导致重复)
        n_list = sorted(list(set(n_list)))
        
        if not n_list:
            print("[错误] 数据量太少，无法进行抽样分析。")
            return

        print(f">>> 开始计算 (重复抽样 200 次)...")
        results = corr_summary_over_ns(X, n_list, n_repeat=200)

        # 5. 打印结果
        print("\n===== 结果 (PCC) =====")
        print("n\tMedian\tP5\tP95")
        for n in n_list:
            r = results[n]
            print(f"{n}\t{r['median']:.4f}\t{r['p5']:.4f}\t{r['p95']:.4f}")

        # 6. 绘图并自动保存
        print("\n>>> 正在绘图并保存...")
        plot_results(results, target_r=0.99, save_path=output_image_path)

    except Exception as e:
        print(f"\n[程序崩溃] 错误信息: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")

if __name__ == "__main__":
    main()