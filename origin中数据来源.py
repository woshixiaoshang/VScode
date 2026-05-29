# %%
import pandas as pd
import os
import shutil
import re
import sys
from collections import defaultdict

# --- 1. 请用户配置这里的参数 ---

# (调试开关：保持 False 即可，新的诊断报告会自动运行)
DEBUG_MODE = False

# (新!) 相似度阈值：
# 当一个 Excel 列 100% 匹配失败时，
# 脚本会寻找“几乎匹配”的 TXT 文件。
# 0.9 = 寻找相似度 90% 以上的文件。
# 0.8 = 寻找相似度 80% 以上的文件。
# 建议保持 0.8 或 0.9
SIMILARITY_THRESHOLD = 0.8

# === 按照您的要求更新了以下路径 ===
# Excel 文件完整路径
EXCEL_FILE_PATH = r"D:\aaaSCNU\邵勇\邵勇-毕业生数据1\5.毕业论文相关的原始数据\第三章 基于金-碳点双模SERS探针的巨噬细胞泡沫化代谢谱\3.代谢谱的可行性\data.xlsx"
SHEET_NAME = "Sheet1" 

# 查找源文件夹
SOURCE_DIR = r"D:\aaaSCNU\邵勇\111\111"

# 结果保存文件夹
TARGET_DIR = r"D:\aaaSCNU\邵勇\邵勇-毕业生数据1\5.毕业论文相关的原始数据\第三章 基于金-碳点双模SERS探针的巨噬细胞泡沫化代谢谱\3.代谢谱的可行性\new"

# Txt 文件内部分隔符 (例如: '\t' for Tab, ',' for 逗号)
TXT_DELIMITER = '\t' 

# === 按照您的要求，匹配“第二列” ===
# Txt 文件中，我们关心的数据在第几列？ (0=第一列, 1=第二列)
TXT_TARGET_COL_INDEX = 1

# --- 2. 脚本逻辑 (已更新) ---

# 预编译正则表达式，用于从 "abc 123.45" 中提取 "123.45"
NUMBER_EXTRACTOR = re.compile(r"(-?\d+\.?\d*)$")

def create_new_filename(full_path, source_root):
    """根据原始相对路径创建一个安全的文件名。"""
    try:
        relative_path = os.path.relpath(full_path, source_root)
        new_name = relative_path.replace(os.sep, "_")
        new_name = re.sub(r'[<>:"/\\|?*]', '_', new_name)
        return new_name
    except ValueError:
        return os.path.basename(full_path)

def normalize_data(item):
    """(已更新) 统一数据格式"""
    if pd.isna(item):
        return None
    clean_item = str(item).strip().lower()
    match = NUMBER_EXTRACTOR.search(clean_item)
    if match:
        clean_item = match.group(1)
    try:
        f_val = float(clean_item)
        if f_val.is_integer():
            clean_item = str(int(f_val))
    except ValueError:
        pass 
    if clean_item:
        return clean_item
    return None

def get_data_fingerprint(data_iterable):
    """为一组数据创建“指纹”"""
    cleaned_data = set()
    for item in data_iterable:
        normalized_item = normalize_data(item)
        if normalized_item:
            cleaned_data.add(normalized_item)
    return frozenset(cleaned_data)

def read_txt_fingerprint(filepath, delimiter, col_index):
    """读取单个TXT文件特定列的指纹"""
    txt_data = set()
    for enc in ['utf-8', 'gbk', 'utf-16', sys.getdefaultencoding()]:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                for line in f:
                    parts = line.strip().split(delimiter)
                    if len(parts) > col_index:
                        normalized_value = normalize_data(parts[col_index])
                        if normalized_value:
                            txt_data.add(normalized_value)
            return get_data_fingerprint(txt_data)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if DEBUG_MODE:
                print(f"  !! 读取 {filepath} 时出错 (编码 {enc}): {e}")
            break
    return None

def jaccard_similarity(fp1, fp2):
    """(新!) 计算两个指纹（set）的相似度"""
    if not fp1 and not fp2:
        return 1.0
    if not fp1 or not fp2:
        return 0.0
    intersection = len(fp1.intersection(fp2))
    union = len(fp1.union(fp2))
    return intersection / union

def main():
    # 1. 检查路径
    if not os.path.exists(EXCEL_FILE_PATH) or not os.path.exists(SOURCE_DIR):
        print("错误: Excel文件或源文件夹路径不存在，请检查配置。")
        return
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    # --- 步骤 A: 读取 Excel，创建所有列的指纹 ---
    print(f"正在读取 Excel: {EXCEL_FILE_PATH} ...")
    
    # excel_fingerprints 存储: {指纹: [列名]}
    excel_fingerprints = defaultdict(list)
    # (新!) excel_fingerprint_lookup 存储: {列名: 指纹}
    excel_fingerprint_lookup = {}
    
    try:
        # 假设 Excel 文件没有表头，数据从第一行开始
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, header=None)
        
        for col_index in df.columns:
            col_data = df[col_index]
            fingerprint = get_data_fingerprint(col_data)
            
            if fingerprint:
                col_name_str = f"Col_{col_index + 1}" # (例如 Col_1, Col_2)
                excel_fingerprints[fingerprint].append(col_name_str)
                excel_fingerprint_lookup[col_name_str] = fingerprint
                
        print(f"Excel 读取完毕。共找到 {len(df.columns)} 列，生成 {len(excel_fingerprints)} 个唯一的“数据指纹”。")
        if DEBUG_MODE:
            print("\n--- [调试] Excel 指纹库 ---")
            for fp, cols in excel_fingerprints.items():
                print(f"  列: {cols} (共 {len(fp)} 项)")
            print("--------------------------\n")

    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    if not excel_fingerprints:
        print("Excel 中没有提取到有效的数据指纹，程序退出。")
        return

    # --- 步骤 B: 遍历 TXT 文件，匹配指纹 ---
    print(f"开始遍历文件夹: {SOURCE_DIR} ...")
    found_files_count = 0
    scanned_files_count = 0
    
    # (新!) 存储所有 TXT 指纹用于后续诊断
    all_txt_fingerprints = {} # {filepath: fingerprint}
    
    # (新!) 存储已匹配的 Excel 指纹
    matched_excel_fingerprints = set()

    for root, dirs, files in os.walk(SOURCE_DIR):
        for filename in files:
            if not filename.lower().endswith(".txt"):
                continue
            
            scanned_files_count += 1
            if scanned_files_count % 100 == 0:
                print(f"  已扫描 {scanned_files_count} 个 .txt 文件...", end='\r')

            filepath = os.path.join(root, filename)
            txt_fingerprint = read_txt_fingerprint(filepath, TXT_DELIMITER, TXT_TARGET_COL_INDEX)

            if not txt_fingerprint: # 跳过空文件
                continue
            
            # (新!) 存储所有 TXT 指纹
            all_txt_fingerprints[filepath] = txt_fingerprint

            if DEBUG_MODE and txt_fingerprint:
                print(f"\n[调试] 扫描文件: {filename} (共 {len(txt_fingerprint)} 项)")

            # === 核心比对 ===
            if txt_fingerprint in excel_fingerprints:
                
                matched_excel_cols = excel_fingerprints[txt_fingerprint]
                matched_cols_str = "_".join(matched_excel_cols)
                safe_cols_str = re.sub(r'[<>:"/\\|?*]', '_', matched_cols_str)
                
                print(f"\n  [匹配成功] 文件: {filename}")
                print(f"    -> 匹配 Excel 列: {safe_cols_str}")

                path_based_name = create_new_filename(filepath, SOURCE_DIR)
                new_filename = f"[匹配_{safe_cols_str}]__[来自_{path_based_name}]"
                dest_path = os.path.join(TARGET_DIR, new_filename)
                
                try:
                    shutil.copy2(filepath, dest_path)
                    found_files_count += 1
                    # (新!) 标记这个 Excel 指纹已被匹配
                    matched_excel_fingerprints.add(txt_fingerprint)
                except Exception as e:
                    print(f"  !! 复制文件失败: {e}")

    print(f"\n--- 任务完成 ---")
    print(f"共扫描 {scanned_files_count} 个 .txt 文件。")
    print(f"共找到并复制了 {found_files_count} 个“整列匹配”的文件到 {TARGET_DIR}")

    # --- (新!) 步骤 C: 诊断未匹配的列 ---
    
    # 找出所有 Excel 指纹中，未被匹配的那些
    unmatched_excel_fingerprints = set(excel_fingerprints.keys()) - matched_excel_fingerprints
    
    if unmatched_excel_fingerprints:
        print("\n--- 诊断报告：未找到完美匹配的 Excel 列 ---")
        
        # 遍历每一个“未匹配”的 Excel 指纹
        for unmatched_fp in unmatched_excel_fingerprints:
            excel_col_names = excel_fingerprints[unmatched_fp]
            print(f"\n[分析] Excel 列: {', '.join(excel_col_names)} (共 {len(unmatched_fp)} 项)")
            
            best_match_file = None
            best_match_similarity = 0.0
            best_match_fp = None

            # 将这个“未匹配”的指纹与“所有”TXT 指纹进行比较
            for filepath, txt_fp in all_txt_fingerprints.items():
                similarity = jaccard_similarity(unmatched_fp, txt_fp)
                
                if similarity > best_match_similarity:
                    best_match_similarity = similarity
                    best_match_file = filepath
                    best_match_fp = txt_fp
            
            # 检查最佳匹配是否达到了我们设定的阈值
            if best_match_similarity >= SIMILARITY_THRESHOLD:
                print(f"  -> [最接近的] 文件: {os.path.basename(best_match_file)}")
                print(f"  -> [相似度]: {best_match_similarity * 100:.2f}%")
                print(f"  -> [TXT 文件项数]: {len(best_match_fp)}")
                
                # 找出具体差异
                in_excel_not_in_txt = unmatched_fp - best_match_fp
                in_txt_not_in_excel = best_match_fp - unmatched_fp
                
                if in_excel_not_in_txt:
                    print(f"  -> [差异] Excel 中独有 (前5项): {list(in_excel_not_in_txt)[:5]}")
                if in_txt_not_in_excel:
                    print(f"  -> [差异] TXT 中独有 (前5项): {list(in_txt_not_in_excel)[:5]}")
            
            else:
                print(f"  -> [未找到相似文件] (最佳相似度 {best_match_similarity * 100:.2f}%，低于 {SIMILARITY_THRESHOLD * 100}%)")
    
    else:
        print("\n--- 诊断报告：所有 Excel 列均已找到完美匹配！ ---")


if __name__ == "__main__":
    main()


