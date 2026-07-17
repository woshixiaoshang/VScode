import subprocess
import sys
import os

# ============================================================
# 配置区：脚本路径（只需设置一次）
# ============================================================
OUTLIER_SCRIPT    = r"D:\APP\Vscode\Vscode-test\拉曼数据预处理\异常值.py"      # ← 改成你的实际路径
PREPROCESS_SCRIPT = r"D:\APP\Vscode\Vscode-test\拉曼数据预处理\Raman预处理.py" # ← 改成你的实际路径

# target.txt 查找逻辑（和两个脚本保持一致，自动找桌面或脚本目录）
def find_target_file():
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    candidate_names = ["target", "target.txt"]
    for base_dir in [desktop_path, current_dir]:
        for name in candidate_names:
            candidate = os.path.join(base_dir, name)
            if os.path.isfile(candidate):
                return candidate
    # 模糊匹配 target* 文件
    for base_dir in [desktop_path, current_dir]:
        if os.path.isdir(base_dir):
            for fname in os.listdir(base_dir):
                if fname.lower().startswith("target"):
                    candidate = os.path.join(base_dir, fname)
                    if os.path.isfile(candidate):
                        return candidate
    return None

# ============================================================
# 从 target.txt 读取路径
# ============================================================
def read_target():
    target_file = find_target_file()
    if target_file is None:
        print("❌ 未找到 target 文件，请在桌面或脚本目录下创建 target.txt")
        print("   格式：")
        print("   异常值")
        print("   /原始数据路径")
        print("   预处理")
        print("   /预处理结果保存路径")
        sys.exit(1)

    print(f"🔎 读取 target 文件：{target_file}")
    with open(target_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    raw_dir    = None
    result_dir = None
    for i, line in enumerate(lines):
        if line == "异常值" and i + 1 < len(lines):
            raw_dir = lines[i + 1]
        if line == "预处理" and i + 1 < len(lines):
            result_dir = lines[i + 1]

    if raw_dir is None:
        print('❌ target 文件中未找到"异常值"标签及其路径')
        sys.exit(1)
    if result_dir is None:
        print('❌ target 文件中未找到"预处理"标签及其路径')
        sys.exit(1)

    return target_file, raw_dir, result_dir

# ============================================================
# 写入 target.txt
# ============================================================
def write_target(target_file, lines):
    with open(target_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def update_target_paths(target_file, raw_dir=None, result_dir=None, preprocess_input=None):
    """更新 target.txt 中的路径标签，保留其他内容。"""
    with open(target_file, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]

    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == "异常值" and raw_dir is not None and i + 1 < len(lines):
            out_lines.append(line)
            out_lines.append(raw_dir)
            i += 2
            continue
        if stripped == "预处理" and result_dir is not None and i + 1 < len(lines):
            out_lines.append(line)
            out_lines.append(result_dir)
            i += 2
            continue
        if stripped == "预处理输入" and preprocess_input is not None and i + 1 < len(lines):
            out_lines.append(line)
            out_lines.append(preprocess_input)
            i += 2
            continue
        out_lines.append(line)
        i += 1

    # 如果原文件中缺少标签，则追加
    if raw_dir is not None and not any(line.strip() == "异常值" for line in lines):
        out_lines.extend(["异常值", raw_dir])
    if result_dir is not None and not any(line.strip() == "预处理" for line in lines):
        out_lines.extend(["预处理", result_dir])
    if preprocess_input is not None and not any(line.strip() == "预处理输入" for line in lines):
        out_lines.extend(["预处理输入", preprocess_input])

    with open(target_file, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

# ============================================================
# 执行单个步骤
# ============================================================
def run_step(name, script):
    print(f"\n{'='*55}")
    print(f"▶  {name}")
    print(f"{'='*55}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n❌ {name} 执行失败，流水线中止")
        sys.exit(1)
    print(f"\n✅ {name} 完成")


def build_processed_save_folder(result_dir, raw_dir):
    raw_dir_norm = raw_dir.rstrip('/\\')
    raw_parent_dir = os.path.dirname(raw_dir_norm)
    raw_second_last = os.path.basename(raw_parent_dir)
    raw_last = os.path.basename(raw_dir_norm)
    if raw_second_last:
        return os.path.join(result_dir, f"{raw_second_last}_processed", raw_last)
    return result_dir


def build_preprocess_input_folder(raw_dir):
    raw_dir_norm = raw_dir.rstrip('/\\')
    parent_dir = os.path.dirname(raw_dir_norm)
    folder_name = os.path.basename(raw_dir_norm)
    return os.path.join(parent_dir, folder_name + "_cleaned", "kept_spectra")

# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("🚀  Raman 数据处理流水线")
    print("=" * 55)

    # 读取 target.txt 里的两个路径
    target_file, raw_dir, result_dir = read_target()

    # 异常值.py 的输出路径（和异常值.py内部逻辑保持一致）
    parent_dir = os.path.dirname(raw_dir.rstrip("/\\"))
    folder_name = os.path.basename(raw_dir.rstrip("/\\"))
    cleaned_dir = os.path.join(parent_dir, folder_name + "_cleaned", "kept_spectra")

    print(f"\n   原始数据路径：{raw_dir}")
    print(f"   筛选输出路径：{cleaned_dir}  ← 自动推算")
    print(f"   读取 target.txt 中 预处理 路径：{result_dir}")
    output_save_folder = build_processed_save_folder(result_dir, raw_dir)
    print(f"   实际最终保存目录：{output_save_folder}  ← 自动生成")

    # ── Step 1：异常值筛选 ──────────────────────────────────
    # target写入：异常值 + 原始数据路径
    update_target_paths(target_file, raw_dir=raw_dir)
    run_step("Step 1：异常值筛选", OUTLIER_SCRIPT)

    # Step1 完成后，确保 target.txt 中包含预处理读取路径
    preprocess_input = build_preprocess_input_folder(raw_dir)
    update_target_paths(target_file, preprocess_input=preprocess_input)
    print(f"   预处理读取路径已写入 target.txt: {preprocess_input}")

    # ── Step 2：预处理 ──────────────────────────────────────
    # target.txt 中“预处理”标签只保持最终结果保存路径
    # 读取数据路径来自预处理输入路径（预处理输入）或由异常值原始路径派生
    update_target_paths(target_file, result_dir=result_dir)
    run_step("Step 2：预处理（基线校正 + 去噪 + 归一化）", PREPROCESS_SCRIPT)

    # ── 流水线结束，恢复 target.txt 为原始格式 ────────────────
    # 只修改异常值和预处理标签后的路径，保留其他标签及内容不变
    update_target_paths(target_file, raw_dir=raw_dir, result_dir=result_dir)
    print(f"\n{'='*55}")
    print(f"🎉  全部完成！")
    print(f"    最终结果保存在：{output_save_folder}")
    print(f"    target.txt 已恢复为原始格式，下次直接改路径重跑即可")
    print(f"{'='*55}")
