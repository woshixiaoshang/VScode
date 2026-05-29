import os
import pandas as pd
from pathlib import Path

def read_data_dir_from_target():
    """
    从桌面的 target.txt 中读取路径
    target.txt 内只需要写一行：数据文件夹的路径
    """
    desktop = Path.home() / "Desktop"
    target_file = desktop / "target.txt"

    if not target_file.exists():
        raise FileNotFoundError(f"未找到 target 文件：{target_file}")

    with open(target_file, "r", encoding="utf-8") as f:
        path_str = f.readline().strip()

    data_dir = Path(path_str)
    if not data_dir.exists():
        raise FileNotFoundError(f"target 文件中的路径不存在：{data_dir}")

    return data_dir


def merge_txt_to_excel_and_csv():
    # 读取路径
    folder_path = read_data_dir_from_target()
    print("读取到的数据文件夹路径：", folder_path)

    # 找所有 txt 文件
    txt_files = sorted([f for f in folder_path.glob("*.txt")])

    if not txt_files:
        raise ValueError("该文件夹中没有 .txt 文件！")

    merged_df = pd.DataFrame()

    for txt in txt_files:
        # 读取两列：shift + intensity
        df = pd.read_csv(txt, sep=None, engine="python", header=None, names=["shift", txt.stem])

        # 第一次：直接放进去
        if merged_df.empty:
            merged_df = df.copy()
        else:
            # 第二次以后：按 shift 合并
            merged_df = pd.merge(merged_df, df, on="shift", how="outer")

    # 按 shift 排序（更规范）
    merged_df = merged_df.sort_values("shift").reset_index(drop=True)

    # 输出 CSV
    out_csv = folder_path / "merged_all_txt.csv"
    merged_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 输出 Excel
    out_excel = folder_path / "merged_all_txt.xlsx"
    merged_df.to_excel(out_excel, index=False)

    print("合并完成！")
    print("CSV 输出：", out_csv)
    print("Excel 输出：", out_excel)


if __name__ == "__main__":
    merge_txt_to_excel_and_csv()
