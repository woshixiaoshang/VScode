import os
import re

def batch_rename_txt_files(directory, pattern=None, replacement=None, prefix=None, suffix=None):
    """
    批量重命名txt文件
    
    参数:
    directory: 文件所在目录
    pattern: 要匹配的字符串模式（正则表达式）
    replacement: 替换成的字符串
    prefix: 添加前缀
    suffix: 添加后缀
    """
    
    # 获取目录下所有txt文件
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    if not txt_files:
        print("该目录下没有找到txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个txt文件")
    
    renamed_count = 0
    for old_name in txt_files:
        new_name = old_name
        
        # 根据参数修改文件名
        if pattern and replacement:
            # 使用正则表达式替换
            new_name = re.sub(pattern, replacement, new_name)
        
        if prefix:
            # 添加前缀
            new_name = prefix + new_name
        
        if suffix:
            # 添加后缀（在.txt之前添加）
            base_name, ext = os.path.splitext(new_name)
            new_name = base_name + suffix + ext
        
        # 如果文件名有变化，进行重命名
        if new_name != old_name:
            old_path = os.path.join(directory, old_name)
            new_path = os.path.join(directory, new_name)
            
            # 处理重名情况
            counter = 1
            while os.path.exists(new_path):
                name, ext = os.path.splitext(new_name)
                new_name = f"{name}_{counter}{ext}"
                new_path = os.path.join(directory, new_name)
                counter += 1
            
            os.rename(old_path, new_path)
            print(f"重命名: {old_name} -> {new_name}")
            renamed_count += 1
        else:
            print(f"跳过: {old_name} (无需修改)")
    
    print(f"\n完成！共重命名 {renamed_count} 个文件")

def sequential_rename(directory, base_name="file", start_num=1):
    """
    顺序重命名txt文件（如 file1.txt, file2.txt...）
    
    参数:
    directory: 文件所在目录
    base_name: 基础文件名
    start_num: 起始编号
    """
    
    txt_files = sorted([f for f in os.listdir(directory) if f.endswith('.txt')])
    
    if not txt_files:
        print("该目录下没有找到txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个txt文件")
    
    for i, old_name in enumerate(txt_files, start=start_num):
        new_name = f"{base_name}{i}.txt"
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)
        
        os.rename(old_path, new_path)
        print(f"重命名: {old_name} -> {new_name}")
    
    print(f"完成！共重命名 {len(txt_files)} 个文件")

# 使用示例
if __name__ == "__main__":
    # 设置你的txt文件所在目录
    target_directory = r"D:\Test-origin"  # 修改为你的实际路径
    
    # 示例1：删除文件名中的特定字符串
    # batch_rename_txt_files(target_directory, pattern="旧文本", replacement="")
    
    # 示例2：给所有txt文件添加前缀
    # batch_rename_txt_files(target_directory, prefix="前缀_")
    
    # 示例3：给所有txt文件添加后缀
    # batch_rename_txt_files(target_directory, suffix="_后缀")
    
    # 示例4：替换文件名中的空格为下划线
    # batch_rename_txt_files(target_directory, pattern=r"\s+", replacement="_")
    
    # 示例5：顺序重命名为 data1.txt, data2.txt...
    sequential_rename(target_directory, base_name="data", start_num=1)
    
    # 取消下面的注释来运行示例
    # sequential_rename(target_directory, base_name="文件", start_num=1)