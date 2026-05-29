import os
import shutil

root = r"D:\aaaSCNU\0data\Raman\20260309\foam80"
target = os.path.join(root, "1200")

# 创建目标目录
os.makedirs(target, exist_ok=True)

for current_path, dirs, files in os.walk(root):

    # 找到名为1200的文件夹
    if os.path.basename(current_path) == "1200":

        # 跳过目标文件夹本身
        if os.path.abspath(current_path) == os.path.abspath(target):
            continue

        for file in files:

            if file.endswith(".txt"):

                src = os.path.join(current_path, file)
                dst = os.path.join(target, file)

                # 如果文件重名，自动加编号
                if os.path.exists(dst):
                    name, ext = os.path.splitext(file)
                    i = 1
                    while True:
                        new_name = f"{name}_{i}{ext}"
                        new_dst = os.path.join(target, new_name)
                        if not os.path.exists(new_dst):
                            dst = new_dst
                            break
                        i += 1

                shutil.copy(src, dst)
                print("Copied:", src)

print("All files copied.")