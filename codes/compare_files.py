import os
import hashlib
from pathlib import Path

def file_hash(filepath, algo='md5'):
    """计算文件的哈希值"""
    h = hashlib.new(algo)
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def compare_file_contents(file1, file2):
    """比较两个文件内容是否一致（哈希）"""
    return file_hash(file1) == file_hash(file2)

def compare_directories(dir1, dir2):
    """比较两个根目录下的同名子文件夹中的文件"""
    dir1 = Path(dir1)
    dir2 = Path(dir2)
    common_subdirs = set(p.name for p in dir1.iterdir() if p.is_dir()) & \
                     set(p.name for p in dir2.iterdir() if p.is_dir())

    all_match = True

    for subdir in common_subdirs:
        path1 = dir1 / subdir
        path2 = dir2 / subdir

        files1 = sorted([f.name for f in path1.iterdir() if f.is_file()])
        files2 = sorted([f.name for f in path2.iterdir() if f.is_file()])

        if files1 != files2:
            print(f"[✘] 子文件夹 '{subdir}' 中的文件名不一致：")
            print(f"    {dir1}/{subdir}: {files1}")
            print(f"    {dir2}/{subdir}: {files2}")
            all_match = False
            continue

        for fname in files1:
            f1 = path1 / fname
            f2 = path2 / fname
            if not compare_file_contents(f1, f2):
                print(f"[✘] 文件内容不一致：'{subdir}/{fname}'")
                all_match = False

    if all_match:
        print("✅ 所有同名子目录中的同名文件都一致！")
    else:
        print("⚠️ 存在不一致内容，请检查上面输出")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("用法：python compare_folders.py <目录1> <目录2>")
        sys.exit(1)
    compare_directories(sys.argv[1], sys.argv[2])
