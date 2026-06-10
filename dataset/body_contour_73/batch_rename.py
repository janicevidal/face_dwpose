#!/usr/bin/env python3
"""
批量重命名图片文件，生成固定位数的数字序列文件名。
示例：将目录中的图片重命名为 0000.png, 0001.png, ..., 0120.png
"""

import os
import argparse
import glob
from natsort import natsorted  # 自然排序，需要安装：pip install natsort

def batch_rename(directory, extensions=None, start=0, digits=4, dry_run=True):
    """
    批量重命名文件

    :param directory:  目标目录路径
    :param extensions: 文件扩展名列表，例如 ['.png', '.jpg', '.jpeg']
    :param start:      起始编号（整数）
    :param digits:     编号位数（自动补零）
    :param dry_run:    是否仅预览，不实际重命名
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']

    # 收集所有匹配扩展名的文件
    files = []
    for ext in extensions:
        pattern = os.path.join(directory, f'*{ext}')
        files.extend(glob.glob(pattern))

    if not files:
        print("未找到任何匹配的图片文件。")
        return

    # 自然排序（例如 1.png, 2.png, 10.png 按数字顺序）
    files = natsorted(files)

    print(f"找到 {len(files)} 个文件，将重命名为 {start:0{digits}d} ~ {start+len(files)-1:0{digits}d}")

    for idx, old_path in enumerate(files):
        new_name = f"{start + idx:0{digits}d}{os.path.splitext(old_path)[1]}"
        new_path = os.path.join(directory, new_name)

        if dry_run:
            print(f"[预览] {os.path.basename(old_path)} -> {new_name}")
        else:
            # 如果目标文件已存在且不是同一个文件，提示跳过（安全保护）
            if os.path.exists(new_path) and new_path != old_path:
                print(f"跳过 {old_path} -> {new_name} (目标文件已存在)")
                continue
            os.rename(old_path, new_path)
            print(f"重命名: {os.path.basename(old_path)} -> {new_name}")

    if dry_run:
        print("\n这是预览模式，未实际修改任何文件。移除 --dry-run 参数以执行重命名。")

def main():
    parser = argparse.ArgumentParser(description="批量重命名图片文件为数字序列")
    parser.add_argument("directory", nargs="?", default=".", help="目标目录（默认为当前目录）")
    parser.add_argument("--ext", nargs="+", default=[".png", ".jpg", ".jpeg", ".gif", ".bmp"],
                        help="要处理的扩展名列表，例如 --ext .png .jpg")
    parser.add_argument("--start", type=int, default=0, help="起始编号（默认 0）")
    parser.add_argument("--digits", type=int, default=4, help="编号位数（默认 4）")
    parser.add_argument("--no-dry-run", action="store_true", help="实际执行重命名（默认仅预览）")
    args = parser.parse_args()

    batch_rename(
        directory=args.directory,
        extensions=args.ext,
        start=args.start,
        digits=args.digits,
        dry_run=not args.no_dry_run
    )

if __name__ == "__main__":
    main()