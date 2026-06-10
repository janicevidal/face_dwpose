import os
import shutil
import argparse
from pathlib import Path

# 支持的图像文件扩展名（可根据需要增减）
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def get_image_files(folder):
    """
    扫描文件夹，返回字典 {基础名: 完整文件名}。
    只处理具有支持扩展名的图像文件。
    """
    image_files = {}
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"文件夹不存在或不是目录: {folder}")
    for filename in os.listdir(folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            base = os.path.splitext(filename)[0]
            image_files[base] = filename
    return image_files

def main():
    parser = argparse.ArgumentParser(
        description="筛选 A 文件夹中有但 B 文件夹中没有的图像，并将图像和对应的 JSON 标注分开保存到不同目录"
    )
    parser.add_argument("folder_A", help="图像文件夹 A 的路径")
    parser.add_argument("folder_B", help="图像文件夹 B 的路径")
    parser.add_argument("json_folder", help="存放 JSON 标注文件的文件夹路径")
    parser.add_argument("output_root", help="输出根目录（将在其中创建 images/ 和 annotations/ 子文件夹）")
    args = parser.parse_args()

    # 获取两个文件夹中的图像文件映射
    images_A = get_image_files(args.folder_A)
    images_B = get_image_files(args.folder_B)

    # 计算仅在 A 中存在的图像基础名
    only_in_A = set(images_A.keys()) - set(images_B.keys())
    if not only_in_A:
        print("没有找到符合条件（仅在 A 中存在）的图像。")
        return

    # 创建分开的输出目录
    out_images_dir = os.path.join(args.output_root, "images")
    out_annotations_dir = os.path.join(args.output_root, "annotations")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_annotations_dir, exist_ok=True)

    # 逐个复制图像及对应的 JSON 文件
    for base in only_in_A:
        # 复制图像文件到 images/ 子目录
        img_filename = images_A[base]
        src_img = os.path.join(args.folder_A, img_filename)
        dst_img = os.path.join(out_images_dir, img_filename)
        shutil.copy2(src_img, dst_img)
        print(f"已复制图像: {img_filename} -> {out_images_dir}")

        # 复制对应的 JSON 标注文件到 annotations/ 子目录
        json_filename = base + ".json"
        src_json = os.path.join(args.json_folder, json_filename)
        if os.path.isfile(src_json):
            dst_json = os.path.join(out_annotations_dir, json_filename)
            shutil.copy2(src_json, dst_json)
            print(f"已复制标注: {json_filename} -> {out_annotations_dir}")
        else:
            print(f"警告: 未找到基础名 '{base}' 对应的 JSON 文件: {src_json}")

    print("处理完成！")
    print(f"图像保存在: {out_images_dir}")
    print(f"标注保存在: {out_annotations_dir}")

if __name__ == "__main__":
    main()