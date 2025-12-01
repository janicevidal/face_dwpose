import os
import shutil
import random
from pathlib import Path

def split_dataset_with_common_names(folders, output_dir, val_ratio=0.01):
    """
    将多个文件夹中的图像分割为训练集和验证集
    确保同名图像同时进入验证集
    
    Args:
        folders: 源文件夹列表
        output_dir: 输出目录
        val_ratio: 验证集比例
    """
    
    # 创建输出目录
    train_dir = Path(output_dir) / 'train'
    val_dir = Path(output_dir) / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有图像信息
    all_images = {}
    
    for folder in folders:
        folder_path = Path(folder)
        print(folder_path)
        if not folder_path.exists():
            print(f"警告: 文件夹 {folder} 不存在，跳过")
            continue
            
        folder_name = folder_path.name
        print(f"正在扫描文件夹: {folder_name}")
        
        # 支持常见的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        for img_path in folder_path.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                filename = img_path.stem  # 不含扩展名的文件名
                
                if filename not in all_images:
                    all_images[filename] = []
                
                all_images[filename].append({
                    'src_path': img_path,
                    'npy_path': str(img_path).split('.')[0] + '.npy',
                    'folder_name': folder_name,
                    'filename': filename,
                    'extension': img_path.suffix
                })
    
    print(f"总共找到 {len(all_images)} 个不同的图像名")
    
    # 计算需要抽取的验证集数量
    total_unique_names = len(all_images)
    val_count = max(1, int(total_unique_names * val_ratio))
    print(f"需要抽取 {val_count} 个图像名到验证集")
    
    # 随机选择验证集的图像名
    all_names = list(all_images.keys())
    random.shuffle(all_names)
    val_names = set(all_names[:val_count])
    train_names = set(all_names[val_count:])
    
    print(f"验证集包含 {len(val_names)} 个图像名")
    print(f"训练集包含 {len(train_names)} 个图像名")
    
    # 复制图像到相应目录
    copied_count = {'train': 0, 'val': 0}
    
    # 处理验证集
    for name in val_names:
        for img_info in all_images[name]:
            new_filename = f"ffhq_{img_info['filename']}_{img_info['folder_name']}{img_info['extension']}"
            new_filename_anno = f"ffhq_{img_info['filename']}_{img_info['folder_name']}.npy"
            dst_path = val_dir / new_filename
            dst_path_anno = val_dir / new_filename_anno
            
            # 复制文件
            shutil.copy2(img_info['src_path'], dst_path)
            shutil.copy2(img_info['npy_path'], dst_path_anno)
            copied_count['val'] += 1
    
    # 处理训练集
    for name in train_names:
        for img_info in all_images[name]:
            new_filename = f"ffhq_{img_info['filename']}_{img_info['folder_name']}{img_info['extension']}"
            new_filename_anno = f"ffhq_{img_info['filename']}_{img_info['folder_name']}.npy"
            dst_path = train_dir / new_filename
            dst_path_anno = train_dir / new_filename_anno
            
            # 复制文件
            shutil.copy2(img_info['src_path'], dst_path)
            shutil.copy2(img_info['npy_path'], dst_path_anno)
            copied_count['train'] += 1
    
    print(f"\n处理完成!")
    print(f"验证集: {copied_count['val']} 张图像")
    print(f"训练集: {copied_count['train']} 张图像")
    print(f"输出目录: {output_dir}")

def main():
    # 配置参数
    folders = ['/data/xiaoshuai/facial_lanmark/ffhq-aug3d/euler', '/data/xiaoshuai/facial_lanmark/ffhq-aug3d/exp', '/data/xiaoshuai/facial_lanmark/ffhq-aug3d/euler-exp']  # 源文件夹
    output_dir = '/data/xiaoshuai/facial_lanmark/ffhq-aug3d/dataset_split'  # 输出目录
    val_ratio = 0.01  # 验证集比例 1%
    
    # 设置随机种子以确保可重复性
    random.seed(33)
    
    # 执行分割
    split_dataset_with_common_names(folders, output_dir, val_ratio)

if __name__ == "__main__":
    main()