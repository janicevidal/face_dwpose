import os
import shutil
import argparse

def batch_rename_and_copy(src_folder, dst_folder, supported_extensions=None):
    """
    批量重命名图像文件并拷贝到目标文件夹
    
    参数:
        src_folder: 源文件夹路径
        dst_folder: 目标文件夹路径
        supported_extensions: 支持的图像文件扩展名列表
    """
    # 设置默认支持的图像格式
    if supported_extensions is None:
        supported_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    # 确保目标文件夹存在
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        print(f"创建目标文件夹: {dst_folder}")
    
    # 获取源文件夹的名称
    folder_name = os.path.basename(os.path.normpath(src_folder))
    
    # 计数器
    processed_count = 0
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_folder):
        file_path = os.path.join(src_folder, filename)
        
        # 跳过目录，只处理文件
        if not os.path.isfile(file_path):
            continue
        
        # 检查文件扩展名是否在支持的列表中
        _, ext = os.path.splitext(filename)
        if ext.lower() not in supported_extensions:
            print(f"跳过不支持的文件: {filename}")
            continue
        
        # 构建新文件名: 文件夹名_原文件名
        new_filename = f"{folder_name}_{filename}"
        dst_path = os.path.join(dst_folder, new_filename)
        
        try:
            # 拷贝文件到目标文件夹
            shutil.copy2(file_path, dst_path)
            print(f"已处理: {filename} -> {new_filename}")
            processed_count += 1
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
    
    print(f"处理完成! 共处理了 {processed_count} 个文件")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='批量重命名图像文件并拷贝到目标文件夹')
    parser.add_argument('source', help='源文件夹路径')
    parser.add_argument('destination', help='目标文件夹路径')
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.npy'],
                       help='支持的图像文件扩展名列表 (默认: .jpg .jpeg .png .gif .bmp)')
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查源文件夹是否存在
    if not os.path.exists(args.source):
        print(f"错误: 源文件夹 '{args.source}' 不存在!")
        return
    
    # 执行重命名和拷贝操作
    batch_rename_and_copy(args.source, args.destination, args.extensions)

if __name__ == "__main__":
    main()