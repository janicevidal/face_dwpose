import os
import shutil

def process_images(folder_a, folder_b, folder_c):
    """
    将文件夹b中的图像复制到文件夹a中，并删除a中文件名在b中出现的文件
    
    参数:
    folder_a: 文件夹a的路径
    folder_b: 文件夹b的路径
    """
    # 确保文件夹存在
    if not os.path.exists(folder_a):
        print(f"错误: 文件夹 {folder_a} 不存在")
        return
    if not os.path.exists(folder_b):
        print(f"错误: 文件夹 {folder_b} 不存在")
        return
    if not os.path.exists(folder_c):
        print(f"错误: 文件夹 {folder_c} 不存在")
        return
    
    # 获取文件夹b中所有图像的文件名（不含后缀）
    b_filenames = set()
    for filename in os.listdir(folder_b):
        if os.path.isfile(os.path.join(folder_b, filename)):
            # 分离文件名和后缀
            name, ext = os.path.splitext(filename)
            b_filenames.add(name)
    
    print(f"文件夹b中找到 {len(b_filenames)} 个图像文件")
    
    c_filenames = set()
        
    for filename in os.listdir(folder_c):
        if os.path.isfile(os.path.join(folder_c, filename)):
            # 分离文件名和后缀
            name, ext = os.path.splitext(filename)
            c_filenames.add(name)
    
    print(f"文件夹c中找到 {len(c_filenames)} 个图像文件")
    
    # 处理文件夹a中的文件
    deleted_count = 0
    for filename in os.listdir(folder_a):
        file_path = os.path.join(folder_a, filename)
        if os.path.isfile(file_path):
            # 分离文件名和后缀
            name, ext = os.path.splitext(filename)
            
            # 如果a中的文件名在b的文件名集合中，删除该文件
            if name in b_filenames or name+"_euler" in b_filenames or name+"_exp" in b_filenames or name+"_euler-exp" in b_filenames:
                os.remove(file_path)
                deleted_count += 1
                print(f"已删除: {filename}")
            
            if name in c_filenames or name+"_euler" in c_filenames or name+"_exp" in c_filenames or name+"_euler-exp" in c_filenames:
                os.remove(file_path)
                deleted_count += 1
                print(f"已删除 val: {filename}")
    
    print(f"从文件夹a中删除了 {deleted_count} 个文件")
    
    import pdb
    pdb.set_trace()
    
    # 将文件夹b中的所有文件复制到文件夹a
    copied_count = 0
    for filename in os.listdir(folder_b):
        src_path = os.path.join(folder_b, filename)
        dst_path = os.path.join(folder_a, filename)
        
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            print(f"已复制: {filename}")
    
    print(f"从文件夹b复制了 {copied_count} 个文件到文件夹a")
    print("操作完成!")

if __name__ == "__main__":
    # 在这里设置你的文件夹路径
    folder_a = "/data/xiaoshuai/facial_lanmark/train_1118/" 
    folder_b = "/data/xiaoshuai/facial_lanmark/ffhq-aug3d/dataset_split/train/" 
    folder_c = "/data/xiaoshuai/facial_lanmark/ffhq-aug3d/dataset_split/val"
    
    # 执行处理
    process_images(folder_a, folder_b, folder_c)