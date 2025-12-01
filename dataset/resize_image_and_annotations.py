import os
import cv2
import glob
import pdb
import numpy as np
import shutil

def resize_image_and_annotations(input_path, output_path, target_max_side=192, ffhq_target_size=320):
    """
    根据图像类型智能缩放图像并更新标注
    
    Args:
        input_path: 输入文件夹路径
        output_path: 输出文件夹路径
        target_max_side: 普通人脸图像的目标长边尺寸，默认为192
        ffhq_target_size: FFHQ图像的目标长边尺寸，默认为320
    """
    
    # 创建输出文件夹
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 获取所有图像文件
    if input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
        input_img_list = [input_path]
    else:
        if input_path.endswith('/'):
            input_path = input_path[:-1]
        input_img_list = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))
    
    if len(input_img_list) == 0:
        raise FileNotFoundError('No input image found...')
    
    processed_count = 0
    skipped_count = 0
    ffhq_processed_count = 0
    
    for i, img_path in enumerate(input_img_list):
        img_name = os.path.basename(img_path)
        print(f'[{i+1}/{len(input_img_list)}] Processing: {img_name}')
        
        # 对应的npy文件路径
        npy_path = os.path.splitext(img_path)[0] + '.npy'
        
        if not os.path.exists(npy_path):
            print(f"Warning: No corresponding npy file for {img_name}, skipping...")
            pdb.set_trace()
            continue
        
        # 加载图像和标注
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Cannot read image {img_path}, skipping...")
            pdb.set_trace()
            continue
            
        infodict = np.load(npy_path, allow_pickle=True).tolist()
        
        # 检查必要的键是否存在
        required_keys = ['landmarks208', 'DFSD_facebbox']
        if not all(key in infodict for key in required_keys):
            print(f"Warning: Missing required keys in {npy_path}, skipping...")
            pdb.set_trace()
            continue
        
        landmarks208 = infodict['landmarks208']
        bboxes = infodict['DFSD_facebbox']
        
        # 检查图像是否为FFHQ类型
        is_ffhq = "ffhq" in img_name.lower()
        
        if is_ffhq:
            # FFHQ图像处理：直接缩放整个图像
            print("  FFHQ image detected, using direct scaling")
            
            # 计算图像的长边
            height, width = image.shape[:2]
            image_max_side = max(height, width)
            
            # 如果图像长边小于等于目标尺寸，直接复制文件
            if image_max_side <= ffhq_target_size:
                # 复制图像文件
                output_img_path = os.path.join(output_path, img_name)
                shutil.copy2(img_path, output_img_path)
                
                # 复制标注文件
                output_npy_name = os.path.splitext(img_name)[0] + '.npy'
                output_npy_path = os.path.join(output_path, output_npy_name)
                shutil.copy2(npy_path, output_npy_path)
                
                skipped_count += 1
                print(f"  FFHQ image size {image_max_side} <= {ffhq_target_size}, copied as is")
                continue
            
            # 计算缩放比例
            scale_factor = ffhq_target_size / image_max_side
            print(f"  Scaling FFHQ image by factor: {scale_factor:.4f}")
            
            # 缩放图像
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # 更新关键点坐标
            updated_landmarks = []
            for landmark in landmarks208:
                if len(landmark) >= 2:
                    updated_x = landmark[0] * scale_factor
                    updated_y = landmark[1] * scale_factor
                    updated_landmarks.append([updated_x, updated_y])
                else:
                    updated_landmarks.append(landmark)  # 保持原样如果格式不对
            
            # 更新人脸框坐标（如果有）
            updated_bboxes = []
            if len(bboxes) > 0:
                for bbox in bboxes:
                    if len(bbox) >= 4:
                        x1, y1, w, h = bbox
                        updated_bbox = [
                            x1 * scale_factor,
                            y1 * scale_factor,
                            w * scale_factor,
                            h * scale_factor
                        ]
                        updated_bboxes.append(updated_bbox)
                    else:
                        updated_bboxes.append(bbox)
            else:
                updated_bboxes = bboxes
            
            ffhq_processed_count += 1
            
        else:
            # 普通人脸图像处理：基于人脸框缩放
            if len(bboxes) == 0:
                print(f"Warning: No face bbox found in {npy_path}, skipping...")
                pdb.set_trace()
                continue
            
            # 获取第一个人脸框（假设每张图只有一个人脸）
            gtbox = bboxes[0]
            if len(gtbox) < 4:
                print(f"Warning: Invalid bbox format in {npy_path}, skipping...")
                pdb.set_trace()
                continue
                
            x1, y1, w, h = gtbox
            face_max_side = max(w, h)
            
            if face_max_side < 40:
               print(f"Error: face_max_side {img_name} has a small value of {(w, h)} ")
               pdb.set_trace()
            
            # 如果人脸框长边小于等于目标尺寸，直接复制文件
            if face_max_side <= target_max_side:
                # 复制图像文件
                output_img_path = os.path.join(output_path, img_name)
                shutil.copy2(img_path, output_img_path)
                
                # 复制标注文件
                output_npy_name = os.path.splitext(img_name)[0] + '.npy'
                output_npy_path = os.path.join(output_path, output_npy_name)
                shutil.copy2(npy_path, output_npy_path)
                
                skipped_count += 1
                print(f"  Face size {face_max_side} <= {target_max_side}, copied as is")
                continue
            
            # 计算缩放比例
            scale_factor = target_max_side / face_max_side
            print(f"  Scaling image by factor: {scale_factor:.4f}")
            
            # 缩放图像
            new_height = int(image.shape[0] * scale_factor)
            new_width = int(image.shape[1] * scale_factor)
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            if max(new_width, new_height) > 2000:
               print(f"Error: Resized image {img_name} has a long side of {max(new_width, new_height)} which exceeds 2000. Skipping.")
               print(x1, y1, w, h)
               pdb.set_trace()
               continue
            
            # 更新关键点坐标
            updated_landmarks = []
            for landmark in landmarks208:
                if len(landmark) >= 2:
                    updated_x = landmark[0] * scale_factor
                    updated_y = landmark[1] * scale_factor
                    updated_landmarks.append([updated_x, updated_y])
                else:
                    updated_landmarks.append(landmark)  # 保持原样如果格式不对
            
            # 更新人脸框坐标
            updated_bboxes = []
            for bbox in bboxes:
                if len(bbox) >= 4:
                    x1, y1, w, h = bbox
                    updated_bbox = [
                        x1 * scale_factor,
                        y1 * scale_factor,
                        w * scale_factor,
                        h * scale_factor
                    ]
                    updated_bboxes.append(updated_bbox)
                else:
                    updated_bboxes.append(bbox)
            
            processed_count += 1
        
        # 更新标注字典
        updated_infodict = infodict.copy()
        updated_infodict['landmarks208'] = updated_landmarks
        updated_infodict['DFSD_facebbox'] = updated_bboxes
        
        # 保存处理后的图像
        output_img_path = os.path.join(output_path, img_name)
        cv2.imwrite(output_img_path, resized_image)
        
        # 保存更新后的标注文件
        output_npy_name = os.path.splitext(img_name)[0] + '.npy'
        output_npy_path = os.path.join(output_path, output_npy_name)
        np.save(output_npy_path, updated_infodict)
        
        print(f"  Resized: {image.shape[:2]} -> {resized_image.shape[:2]}")
    
    print(f"\nProcessing completed!")
    print(f"Total images: {len(input_img_list)}")
    print(f"FFHQ images resized: {ffhq_processed_count}")
    print(f"Regular images resized: {processed_count}")
    print(f"Skipped (no resize needed): {skipped_count}")

def main():
    # 设置输入输出路径
    # input_path = "/data/xiaoshuai/facial_lanmark/train_1118"
    # output_path = '/data/xiaoshuai/facial_lanmark/train_1118_resized'
    
    input_path = "/data/xiaoshuai/facial_lanmark/val_1118"
    output_path = '/data/xiaoshuai/facial_lanmark/val_1118_resized'
    
    # 目标尺寸设置
    regular_target_size = 192  # 普通人脸图像的目标长边尺寸
    ffhq_target_size = 320     # FFHQ图像的目标长边尺寸
    
    # 执行缩放处理
    resize_image_and_annotations(input_path, output_path, regular_target_size, ffhq_target_size)

if __name__ == '__main__':
    main()