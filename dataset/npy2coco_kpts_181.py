import numpy as np
import json
import cv2
import os
from glob import glob
from tqdm import tqdm

def convert_to_coco_format(image_dir, npy_dir, output_json_path):
    """
    将npy格式的人脸标定数据转换为COCO格式，每个图像只处理一个边界框
    
    Args:
        image_dir: 图像文件夹路径
        npy_dir: npy文件文件夹路径
        output_json_path: 输出的JSON文件路径
    """
    
    # 初始化COCO数据结构
    coco_format = {
        "info": {
            "description": "Face Landmarks 181 Dataset",
            "version": "1.0",
            "year": 2026,
            "contributor": "",
            "date_created": ""
        },
        "licenses": [{
            "id": 1,
            "name": "License",
            "url": ""
        }],
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "face",
            "supercategory": "person",
            "keypoints": [str(i) for i in range(181)],  # 181个关键点
            "skeleton": []
        }]
    }
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(image_dir, ext)))
    
    # 计数器
    image_id = 0
    annotation_id = 0
    
    # 处理每张图像
    for img_path in tqdm(image_paths, desc="Converting annotations"):
        # 获取对应的npy文件路径
        img_name = os.path.basename(img_path)
        npy_name = os.path.splitext(img_name)[0] + '.npy'
        npy_path = os.path.join(npy_dir, npy_name)
        
        if not os.path.exists(npy_path):
            print(f"Warning: No npy file found for {img_name}")
            continue
        
        # 读取图像获取尺寸
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Cannot read image {img_path}")
            continue
            
        height, width = image.shape[:2]
        
        # 添加图像信息到COCO格式
        coco_format["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": width,
            "height": height,
            "date_captured": "",
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        })
        
        # 读取npy文件
        try:
            infodict = np.load(npy_path, allow_pickle=True).tolist()
            landmarks181 = infodict["landmarks181"]
            bboxes = infodict["DFSD_facebbox"]
            euler_info = infodict["eulerXY"]
            # 左上负右下正，转换为右手坐标系
            roll, pitch, yaw = euler_info[0], euler_info[1], -euler_info[2]  
            roll_np, yaw_np, pitch_np = np.asarray(roll), np.asarray(yaw), np.asarray(pitch)
            
            if yaw_np < -72:
                yaw_np = -72
            if yaw_np > 72:    
                yaw_np = 72
            
            if pitch_np < -48:
                pitch_np = -48 
            if pitch_np > 48:  
                pitch_np = 48
            
            if roll_np < -99:
                roll_np = -99 
            if roll_np > 99:  
                roll_np = 99
        
        except Exception as e:
            print(f"Error reading npy file {npy_path}: {e}")
            continue
        
        # 只处理第一个边界框（单个人脸）
        if len(bboxes) > 0:
            bbox = bboxes[0]  # 只取第一个边界框
            x1, y1, w, h = bbox
            
            # 计算面积
            area = w * h
            
            # 准备关键点数据 (COCO格式: [x1,y1,v1, x2,y2,v2, ...])
            keypoints = []
            for j in range(181):
                # 确保我们不会超出landmarks181的范围
                if j < len(landmarks181):
                    x, y = landmarks181[j]
                    keypoints.extend([float(x), float(y), 1])  # 2表示标注了且可见
                else:
                    keypoints.extend([0, 0, 0])  # 0表示未标注
            
            # 添加标注信息到COCO格式
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # 人脸类别
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(area),
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": 181,
                # 添加欧拉角信息
                "euler_angles": {
                    "roll": float(roll_np),
                    "yaw": float(yaw_np),
                    "pitch": float(pitch_np)
                },
                "has_euler_angles": True
            })
            
            annotation_id += 1
        
        image_id += 1
    
    # 保存为JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=4)
    
    print(f"Conversion completed. Saved to {output_json_path}")
    print(f"Total images: {len(coco_format['images'])}")
    print(f"Total annotations: {len(coco_format['annotations'])}")

if __name__ == "__main__":

    image_dir = "/data/xiaoshuai/facial_landmark_181/train_0310/train/" 
    npy_dir = "/data/xiaoshuai/facial_landmark_181/train_0310/train/"
    output_json_path = "/data/xiaoshuai/facial_landmark_181/train_0310/annotations/train_all_annotations.json"
    
    # image_dir = "/data/xiaoshuai/facial_landmark_181/train_0310/val/" 
    # npy_dir = "/data/xiaoshuai/facial_landmark_181/train_0310/val/"
    # output_json_path = "/data/xiaoshuai/facial_landmark_181/train_0310/annotations/val_all_annotations.json"
    
    convert_to_coco_format(image_dir, npy_dir, output_json_path)