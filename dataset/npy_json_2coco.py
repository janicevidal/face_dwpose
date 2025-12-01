import numpy as np
import json
import cv2
import os
import math
from glob import glob
from tqdm import tqdm

def parse_kuaishou_euler(jsonf):
    """解析快手欧拉角数据"""
    assert os.path.exists(jsonf), "file not exists: {}".format(jsonf)
    with open(jsonf, "r") as f:
        data = json.load(f)

    try:
        data = data["data"]
        faceInfos = data["face"]  # 列表类型 多个人脸
        
        if len(faceInfos) == 0:
            print(f"Warning: No face detected in {jsonf}")
            return None
            
        faceInfo = faceInfos[0]
 
        # 每个人脸信息包括:  ['roll', 'yaw', 'pitch', 'age', 'rect', 'point']
        roll = faceInfo["roll"] 
        yaw = faceInfo["yaw"]
        pitch = faceInfo["pitch"]

        return (roll, yaw, pitch)
    
    except (KeyError, Exception) as e:
        print(f"Error: {jsonf} ==> Parsing json failed: {e}")
        return None

def parse_kuaishou_euler_bbox(jsonf):
    """解析快手欧拉角数据和边界框"""
    assert os.path.exists(jsonf), "file not exists: {}".format(jsonf)
    with open(jsonf, "r") as f:
        data = json.load(f)

    try:
        data = data["data"]
        faceInfos = data["face"]  # 列表类型 多个人脸
        
        if len(faceInfos) == 0:
            print(f"Warning: No face detected in {jsonf}")
            return None
            
        faceInfo = faceInfos[0]
 
        # 每个人脸信息包括:  ['roll', 'yaw', 'pitch', 'age', 'rect', 'point']
        roll = faceInfo["roll"] 
        yaw = faceInfo["yaw"]
        pitch = faceInfo["pitch"]
        
        bbox = faceInfo["rect"] 

        return (roll, yaw, pitch, bbox["left"], bbox["top"], bbox["width"], bbox["height"])
    
    except (KeyError, Exception) as e:
        print(f"Error: {jsonf} ==> Parsing json failed: {e}")
        return None

def get_euler_json_path(img_name, json_base_paths):
    """
    根据图像名称获取对应的json文件路径
    
    Args:
        img_name: 图像文件名
        json_base_paths: JSON文件基础路径配置
    """
    # 路径映射
    EULER_PATH_MAP = json_base_paths.get("normal", {
        "ffhq": "/data/caiachang/ffhq/KuaiShou-ldms/rawData_mask",
        "only1face": "/data/caiachang/onlyOneFace/KuaiShou-ldms",
        "celeba": "/data/caiachang/CelebA/video-ldms/KuaiShou-ldms"
    })
    
    EULER_AUG3D_PATH = json_base_paths.get("aug3d", {
        "euler-exp": "/data/caiachang/video-ldms-ok/00Aug3d/ffhq/KuaiShou-pts/aug-all",
        "euler": "/data/caiachang/video-ldms-ok/00Aug3d/ffhq/KuaiShou-pts/aug-euler",
        "exp": "/data/caiachang/video-ldms-ok/00Aug3d/ffhq/KuaiShou-pts/aug-exp"
    })
    
    # 根据图像名称确定对应的json文件路径
    if "euler-exp" in img_name or "euler" in img_name or "exp" in img_name:
        prefix = img_name.split('_')[-1].split('.')[0]
        if prefix in EULER_AUG3D_PATH:
            euler_base_path = EULER_AUG3D_PATH[prefix]
            euler_path = os.path.join(euler_base_path, img_name.split('_')[1] + '.jpg.json')
            return euler_path, "aug3d"
    else:
        prefix = img_name.split('_')[0]
        if prefix in EULER_PATH_MAP:
            euler_base_path = EULER_PATH_MAP[prefix]
            euler_path = os.path.join(euler_base_path, os.path.splitext(img_name.split('_')[1])[0] + '.json')
            return euler_path, "normal"
    
    return None, None

def convert_to_coco_format(image_dir, npy_dir, output_json_path, json_base_paths=None):
    """
    将npy格式的人脸标定数据转换为COCO格式，每个图像只处理一个边界框
    添加欧拉角标注，并支持更新人脸bbox标注
    
    Args:
        image_dir: 图像文件夹路径
        npy_dir: npy文件文件夹路径
        output_json_path: 输出的JSON文件路径
        json_base_paths: JSON文件基础路径配置
    """
    
    # 初始化JSON基础路径
    if json_base_paths is None:
        json_base_paths = {}
    
    # 初始化COCO数据结构
    coco_format = {
        "info": {
            "description": "Face Detection and Landmarks Dataset with Euler Angles",
            "version": "1.0",
            "year": 2023,
            "contributor": "",
            "date_created": "2023/01/01"
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
            "keypoints": [str(i) for i in range(235)],  # 235个关键点
            "skeleton": []
        }]
    }
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(image_dir, ext)))
    
    # 统计信息
    stats = {
        "total_images": 0,
        "successful_annotations": 0,
        "euler_angles_found": 0,
        "bbox_updated": 0
    }
    
    # 计数器
    image_id = 0
    annotation_id = 0
    
    # 处理每张图像
    for img_path in tqdm(image_paths, desc="Converting annotations"):
        stats["total_images"] += 1
        
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
            "date_captured": "2023/01/01",
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        })
        
        # 读取npy文件
        try:
            infodict = np.load(npy_path, allow_pickle=True).tolist()
            landmarks235 = infodict["landmarks208"]  # 注意这里使用了landmarks208键名
            bboxes = infodict["DFSD_facebbox"]
        except Exception as e:
            print(f"Error reading npy file {npy_path}: {e}")
            continue
        
        # 只处理第一个边界框（单个人脸）
        if len(bboxes) > 0:
            bbox = bboxes[0]  # 只取第一个边界框
            x1, y1, w, h = bbox
            
            # 初始化欧拉角
            roll, yaw, pitch = 0.0, 0.0, 0.0
            has_euler_angles = False
            
            # 尝试获取欧拉角数据
            euler_json_path, data_type = get_euler_json_path(img_name, json_base_paths)
            if euler_json_path and os.path.exists(euler_json_path):
                try:
                    if data_type == "aug3d":
                        euler_info = parse_kuaishou_euler_bbox(euler_json_path)
                        if euler_info is not None:
                            roll, yaw, pitch, euler_x1, euler_y1, euler_w, euler_h = euler_info
                            has_euler_angles = True
                            stats["euler_angles_found"] += 1
                            
                            # 使用欧拉角数据中的bbox更新原来的bbox
                            x1, y1, w, h = euler_x1, euler_y1, euler_w, euler_h
                            stats["bbox_updated"] += 1
                    else:  # normal
                        euler_info = parse_kuaishou_euler(euler_json_path)
                        if euler_info is not None:
                            roll, yaw, pitch = euler_info
                            has_euler_angles = True
                            stats["euler_angles_found"] += 1
                except Exception as e:
                    print(f"Error parsing euler json {euler_json_path}: {e}")
            
            # 转换为角度
            yaw_deg = yaw * 180 / np.pi
            pitch_deg = pitch * 180 / np.pi
            roll_deg = roll * 180 / np.pi
            
            if yaw_deg < -72:
                yaw_deg = -72
            if yaw_deg > 72:    
                yaw_deg = 72
            
            if pitch_deg < -48:
                pitch_deg = -48 
            if pitch_deg > 48:  
                pitch_deg = 48
            
            if roll_deg < -99:
                roll_deg = -99 
            if roll_deg > 99:  
                roll_deg = 99
            
            # 计算面积
            area = w * h
            
            # 准备关键点数据 (COCO格式: [x1,y1,v1, x2,y2,v2, ...])
            keypoints = []
            for j in range(235):
                # 确保我们不会超出landmarks235的范围
                if j < len(landmarks235):
                    x, y = landmarks235[j]
                    keypoints.extend([float(x), float(y), 1])  # 2表示标注了且可见
                else:
                    keypoints.extend([0, 0, 0])  # 0表示未标注
            
            # 创建标注信息
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # 人脸类别
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(area),
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": 235,
                # 添加欧拉角信息
                "euler_angles": {
                    "roll": float(roll_deg),
                    "yaw": float(yaw_deg),
                    "pitch": float(pitch_deg)
                },
                "has_euler_angles": has_euler_angles
            }
            
            # 添加标注信息到COCO格式
            coco_format["annotations"].append(annotation)
            
            annotation_id += 1
            stats["successful_annotations"] += 1
        
        image_id += 1
    
    # 保存为JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=4)
    
    # 打印统计信息
    print(f"\n=== 转换完成 ===")
    print(f"输出文件: {output_json_path}")
    print(f"总图像数: {stats['total_images']}")
    print(f"成功标注数: {stats['successful_annotations']}")
    print(f"找到欧拉角: {stats['euler_angles_found']}")
    print(f"更新边界框: {stats['bbox_updated']}")
    
    # 保存统计信息
    stats_path = output_json_path.replace('.json', '_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("COCO格式转换统计信息\n")
        f.write("===================\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"统计信息已保存: {stats_path}")
    
    return coco_format, stats

if __name__ == "__main__":
    # 定义JSON基础路径配置
    json_base_paths = {
        "normal": {
            "ffhq": "/data/caiachang/ffhq/KuaiShou-ldms/rawData_mask",
            "only1face": "/data/caiachang/onlyOneFace/KuaiShou-ldms", 
            "celeba": "/data/caiachang/CelebA/video-ldms/KuaiShou-ldms"
        },
        "aug3d": {
            "euler-exp": "/data/caiachang/video-ldms-ok/00Aug3d/ffhq/KuaiShou-pts/aug-all",
            "euler": "/data/caiachang/video-ldms-ok/00Aug3d/ffhq/KuaiShou-pts/aug-euler",
            "exp": "/data/caiachang/video-ldms-ok/00Aug3d/ffhq/KuaiShou-pts/aug-exp"
        }
    }

    image_dir = "/data/xiaoshuai/facial_lanmark/train_1121/" 
    npy_dir = "/data/xiaoshuai/facial_lanmark/train_1121/"
    output_json_path = "/data/xiaoshuai/facial_lanmark/train_1121/annotations/train_angles_annotations.json"
    
    # image_dir = "/data/xiaoshuai/facial_lanmark/val_1118/" 
    # npy_dir = "/data/xiaoshuai/facial_lanmark/val_1118/"
    # output_json_path = "/data/xiaoshuai/facial_lanmark/val_1118/annotations/val_angles_annotations.json"
    
    convert_to_coco_format(image_dir, npy_dir, output_json_path, json_base_paths)