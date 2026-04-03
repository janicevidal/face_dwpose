#!/usr/bin/env python3
# encoding: utf-8

import os
import json
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def convert_txt_to_coco(txt_dir, img_dir, output_json_path):
    """
    将包含关键点标注的 TXT 文件转换为 COCO 格式的 JSON 文件。

    参数：
        txt_dir: 存放 TXT 文件的文件夹路径
        img_dir: 存放图像文件的文件夹路径
        output_json_path: 输出的 JSON 文件路径
    """
    # 获取所有 TXT 文件
    txt_paths = glob(os.path.join(txt_dir, "*.txt"))
    if not txt_paths:
        print("错误：未找到任何 TXT 文件。")
        return
    
    def count_keypoints_from_first_file():
        with open(txt_paths[0], 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) == 1:
            # 单行格式
            parts = lines[0].split(',')
            return (len(parts) - 1) // 3
        else:
            # 多行格式
            return len(lines)  # 每行一个点
    
    num_points = count_keypoints_from_first_file()

    if num_points <= 0:
        print("错误：无法确定关键点数量。")
        return
    print(f"检测到关键点数量：{num_points}")

    # 初始化 COCO 数据结构
    coco_format = {
        "info": {
            "description": "Keypoint Dataset from TXT annotations",
            "version": "1.0",
            "year": 2026,
            "contributor": "",
            "date_created": "2026/01/01"
        },
        "licenses": [{
            "id": 1,
            "name": "Unknown",
            "url": ""
        }],
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "person",          # 可根据实际修改类别名
            "supercategory": "person",
            "keypoints": [str(i) for i in range(num_points)],  # 关键点名称列表
            "skeleton": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), 
                         (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), 
                         (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (25, 27), (27, 28), (28, 29), (29, 30), 
                         (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 37), (37, 36), (37, 38), (38, 39), (39, 40), 
                         (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), 
                         (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 60), 
                         (60, 61), (61, 62), (63, 67), (67, 65), (64, 67), (67, 66), (66, 68), (68, 72), (72, 70), (69, 72), (72, 71)] # 骨架连接（可选）
        }]
    }

    image_id = 0
    annotation_id = 0

    for txt_path in tqdm(txt_paths, desc="处理 TXT 文件"):
        # 读取 TXT 文件
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            continue
        
        # 判断格式：单行还是多行
        if len(lines) == 1:
            # 单行格式：第一项为图像文件名，之后为关键点数据
            parts = lines[0].split(',')
            if len(parts) < 2:
                continue
            img_name = os.path.basename(parts[0]).replace('.txt', '.jpg')
            try:
                values = list(map(float, parts[1:]))
            except ValueError:
                print(f"警告：文件 {txt_path} 包含非数值数据，跳过")
                continue
            
            if len(values) != 3 * num_points:
                print(f"警告：文件 {txt_path} 的关键点数量与预期 ({num_points}) 不符，跳过")
                continue
        else:
            # continue
            # 多行格式：每行三个数值，图像文件名取自 TXT 文件名
            img_name = os.path.basename(txt_path).replace('.txt', '.jpg')
            values = []
            for line in lines:
                parts = line.split(',')
                if len(parts) != 3:
                    print(f"警告：文件 {txt_path} 的行格式不正确（应为 x,y,v），跳过该行")
                    continue
                try:
                    x, y, v = map(float, parts)
                except ValueError:
                    print(f"警告：文件 {txt_path} 包含非数值数据，跳过该行")
                    continue
                values.extend([x, y, v])
            
            if len(values) != 3 * num_points:
                print(f"警告：文件 {txt_path} 的关键点数量与预期 ({num_points}) 不符，跳过")
                continue
            
        img_name = os.path.basename(img_name)

        # 读取图像尺寸
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            # 尝试其他常见扩展名
            base = os.path.splitext(img_path)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                test_path = base + ext
                if os.path.exists(test_path):
                    img_path = test_path
                    img_name = os.path.basename(test_path)
                    break
            else:
                print(f"警告：找不到图像 {img_name}，跳过")
                continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图像 {img_path}，跳过")
            continue
        height, width = img.shape[:2]

        # 收集可见点（v>0）用于计算边界框
        xs, ys = [], []
        for i in range(0, len(values), 3):
            v = int(values[i+2])
            if v > 0:
                xs.append(values[i])
                ys.append(values[i+1])

        if xs and ys:
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            # 确保边界框在图像内
            x_min = int(max(0, x_min) + 0.5)
            y_min = int(max(0, y_min) + 0.5)
            x_max = int(min(width, x_max) + 0.5)
            y_max = int(min(height, y_max) + 0.5)
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            if bbox_w <= 0 or bbox_h <= 0:
                bbox = [0, 0, width, height]
                area = width * height
            else:
                bbox = [x_min, y_min, bbox_w, bbox_h]
                area = bbox_w * bbox_h
        else:
            print(f"警告：图像 {img_name} 没有有效关键点，跳过")
            continue

        # 构建 COCO 关键点列表，不可见点坐标置零
        keypoints_coco = []
        num_valid = 0
        for i in range(0, len(values), 3):
            x, y, v = values[i], values[i+1], int(values[i+2])
            if v > 0:
                num_valid += 1
            else:
                # 不可见点（v <= 0）坐标设为 0
                x = 0.0
                y = 0.0
            keypoints_coco.extend([float(x), float(y), v])

        # 添加图像信息
        coco_format["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": width,
            "height": height,
            "date_captured": "2026/01/01",
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        })

        # 添加标注信息
        coco_format["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [float(v) for v in bbox],
            "area": float(area),
            "iscrowd": 0,
            "keypoints": keypoints_coco,
            "num_keypoints": num_valid
        })

        image_id += 1
        annotation_id += 1

    # 保存 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=4, ensure_ascii=False)

    print(f"转换完成。共处理 {len(coco_format['images'])} 张图像，{len(coco_format['annotations'])} 个标注。")
    print(f"输出文件：{output_json_path}")

if __name__ == "__main__":
    # 请根据实际情况修改以下路径
    # txt_dir = "/data/xiaoshuai/body_keypoint/testtxts/"      # TXT 文件所在文件夹
    # img_dir = "/data/xiaoshuai/body_keypoint/testimages/"    # 图像所在文件夹
    # output_json = "/data/xiaoshuai/body_keypoint/val_annotations.json"           # 输出 JSON 文件名
    
    # txt_dir = "/data/xiaoshuai/body_keypoint/traintxts/"
    # img_dir = "/data/xiaoshuai/body_keypoint/trainimages/"
    # # output_json = "/data/xiaoshuai/body_keypoint/train_annotations.json"
    # output_json = "/data/xiaoshuai/body_keypoint/train_annotations_2.json"
    
    # txt_dir = "/data/xiaoshuai/body_keypoint/traintxts_ok/"
    # img_dir = "/data/xiaoshuai/body_keypoint/trainimages_ok/"
    # output_json = "/data/xiaoshuai/body_keypoint/train_annotations_ok.json"
    
    txt_dir = "/data/xiaoshuai/body_keypoint/testtxts_ok/"
    img_dir = "/data/xiaoshuai/body_keypoint/testimages_ok/"
    output_json = "/data/xiaoshuai/body_keypoint/val_annotations_ok.json"

    convert_txt_to_coco(txt_dir, img_dir, output_json)