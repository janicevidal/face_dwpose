#!/usr/bin/env python3
# encoding: utf-8

import json
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

def visualize_annotation(image_path, ann, skeleton, output_path=None):
    """
    在图像上绘制边界框和关键点，并保存或返回图像。
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"警告：无法读取图像 {image_path}")
        return None

    # 绘制边界框
    # x, y, w, h = ann['bbox']
    # cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)

    # 提取可见关键点
    keypoints = ann['keypoints']
    pts = []
    for i in range(0, len(keypoints), 3):
        x_kp, y_kp, v = keypoints[i], keypoints[i+1], keypoints[i+2]
        if v == 1:
            point_color = (255, 0, 0)   # 蓝色表示遮挡
        elif v == 2:
            point_color = (0, 255, 0)   # 绿色表示可见
        else:
            point_color = (0, 0, 255)    # 红色表示未知
            
        if v > 0:  # 只绘制可见点
            cv2.circle(img, (int(x_kp), int(y_kp)), 3, point_color, -1)
            pts.append((int(x_kp), int(y_kp)))
        else:
            pts.append(None)  # 保持索引对应，但不可见点无坐标

    # 绘制骨架（如果定义了连接）
    for conn in skeleton:
        if pts[conn[0]] is not None and pts[conn[1]] is not None:
            if conn[0] < 63:
                cv2.line(img, pts[conn[0]], pts[conn[1]], (255, 128, 0), 2)
            else:
                cv2.line(img, pts[conn[0]], pts[conn[1]], (0, 0, 255), 2)

    if output_path:
        cv2.imwrite(output_path, img)
        return None
    else:
        return img

def validate_coco_dataset(image_dir, json_file, visualize_all=False, output_dir='vis_output'):
    """
    验证COCO格式的关键点数据集。
    - 检查图像与标注的一致性
    - 统计标注数量
    - 可选：为所有图像生成可视化结果
    """
    # 加载COCO数据
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    # 建立 image_id -> image_info 映射
    id_to_image = {img['id']: img for img in coco_data['images']}
    # 建立 image_id -> annotations 列表映射
    image_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)

    # 统计信息
    total_images = len(coco_data['images'])
    total_annotations = len(coco_data['annotations'])
    print(f"图像总数: {total_images}")
    print(f"标注总数: {total_annotations}")

    # 检查每个图像的标注数量
    multi_ann_images = []
    zero_ann_images = []
    for img_id, img_info in id_to_image.items():
        ann_count = len(image_to_anns.get(img_id, []))
        if ann_count == 0:
            zero_ann_images.append(img_info['file_name'])
        elif ann_count > 1:
            multi_ann_images.append(img_info['file_name'])

    if zero_ann_images:
        print(f"警告：{len(zero_ann_images)} 张图像没有标注：")
        for name in zero_ann_images[:10]:  # 只显示前10个
            print(f"  {name}")
        if len(zero_ann_images) > 10:
            print(f"  ... 等共 {len(zero_ann_images)} 张")
    else:
        print("所有图像都有至少一个标注。")

    if multi_ann_images:
        print(f"警告：{len(multi_ann_images)} 张图像有多个标注：")
        for name in multi_ann_images[:10]:
            print(f"  {name}")
        if len(multi_ann_images) > 10:
            print(f"  ... 等共 {len(multi_ann_images)} 张")
    else:
        print("所有图像都有且仅有一个标注（符合单人数据集预期）。")

    # 如果不需要可视化，直接返回统计信息
    if not visualize_all:
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取骨架连接（从 categories 中读取，若不存在则使用空列表）
    skeleton = []
    if 'categories' in coco_data and len(coco_data['categories']) > 0:
        skeleton = coco_data['categories'][0].get('skeleton', [])
    if not skeleton:
        print("警告：JSON中未定义骨架连接，可视化时将不会绘制连线。")

    # 为每个有标注的图像生成可视化
    print(f"开始生成可视化图像，保存至 {output_dir} ...")
    for img_id, anns in tqdm(image_to_anns.items(), desc="可视化进度"):
        img_info = id_to_image.get(img_id)
        if img_info is None:
            continue

        img_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            print(f"图像文件不存在：{img_path}")
            continue

        # 如果该图像有多个标注，为每个标注单独生成一张图（文件名加后缀）
        if len(anns) == 1:
            out_name = os.path.splitext(img_info['file_name'])[0] + '_vis.jpg'
            out_path = os.path.join(output_dir, out_name)
            visualize_annotation(img_path, anns[0], skeleton, out_path)
        else:
            for idx, ann in enumerate(anns):
                out_name = os.path.splitext(img_info['file_name'])[0] + f'_ann{idx}_vis.jpg'
                out_path = os.path.join(output_dir, out_name)
                visualize_annotation(img_path, ann, skeleton, out_path)

    print(f"可视化完成，结果保存在 {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='验证COCO格式关键点数据集')
    parser.add_argument('--image_dir', type=str, required=True, help='图像文件夹路径')
    parser.add_argument('--json_file', type=str, required=True, help='COCO格式的JSON标注文件路径')
    parser.add_argument('--visualize', action='store_true', help='是否为所有图像生成可视化结果')
    parser.add_argument('--output_dir', type=str, default='vis_output', help='可视化结果输出目录（默认：vis_output）')
    args = parser.parse_args()

    validate_coco_dataset(
        image_dir=args.image_dir,
        json_file=args.json_file,
        visualize_all=args.visualize,
        output_dir=args.output_dir
    )