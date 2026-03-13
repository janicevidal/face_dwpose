import json
import cv2
import numpy as np
import os
from math import cos, sin

# ==================== 配置区域 ====================
# image_dir = "/data/xiaoshuai/facial_lanmark/train_0126/images/"
# json_path = '/data/xiaoshuai/facial_lanmark/train_0126/annotations/train_filtered_annotations.json'
# output_dir = "/data/xiaoshuai/facial_lanmark/train_0126/visualize/filtered_train_0126"

image_dir = "/data/xiaoshuai/facial_lanmark/train_0126/val_1229/"
json_path = '/data/xiaoshuai/facial_lanmark/train_0126/annotations/val_filtered_annotations.json'
output_dir = "/data/xiaoshuai/facial_lanmark/train_0126/visualize/filtered_val_0126"    

# =================================================

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 加载COCO数据
with open(json_path, 'r') as f:
    coco_data = json.load(f)

def draw_axis(img, yaw_, pitch_, roll_, tdx=None, tdy=None, size=100):
    """
    在图像上绘制头部姿态轴
    """
    pitch = -(pitch_ * np.pi / 180)
    yaw = yaw_ * np.pi / 180
    roll = -(roll_ * np.pi / 180)
    
    yaw_value = yaw_
    pitch_value = pitch_
    roll_value = roll_

    if tdx is not None and tdy is not None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis (red) - 向右
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis (green) - 向下
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (blue) - 向外
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    pitch_color = (255, 255, 0)
    yaw_color   = (0, 255, 0)
    roll_color  = (0, 0, 255)

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 4)
    
    cv2.putText(img, f"Pitch:{pitch_value:.2f}", (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, pitch_color)
    cv2.putText(img, f"Yaw:{yaw_value:.2f}", (0, 30), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
    cv2.putText(img, f"Roll:{roll_value:.2f}", (0, 50), cv2.FONT_HERSHEY_PLAIN, 1, roll_color)

    return img

# 构建 image_id -> 标注列表 的映射
annotations_by_image = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    if img_id not in annotations_by_image:
        annotations_by_image[img_id] = []
    annotations_by_image[img_id].append(ann)

print(f"总共 {len(coco_data['images'])} 张图像，{len(coco_data['annotations'])} 个标注")

# 遍历所有图像
for img_info in coco_data['images']:
    img_id = img_info['id']
    file_name = img_info['file_name']
    img_path = os.path.join(image_dir, file_name)
    
    # 读取图像
    image = cv2.imread(img_path)
    if image is None:
        print(f"警告：无法读取图像 {img_path}，跳过")
        continue

    # 获取该图像对应的所有标注
    anns = annotations_by_image.get(img_id, [])
    if not anns:
        print(f"图像 {file_name} 没有标注，跳过")
        continue

    # 为了区分多个标注，可以使用不同颜色的框（这里简单使用固定颜色，可扩展）
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255)]
    
    for idx, ann in enumerate(anns):
        color = colors[idx % len(colors)]  # 不同标注使用不同颜色

        # 绘制 bbox
        x, y, w, h = ann['bbox']
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

        # 绘制 square
        # x_s, y_s, w_s, h_s = ann['square']
        # cv2.rectangle(image, (int(x_s), int(y_s)), (int(x_s + w_s), int(y_s + h_s)), (0, 255, 255), 2)

        # 绘制关键点
        keypoints = ann['keypoints']
        for i in range(0, len(keypoints), 3):
            xx, yy, v = keypoints[i], keypoints[i+1], keypoints[i+2]
            if v > 0:  # 只绘制可见关键点
                cv2.circle(image, (int(xx), int(yy)), 2, (0, 255, 0), -1)

        # 绘制头部姿态轴（仅当存在欧拉角信息）
        if ann.get('has_euler_angles', False) and 'euler_angles' in ann:
            yaw = ann['euler_angles']['yaw']
            pitch = ann['euler_angles']['pitch']
            roll = ann['euler_angles']['roll']
            # 使用bbox中心作为轴起点，轴长度取bbox宽度的一半
            tdx = int(x + w / 2)
            tdy = int(y + h / 2)
            size = w // 2
            image = draw_axis(image, yaw, pitch, roll, tdx, tdy, size)
        else:
            print(f"图像 {file_name} 的标注 {idx} 缺少欧拉角信息")

    # 保存可视化结果
    output_path = os.path.join(output_dir, file_name)
    # 如果文件名可能重复，可以添加前缀或使用其他命名方式，这里直接使用原文件名
    cv2.imwrite(output_path, image)
    print(f"已保存: {output_path}")

print("所有图像可视化完成！")