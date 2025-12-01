import json
import cv2
import numpy as np
import os
from math import cos, sin

image_dir = "/data/xiaoshuai/facial_lanmark/val_1118/" 

# 加载转换后的COCO格式文件
with open('/data/xiaoshuai/facial_lanmark/val_1118/annotations/val_angles_annotations.json', 'r') as f:
    coco_data = json.load(f)

def draw_axis(img, yaw_, pitch_, roll_, tdx=None, tdy=None, size = 100):

    pitch = -(pitch_ * np.pi / 180)
    yaw = yaw_ * np.pi / 180
    roll = -(roll_ * np.pi / 180)
    
    yaw_value = yaw_
    pitch_value = pitch_
    roll_value = roll_

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    pitch_color = (255,255,0)
    yaw_color   = (0,255,0)
    roll_color  = (0,0,255)

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)
    
    cv2.putText(img, "Pitch:{:.2f}".format(pitch_value), (0,10), cv2.FONT_HERSHEY_PLAIN, 1, pitch_color)
    cv2.putText(img, "Yaw:{:.2f}".format(yaw_value), (0,30), cv2.FONT_HERSHEY_PLAIN, 1, yaw_color)
    cv2.putText(img, "Roll:{:.2f}".format(roll_value), (0,50), cv2.FONT_HERSHEY_PLAIN, 1, roll_color)

    return img

# 检查每个图像的标注数量
image_annotations = {}
for ann in coco_data['annotations']:
    image_id = ann['image_id']
    if image_id not in image_annotations:
        image_annotations[image_id] = 0
    image_annotations[image_id] += 1

# 打印每个图像的标注数量
print("Annotations per image:")
for img_id, count in image_annotations.items():
    print(f"Image {img_id}: {count} annotation(s)")

# 检查是否有图像有多个标注
multi_annotation_images = [img_id for img_id, count in image_annotations.items() if count > 1]
if multi_annotation_images:
    print(f"Warning: {len(multi_annotation_images)} images have multiple annotations")
else:
    print("All images have exactly one annotation")

# 选择一个样本进行可视化
img_info = coco_data['images'][10]  # 选择第10张图像
annotation = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_info['id']][0]

print(img_info['file_name'])
# 读取图像
image = cv2.imread(os.path.join(image_dir, img_info['file_name']))

# 绘制bbox
x, y, w, h = annotation['bbox']
cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)

# 绘制关键点
keypoints = annotation['keypoints']
for i in range(0, len(keypoints), 3):
    xx, yy, v = keypoints[i], keypoints[i+1], keypoints[i+2]
    if v > 0:  # 只绘制可见的关键点
        cv2.circle(image, (int(xx), int(yy)), 2, (0, 255, 0), -1)

# 绘制头部姿态轴
yaw = annotation['euler_angles']['yaw']
pitch = annotation['euler_angles']['pitch']
roll = annotation['euler_angles']['roll']
image = draw_axis(image, yaw, pitch, roll, tdx=int(x + w/2), tdy=int(y + h/2), size=w//2)

# 保存或显示结果
cv2.imwrite('validation_result.jpg', image)
print("Validation image saved as validation_result.jpg")