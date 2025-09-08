import json
import cv2
import numpy as np
import os

image_dir = "/data/caiachang/video-ldms-ok/TEST/only1face/" 

# 加载转换后的COCO格式文件
with open('coco_face_annotations.json', 'r') as f:
    coco_data = json.load(f)

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
img_info = coco_data['images'][0]
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
    x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
    if v > 0:  # 只绘制可见的关键点
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)

# 保存或显示结果
cv2.imwrite('validation_result.jpg', image)
print("Validation image saved as validation_result.jpg")