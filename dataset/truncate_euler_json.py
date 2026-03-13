import json
import os

def filter_coco_annotations(input_json_path, output_json_path, angle_threshold=30, filename_pattern="lpff"):
    """
    根据角度条件和文件名条件筛选COCO标注，并保存为新JSON文件。

    参数:
        input_json_path (str): 输入的COCO格式JSON文件路径。
        output_json_path (str): 输出的筛选后JSON文件路径。
        angle_threshold (float): 角度阈值，yaw或pitch大于该值则保留（默认30）。
        filename_pattern (str): 图片文件名需包含的字符串（默认"lpff"）。
    """
    # 读取原始JSON文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 提取原始列表
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    # 其他字段原样保留
    other_fields = {k: v for k, v in coco_data.items() if k not in ['images', 'annotations']}

    # 构建 image_id -> file_name 映射
    image_id_to_filename = {img['id']: img['file_name'] for img in images}

    # 用于收集符合条件的 annotation 和对应的 image_id
    kept_annotations = []
    kept_image_ids = set()

    val_path = '/data/xiaoshuai/facial_landmark_181/val'
    
    for ann in annotations:
        # 获取图片文件名
        img_id = ann['image_id']
        filename = image_id_to_filename.get(img_id, '')

        # 检查文件名条件
        # filename_condition1 = filename_pattern in filename
        # filename_condition2 = "auglpff" not in filename
        
        # filename_condition = filename_condition1 and filename_condition2
        
        if "auglpff" in filename:
            continue
        
        # 标定有误的图像
        if "lpff_19250" in filename or "lpff_17627" in filename or "lpff_18161" in filename:
            continue
        
        # 去除验证集内的图像
        file_path = os.path.join(val_path, filename.split('.')[0] + ".npy")
        if os.path.isfile(file_path):
            print(file_path)
            continue
        
        filename_condition = ("lpff" in filename) and ("auglpff" not in filename)

        # 检查角度条件（确保有 euler_angles 字段且 has_euler_angles 为 True）
        angle_condition = False
        if ann.get('has_euler_angles', False) and 'euler_angles' in ann:
            euler = ann['euler_angles']
            yaw = euler.get('yaw', 0)
            pitch = euler.get('pitch', 0)

            # if abs(yaw) > angle_threshold or abs(pitch) > angle_threshold:
            if abs(yaw) > angle_threshold or abs(pitch) > angle_threshold * 0.6:
                angle_condition = True

        # 满足任一条件则保留
        if filename_condition or angle_condition:
            # 非 align 模型，删除原标注中的 square 属性（若存在）
            ann.pop('square', None)
            
            kept_annotations.append(ann)
            kept_image_ids.add(img_id)

    # 筛选图片
    kept_images = [img for img in images if img['id'] in kept_image_ids]

    # 构建新的COCO字典
    new_coco = {
        'images': kept_images,
        'annotations': kept_annotations,
        **other_fields   # 保留其他字段（如 categories, info 等）
    }

    # 写入新文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_coco, f, indent=4, ensure_ascii=False)

    print(f"筛选完成！原始标注数：{len(annotations)}，保留标注数：{len(kept_annotations)}")
    print(f"原始图片数：{len(images)}，保留图片数：{len(kept_images)}")
    print(f"结果已保存至：{output_json_path}")

if __name__ == "__main__":

    # input_file = "/data/xiaoshuai/facial_lanmark/train_0126/annotations/train_angles_annotations.json"
    # output_file = "/data/xiaoshuai/facial_lanmark/train_0126/annotations/train_filtered_annotations.json"
    # filter_coco_annotations(input_file, output_file)
    # input_file = "/data/xiaoshuai/facial_lanmark/train_0126/annotations/val_angles_annotations.json"
    # output_file = "/data/xiaoshuai/facial_lanmark/train_0126/annotations/val_filtered_annotations.json"
    # filter_coco_annotations(input_file, output_file, angle_threshold=20)
    
    # input_file = "/data/xiaoshuai/facial_lanmark/train_0126/annotations/val_angles_annotations.json"
    # output_file = "/data/xiaoshuai/facial_lanmark/train_0126/annotations/val_filtered_annotations_refine.json"
    # filter_coco_annotations(input_file, output_file, angle_threshold=20)
    input_file = "/data/xiaoshuai/facial_lanmark/train_0126/annotations/train_angles_annotations.json"
    output_file = "/data/xiaoshuai/facial_lanmark/train_0126/annotations/train_filtered_annotations_refine.json"
    filter_coco_annotations(input_file, output_file)