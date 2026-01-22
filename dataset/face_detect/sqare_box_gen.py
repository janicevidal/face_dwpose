import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from nets.nn import FaceDetector
from nets.damofd import DAMOFD


def compute_iou_xyxy(bbox1, bbox2, extension=0):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.asarray(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.asarray(bbox2)

    bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2 = bbox1
    bbox2_x1, bbox2_y1, bbox2_x2, bbox2_y2 = bbox2

    s1 = (bbox1_x2 - bbox1_x1 + extension) * (bbox1_y2 - bbox1_y1 + extension)
    s2 = (bbox2_x2 - bbox2_x1 + extension) * (bbox2_y2 - bbox2_y1 + extension)

    xmin, ymin = max(bbox1_x1, bbox2_x1), max(bbox1_y1, bbox2_y1)
    xmax, ymax = min(bbox1_x2, bbox2_x2), min(bbox1_y2, bbox2_y2)

    inter_h = max(ymax - ymin + extension, 0)
    inter_w = max(xmax - xmin + extension, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    iou = intersection / union
    return iou


def compute_iou_xywh(bbox1, bbox2, extension=0):
    assert len(bbox1) == len(bbox2) == 4
    return compute_iou_xyxy(
        bbox1=np.array([bbox1[0], bbox1[1], bbox1[2] + bbox1[0] - 1, bbox1[3] + bbox1[1] - 1]),
        bbox2=np.array([bbox2[0], bbox2[1], bbox2[2] + bbox2[0] - 1, bbox2[3] + bbox2[1] - 1]),
        extension=extension
    )


def filter_max_iou_box(pre_facebox, boxes):
    max_iou = 0.0
    selected_box = None
    selected_idx = -1
    
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max, _ = box
        
        # 将当前框转换为xywh格式
        now_facebox = np.asarray([x_min, y_min, x_max - x_min + 1, y_max - y_min + 1], dtype=np.int32)
        
        # 计算IoU
        iou = compute_iou_xywh(pre_facebox, now_facebox)
        
        # 更新最大IoU和选中的框
        if iou > max_iou:
            max_iou = iou
            selected_box = now_facebox
            selected_idx = i
    
    # 如果没有找到合适的框（理论上不会，除非boxes为空）
    if selected_box is None and len(boxes) > 0:
        # 如果所有IoU都为0，选择第一个框作为后备
        x_min, y_min, x_max, y_max, _ = boxes[0]
        selected_box = np.asarray([x_min, y_min, x_max - x_min + 1, y_max - y_min + 1], dtype=np.int32)
        selected_idx = 0
        max_iou = 0.0
    
    return selected_box, max_iou, selected_idx


def filter_max_iou_box_with_threshold(pre_facebox, boxes, iou_threshold=0.5):
    # 先获取最大IoU的框
    selected_box, max_iou, selected_idx= filter_max_iou_box(pre_facebox, boxes)
    
    # 检查是否超过阈值
    is_valid = max_iou >= iou_threshold
    
    return selected_box, max_iou, selected_idx, is_valid


def generate_square_box(keypoints, detection_box, image_size):
    det_x1, det_y1, w, h = detection_box
    det_x2 = det_x1 + w -1
    det_y2 = det_y1 + h -1
    
    kp_x_min = np.min(keypoints[:, 0])
    kp_x_max = np.max(keypoints[:, 0])
    kp_y_min = np.min(keypoints[:, 1])
    kp_y_max = np.max(keypoints[:, 1])
        
    box_x1 = min(det_x1, kp_x_min)
    box_x2 = max(det_x2, kp_x_max)

    box_y2 = max(det_y2, kp_y_max)
    
    # case 1    
    # shift_ratio = 0.333
    # expand_ratio = 0.1
    # max_y_shift = height * 0.15
    # center_y = center_y + min(y_offset * 0.667, max_y_shift)
    
    # case 15    
    # shift_ratio = 0.2
    # expand_ratio = 0.115
    # max_y_shift = height * 0.15
    # center_y = center_y + min(y_offset * 0.667, max_y_shift)

    # case 20
    # shift_ratio = 0.115
    # expand_ratio = 0.2
    # max_y_shift = height * 0.15
    # center_y = center_y + min(y_offset * 0.667, max_y_shift)
    
    # case 20_d
    shift_ratio = 0.05
    expand_ratio = 0.2
    
    box_y1 = det_y1 + (kp_y_min - det_y1) * (1 - shift_ratio)
        
    width = box_x2 - box_x1
    height = box_y2 - box_y1
    target_size = max(width, height)

    center_x = (box_x1 + box_x2) / 2
    center_y = (box_y1 + box_y2) / 2
    
    keypoint_centroid = 0.5 * (kp_y_max + kp_y_min)
    
    y_offset = keypoint_centroid - center_y
    
    max_y_shift = height * 0.3
    if y_offset > 0:
        center_y = center_y + min(y_offset * 0.8, max_y_shift)
    else:
        center_y = center_y
        
    # 外扩    
    size = int(target_size * (1 + expand_ratio))
        
    x = int(center_x - size / 2)
    y = int(center_y - size / 2)
        
    if image_size is not None:
        img_w, img_h = image_size
        x = max(0, x)
        y = max(0, y)
        
        if x + size > img_w:
            if size <= img_w:
                x = img_w - size
            else:
                size = img_w
                x = 0
        
        if y + size > img_h:
            if size <= img_h:
                y = img_h - size
            else:
                size = min(size, img_h)
                y = 0
    
    return x, y, max(size, 10)


def normalize_landmarks_by_squarebox(landmarks, squarebox):
    """
    根据squarebox归一化关键点
    Args:
        landmarks: 原始关键点坐标，shape=(N, 2)
        squarebox: squarebox坐标，格式为[x_min, y_min, size, size]
    Returns:
        归一化后的关键点坐标，shape=(N, 2)
    """
    x_min, y_min, size = squarebox[0], squarebox[1], squarebox[2]
    
    # 归一化到 [0, 1] 范围
    normalized_landmarks = np.zeros_like(landmarks, dtype=np.float32)
    normalized_landmarks[:, 0] = (landmarks[:, 0] - x_min) / size
    normalized_landmarks[:, 1] = (landmarks[:, 1] - y_min) / size
    
    return normalized_landmarks


def draw_mean_landmarks_on_blank_image(mean_landmarks, output_path, image_size=96):
    """
    在空白图像上绘制平均关键点
    Args:
        mean_landmarks: 平均关键点坐标，shape=(N, 2)，坐标范围[0, 1]
        output_path: 输出图像路径
        image_size: 输出图像大小
    """
    # 创建空白图像
    blank_image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    
    # 将归一化坐标转换为像素坐标
    pixel_landmarks = mean_landmarks * image_size
    
    # 绘制关键点
    for i, (x, y) in enumerate(pixel_landmarks):
        # 绘制点
        cv2.circle(blank_image, (int(x), int(y)), 1, (0, 0, 0), -1)
        
        # 可选：添加序号标签
        # cv2.putText(blank_image, str(i), (int(x)+2, int(y)), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    # 保存图像
    cv2.imwrite(output_path, blank_image)
    print(f"平均关键点图像已保存到: {output_path}")


def main():
    input_dir = "/data/xiaoshuai/facial_lanmark/val_1118/"
    output_dir = "/data/xiaoshuai/facial_lanmark/val_1118_npy"    
    error_dir = "/data/xiaoshuai/facial_lanmark/face_detect/val_1118_det_error"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    model_path = "weights/SCRFD_10G.onnx" 
    detector_scrfd = FaceDetector(onnx_file=model_path)
    detector_scrfd.prepare(-1)
    
    model_path = "weights/DamoFD_10g_shape640x640.onnx"
    detector = DAMOFD(model_file=model_path)
    detector.prepare(-1)
    
    image_names = sorted(glob.glob(f"{input_dir}/*.png") + glob.glob(f"{input_dir}/*.jpg") + glob.glob(f"{input_dir}/*.jpeg") + glob.glob(f"{input_dir}/*.JPG"))
    
    print(f"There are {len(image_names)} images.")
    
    # 用于存储所有归一化后的关键点
    all_normalized_landmarks = []
    valid_count = 0
    
    for image_name in tqdm(image_names): 
        input_image = cv2.imread(image_name)
        
        if input_image.shape[0] == input_image.shape[1] == 512 or input_image.shape[0] == input_image.shape[1] == 1024:
            boxes, _ = detector.detect_small(input_image, input_size=(640, 640))
            if len(boxes) == 0:
                boxes, _ = detector.detect(input_image, input_size=(640, 640))
        else:
            boxes, _ = detector.detect(input_image, input_size=(640, 640))
            if len(boxes) == 0:
                boxes, _ = detector.detect_small(input_image, input_size=(640, 640))
        
        if len(boxes) == 0:
            if input_image.shape[0] == input_image.shape[1] == 512 or input_image.shape[0] == input_image.shape[1] == 1024:
                boxes, _ = detector_scrfd.detect(input_image, input_size=(512, 512))
                if len(boxes) == 0:
                    boxes, _ = detector_scrfd.detect(input_image, input_size=(640, 640))
            else:
                boxes, _ = detector_scrfd.detect(input_image, input_size=(640, 640))
                if len(boxes) == 0:
                    boxes, _ = detector_scrfd.detect(input_image, input_size=(512, 512))
                        
        if len(boxes) == 0:                
            nimg = np.copy(input_image)
            cv2.imwrite(filename=os.path.join(error_dir, os.path.basename(image_name)), img=nimg)
            print(os.path.join(error_dir, os.path.basename(image_name)))
        else:
            npy_name = image_name.split('.')[0] + '.npy'
            
            infodict = np.load(npy_name, allow_pickle=True).item()
            
            pre_faceboxes = infodict["DFSD_facebbox"]
            if pre_faceboxes.shape[0] != 1:
                print(pre_faceboxes, len(pre_faceboxes), npy_name)
                continue
            
            pre_facebox = pre_faceboxes[0]
    
            selected_box, max_iou, selected_idx, is_valid = filter_max_iou_box_with_threshold(pre_facebox, boxes, iou_threshold=0.3)
            
            if not is_valid:
                if input_image.shape[0] == input_image.shape[1] == 512 or input_image.shape[0] == input_image.shape[1] == 1024:
                    boxes, _ = detector_scrfd.detect(input_image, input_size=(512, 512))
                    if len(boxes) == 0:
                        boxes, _ = detector_scrfd.detect(input_image, input_size=(640, 640))
                else:
                    boxes, _ = detector_scrfd.detect(input_image, input_size=(640, 640))
                    if len(boxes) == 0:
                        boxes, _ = detector_scrfd.detect(input_image, input_size=(512, 512))
                    
                if len(boxes) == 0:
                    is_valid = False
                else:
                    selected_box, max_iou, selected_idx, is_valid = filter_max_iou_box_with_threshold(pre_facebox, boxes, iou_threshold=0.3)
                
            if not is_valid:
                nimg = np.copy(input_image)
                
                cv2.rectangle(nimg, pt1=(pre_facebox[0], pre_facebox[1]), pt2=(pre_facebox[0]+pre_facebox[2]-1, pre_facebox[1]+pre_facebox[3]-1), color=(0, 255, 255), thickness=2)
                
                for box in boxes:
                    x_min, y_min, x_max, y_max, _ = box
                    cv2.rectangle(nimg, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 255), 1)

                    cv2.line(nimg, (int(x_min), int(y_min)), (int(x_min + 15), int(y_min)), (255, 0, 255), 3)
                    cv2.line(nimg, (int(x_min), int(y_min)), (int(x_min), int(y_min + 15)), (255, 0, 255), 3)

                    cv2.line(nimg, (int(x_max), int(y_max)), (int(x_max - 15), int(y_max)), (255, 0, 255), 3)
                    cv2.line(nimg, (int(x_max), int(y_max)), (int(x_max), int(y_max - 15)), (255, 0, 255), 3)

                    cv2.line(nimg, (int(x_max - 15), int(y_min)), (int(x_max), int(y_min)), (255, 0, 255), 3)
                    cv2.line(nimg, (int(x_max), int(y_min)), (int(x_max), int(y_min + 15)), (255, 0, 255), 3)

                    cv2.line(nimg, (int(x_min), int(y_max - 15)), (int(x_min), int(y_max)), (255, 0, 255), 3)
                    cv2.line(nimg, (int(x_min), int(y_max)), (int(x_min + 15), int(y_max)), (255, 0, 255), 3)

                cv2.imwrite(filename=os.path.join(error_dir, os.path.basename(image_name)), img=nimg)
            
            else:
                infodict["DFSD_facebbox"] = [selected_box]
                
                # 生成squarebox
                landmarks208 = infodict["landmarks208"]
                x_min, y_min, size = generate_square_box(landmarks208, selected_box, (input_image.shape[1], input_image.shape[0]))
                squarebox = np.asarray([x_min, y_min, size, size], dtype=np.int32)
                
                # 归一化关键点
                normalized_landmarks = normalize_landmarks_by_squarebox(landmarks208, squarebox)
                all_normalized_landmarks.append(normalized_landmarks)
                valid_count += 1
                
                # 保存squarebox到infodict
                infodict["squarebbox"] = [squarebox]
                np.save(file=os.path.join(output_dir, os.path.basename(npy_name)), arr=infodict)
    
    # 计算所有样本下的点位均值
    if valid_count > 0:
        print(f"成功处理 {valid_count} 个样本")
        
        # 转换为numpy数组
        all_normalized_landmarks = np.array(all_normalized_landmarks)
        
        # 计算均值
        mean_landmarks = np.mean(all_normalized_landmarks, axis=0)
        
        # 保存均值关键点
        mean_landmarks_path = os.path.join(output_dir, "mean_landmarks_0.npy")
        np.save(mean_landmarks_path, mean_landmarks)
        print(f"平均关键点已保存到: {mean_landmarks_path}")
        
        # 在96x96空白图像上绘制平均关键点
        output_image_path = os.path.join(output_dir, "mean_landmarks_on_96x96_0.png")
        draw_mean_landmarks_on_blank_image(mean_landmarks, output_image_path, image_size=96)
        
        # 可选：打印一些统计信息
        print(f"平均关键点范围: x=[{np.min(mean_landmarks[:, 0]):.4f}, {np.max(mean_landmarks[:, 0]):.4f}], "
              f"y=[{np.min(mean_landmarks[:, 1]):.4f}, {np.max(mean_landmarks[:, 1]):.4f}]")
    else:
        print("没有成功处理的样本，无法计算平均关键点")


if __name__ == '__main__':
    main()