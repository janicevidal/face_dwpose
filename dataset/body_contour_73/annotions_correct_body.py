import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
from rtmlib import YOLOX

def expand_bbox_xyxy(bbox, scale=1.1, img_w=None, img_h=None):
    """将 xyxy 格式的 bbox 外扩 scale 倍，并边界截断"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    new_w = w * scale
    new_h = h * scale
    new_x1 = cx - new_w / 2.0
    new_x2 = cx + new_w / 2.0
    new_y1 = cy - new_h / 2.0
    new_y2 = cy + new_h / 2.0

    x1_new = int(np.floor(new_x1))
    y1_new = int(np.floor(new_y1))
    x2_new = int(np.ceil(new_x2))
    y2_new = int(np.ceil(new_y2))

    if img_w is not None and img_h is not None:
        x1_new = max(0, min(x1_new, img_w - 1))
        y1_new = max(0, min(y1_new, img_h - 1))
        x2_new = max(0, min(x2_new, img_w - 1))
        y2_new = max(0, min(y2_new, img_h - 1))

    if x2_new <= x1_new:
        x2_new = x1_new + 1
    if y2_new <= y1_new:
        y2_new = y1_new + 1
    return (x1_new, y1_new, x2_new, y2_new)

def xywh_to_xyxy(bbox_xywh):
    x, y, w, h = bbox_xywh
    return (x, y, x + w - 1, y + h - 1)

def xyxy_to_xywh(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

def compute_iou_xyxy(bbox1, bbox2):
    """计算两个 xyxy 框的 IoU"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def select_max_iou_box(bbox_ref, det_boxes):
    """从检测框列表中选择与参考框 IoU 最大的框，返回 (box_xyxy, iou)"""
    best_box = None
    best_iou = -1.0
    for box in det_boxes:
        # 假设 det_boxes 格式为 [x1, y1, x2, y2, score] 或 [x1, y1, x2, y2]
        if len(box) >= 4:
            box_xyxy = box[:4]
        else:
            continue
        iou = compute_iou_xyxy(bbox_ref, box_xyxy)
        if iou > best_iou:
            best_iou = iou
            best_box = box_xyxy
    return best_box, best_iou

def main():
    debug = False
    # debug = True
    
    input_dir = "/data/xiaoshuai/body_keypoint/train0609/_images/"
    anno_dir = "/data/xiaoshuai/body_keypoint/train0609/_annos/"
    output_dir = "/data/xiaoshuai/body_keypoint/train0609/_box_correct_annos"
    # output_dir = "/data/xiaoshuai/body_keypoint/train0609/_box_correct_annos_debug"
    
    os.makedirs(output_dir, exist_ok=True)
    if debug:
        debug_dir = "/data/xiaoshuai/body_keypoint/train0609/_det_debug_vis"
        os.makedirs(debug_dir, exist_ok=True)

    # 初始化人体检测模型
    device = 'cuda'
    backend = 'onnxruntime'
    human_detector = YOLOX(onnx_model='/data/xiaoshuai/facial_lanmark/face_detect/weights/yolox_l.onnx',
                           backend=backend, device=device)
    # human_detector = YOLOX(onnx_model='/data/xiaoshuai/facial_lanmark/face_detect/weights/yolox_l.onnx',
    #                        backend=backend, device=device, score_thr=0.5)

    image_names = sorted(glob.glob(f"{input_dir}/*.png") + glob.glob(f"{input_dir}/*.jpg") +
                         glob.glob(f"{input_dir}/*.jpeg") + glob.glob(f"{input_dir}/*.JPG"))

    print(f"There are {len(image_names)} images.")

    for image_name in tqdm(image_names):
        npy_name = os.path.join(anno_dir, os.path.basename(image_name).split('.')[0] + '.npy')
        savef = os.path.join(output_dir, os.path.basename(npy_name))
        if os.path.exists(savef):
            print(f"{savef} exists, Skip...")
            continue

        input_image = cv2.imread(image_name)
        if input_image is None:
            print(f"Failed to read image: {image_name}")
            continue

        h_img, w_img = input_image.shape[:2]

        try:
            infodict = np.load(npy_name, allow_pickle=True).item()
        except Exception as e:
            print(f"Failed to load {npy_name}: {e}")
            continue

        # 必要字段检查
        if "keypoints" not in infodict or "visibility" not in infodict or "bbox" not in infodict:
            print(f"Missing keypoints/visibility/bbox in {npy_name}, skip.")
            continue

        keypoints = infodict["keypoints"]  # shape (N, 2) 或 (N, 3)
        visibility = np.asarray(infodict["visibility"])
        original_bbox_xywh = infodict["bbox"][0]  # xywh
        # print(infodict["bbox"])
        
        original_bbox_xyxy = xywh_to_xyxy(original_bbox_xywh)

        # 确保 keypoints 为 Nx2
        if keypoints.ndim == 2 and keypoints.shape[1] >= 2:
            pts = keypoints[:, :2].copy().astype(np.int32)
        else:
            print(f"Unexpected keypoints shape {keypoints.shape}")
            continue

        # 可见性掩码
        visible_mask = (visibility != 0)
        visible_pts = pts[visible_mask]

        # 如果没有可见点，仅将不可见点坐标置0，不更新 bbox
        if len(visible_pts) == 0:
            print(f"No visible keypoints in {npy_name}, only zeroing invisible points.")
            pts[~visible_mask] = [0, 0]
            if keypoints.shape[1] > 2:
                infodict["keypoints"][:, :2] = pts
            else:
                infodict["keypoints"] = pts
            np.save(savef, infodict)
            continue
        
        # 1. 可见点外接矩形 (xyxy)
        x_min = int(np.min(visible_pts[:, 0]))
        y_min = int(np.min(visible_pts[:, 1]))
        x_max = int(np.max(visible_pts[:, 0]))
        y_max = int(np.max(visible_pts[:, 1]))
        if x_max <= x_min: x_max = x_min + 1
        if y_max <= y_min: y_max = y_min + 1
        contour_bbox_xyxy = (x_min, y_min, x_max, y_max)
        
        original_x1, original_y1, original_x2, original_y2 = original_bbox_xyxy
        points_out_x = 0
        for (x, y) in visible_pts:
            if x < original_x1 or x > original_x2:
                points_out_x += 1
        ratio_out_x = points_out_x / len(visible_pts)
        need_redetect = (ratio_out_x > 0.3)  # 如果超过30%可见点在原始 bbox 左右外侧，触发重检测
        
        if need_redetect:
            print(f"Redetecting for {os.path.basename(image_name)}")
            # 使用人体检测器
            detections = human_detector(np.array(input_image))
            if len(detections) > 0:
                # 选取与原始外接矩形 (contour_bbox_xyxy) IoU 最大的检测框
                best_det_box, best_iou = select_max_iou_box(contour_bbox_xyxy, detections)
                print(f"  Best det box: {best_det_box}, IoU with contour box: {best_iou:.4f}")
                print(f"  Original bbox: {original_bbox_xyxy}")
                original_bbox_xyxy = (round(best_det_box[0]), round(best_det_box[1]), round(best_det_box[2]), round(best_det_box[3])) if best_det_box is not None else original_bbox_xyxy
            else:
                print(f"  No detection found, keep merged box.")
                
        # 2. 外扩 1.1 倍
        expanded_bbox_xyxy = expand_bbox_xyxy(contour_bbox_xyxy, scale=1.05, img_w=w_img, img_h=h_img)

        # 3. 初始合并框 (expanded 与 original 的并集)
        merged_x1 = min(expanded_bbox_xyxy[0], original_bbox_xyxy[0])
        merged_y1 = min(expanded_bbox_xyxy[1], original_bbox_xyxy[1])
        merged_x2 = max(expanded_bbox_xyxy[2], original_bbox_xyxy[2])
        merged_y2 = max(expanded_bbox_xyxy[3], original_bbox_xyxy[3])
        merged_bbox_xyxy = (merged_x1, merged_y1, merged_x2, merged_y2)

        # 5. 判断可见点是否超过一半不在 merged_bbox 内
        # points_in = 0
        # for (x, y) in visible_pts:
        #     if merged_x1 <= x <= merged_x2 and merged_y1 <= y <= merged_y2:
        #         points_in += 1
        # ratio_out = 1.0 - points_in / len(visible_pts)
        # need_redetect = (ratio_out > 0.5)

        final_bbox_xyxy = merged_bbox_xyxy  # 默认

        # 转换为 xywh 并更新 bbox
        final_bbox_xywh = xyxy_to_xywh(final_bbox_xyxy)
        infodict["bbox"] = [np.asarray(final_bbox_xywh, dtype=np.int32)]
        # infodict["bbox"] = [list(final_bbox_xywh)]
        # print(infodict["bbox"])

        # 不可见点坐标置 0
        pts[~visible_mask] = [0, 0]
        if keypoints.shape[1] > 2:
            infodict["keypoints"][:, :2] = pts
        else:
            infodict["keypoints"] = pts

        # 保存更新后的 npy
        np.save(savef, infodict)

        if debug:
            # 可选：生成调试可视化
            debug_img = input_image.copy()
            # 原始 bbox (绿色)
            x1_o, y1_o, x2_o, y2_o = original_bbox_xyxy
            cv2.rectangle(debug_img, (x1_o, y1_o), (x2_o, y2_o), (0, 255, 0), 2)
            # 外扩后 bbox (蓝色)
            x1_e, y1_e, x2_e, y2_e = expanded_bbox_xyxy
            cv2.rectangle(debug_img, (x1_e, y1_e), (x2_e, y2_e), (255, 0, 0), 2)
            # 最终 bbox (红色)
            x1_f, y1_f, x2_f, y2_f = final_bbox_xyxy
            cv2.rectangle(debug_img, (x1_f, y1_f), (x2_f, y2_f), (0, 0, 255), 2)
            # 可见关键点 (黄色)
            for (x, y) in visible_pts:
                cv2.circle(debug_img, (int(x), int(y)), 2, (0, 255, 255), -1)
            # 如果触发重检测，在图像上写文字
            if need_redetect:
                cv2.putText(debug_img, "Re-detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(debug_dir, "debug_" + os.path.basename(image_name)), debug_img)

if __name__ == '__main__':
    main()