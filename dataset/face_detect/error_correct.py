import cv2
import os
import glob
import numpy as np

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
    
    
def main():
    input_dir = "/data/xiaoshuai/facial_lanmark/face_detect/train_1121_det_error/"
    npy_dir = "/data/xiaoshuai/facial_lanmark/train_1121/"
    output_dir = "/data/xiaoshuai/facial_lanmark/train_1121_box_correct"    
    error_dir = "/data/xiaoshuai/facial_lanmark/face_detect/train_1121_det_error_"
    
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
    
    for image_name in tqdm(image_names): 
        # print(image_name)
        # npy_name = image_name.split('.')[0] + '.npy'
        npy_name = os.path.join(npy_dir, os.path.basename(image_name).split('.')[0] + '.npy')
        
        savef = os.path.join(output_dir, os.path.basename(npy_name))
        if os.path.exists(savef):
            print(f"{savef} exists, Skip...")
            continue
    
        input_image = cv2.imread(image_name)
        
        if input_image.shape[0] == input_image.shape[1] == 512 or input_image.shape[0] == input_image.shape[1] == 1024:
            boxes, _ = detector.detect_small(input_image, input_size=(640, 640))
            if len(boxes) == 0:
                boxes, _ = detector.detect(input_image, input_size=(640, 640))
                # boxes, _ = detector.detect(input_image, input_size=(640, 640), max_num=1, metric='default')
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
            
            infodict = np.load(npy_name, allow_pickle=True).item()
            
            pre_faceboxes = infodict["DFSD_facebbox"]
            
            if pre_faceboxes.shape[0] != 1:
                print(pre_faceboxes, len(pre_faceboxes), npy_name)
                nimg = np.copy(input_image)
                cv2.imwrite(filename=os.path.join(error_dir, os.path.basename(image_name)), img=nimg)
            
            pre_facebox = pre_faceboxes[0]  # xywh格式
    
            selected_box, max_iou, selected_idx, is_valid = filter_max_iou_box_with_threshold(pre_facebox, boxes, iou_threshold=0.0)
            # print(max_iou)
            
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
                cv2.imwrite(filename=os.path.join(error_dir, os.path.basename(image_name)), img=nimg)
                
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

                cv2.imwrite(filename=os.path.join(error_dir, "vis_" + os.path.basename(image_name)), img=nimg)
            
            else:
                infodict["DFSD_facebbox"] = [selected_box]
                
                np.save(file=savef, arr=infodict)


if __name__ == '__main__':
    main()