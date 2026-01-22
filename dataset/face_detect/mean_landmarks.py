import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

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
    x_min, y_min, size = squarebox[0], squarebox[1], squarebox[2]

    normalized_landmarks = np.zeros_like(landmarks, dtype=np.float32)
    normalized_landmarks[:, 0] = (landmarks[:, 0] - x_min) / size
    normalized_landmarks[:, 1] = (landmarks[:, 1] - y_min) / size
    
    return normalized_landmarks

def draw_mean_landmarks_on_blank_image(mean_landmarks, output_path, image_size=96):
    blank_image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    
    pixel_landmarks = mean_landmarks * image_size
    
    for i, (x, y) in enumerate(pixel_landmarks):
        cv2.circle(blank_image, (int(x+0.5), int(y+0.5)), 1, (0, 0, 0), 0)

        # cv2.putText(blank_image, str(i), (int(x)+2, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        if i == 141 or i == 159 or i == 201 or i == 202:
            print(i, x, y)
    
    cv2.imwrite(output_path, blank_image)
    print(f"平均关键点图像已保存到: {output_path}")
    
    
def main():    
    # input_path = "/data/xiaoshuai/facial_lanmark/val_1118/" 
    # npy_path = "/data/xiaoshuai/facial_lanmark/val_1118_npy" 
    # output_path = '/data/xiaoshuai/facial_lanmark/val_1118_npy_mean'
    
    # input_path = "/data/xiaoshuai/facial_lanmark/train_1121/" 
    # npy_path = "/data/xiaoshuai/facial_lanmark/train_1121_box_correct" 
    # output_path = '/data/xiaoshuai/facial_lanmark/train_1121_box_correct_square'
    
    # input_path = "/data/xiaoshuai/facial_lanmark/datasets/lpff/images_all_1226/" 
    # npy_path = "/data/xiaoshuai/facial_lanmark/datasets/lpff/train_1226_box_correct/" 
    # output_path = '/data/xiaoshuai/facial_lanmark/datasets/lpff/train_1226_box_correct_square'
    # mean_path = '/data/xiaoshuai/facial_lanmark/datasets/lpff/mean_face_15'
    
    # input_path = "/data/xiaoshuai/facial_lanmark/train_1226/images" 
    # npy_path = "/data/xiaoshuai/facial_lanmark/train_1226/box_correct" 
    # output_path = '/data/xiaoshuai/facial_lanmark/train_1226/box_correct_square_15'
    # mean_path = '/data/xiaoshuai/facial_lanmark/train_1226/mean_face_15'
    
    # input_path = "/data/xiaoshuai/facial_lanmark/train_1226/images" 
    # npy_path = "/data/xiaoshuai/facial_lanmark/train_1226/box_correct" 
    # output_path = '/data/xiaoshuai/facial_lanmark/train_1226/box_correct_square_20'
    # mean_path = '/data/xiaoshuai/facial_lanmark/train_1226/mean_face_20'
    
    # input_path = "/data/xiaoshuai/facial_lanmark/train_1226/images" 
    # npy_path = "/data/xiaoshuai/facial_lanmark/train_1226/box_correct" 
    # output_path = '/data/xiaoshuai/facial_lanmark/train_1226/box_correct_square_20_d'
    # mean_path = '/data/xiaoshuai/facial_lanmark/train_1226/mean_face_20_d'
    
    # input_path = "/data/xiaoshuai/facial_lanmark/datasets/lpff/val_1229/images_all" 
    # npy_path = "/data/xiaoshuai/facial_lanmark/datasets/lpff/val_1229/box_correct/" 
    # output_path = '/data/xiaoshuai/facial_lanmark/datasets/lpff/val_1229/box_correct_square_20_d'
    # mean_path = '/data/xiaoshuai/facial_lanmark/datasets/lpff/val_1229/mean_face_20_d'
    
    input_path = "/data/xiaoshuai/facial_lanmark/train_1226/val_1118/" 
    npy_path = "/data/xiaoshuai/facial_lanmark/train_1226/val_1118_box_correct/" 
    output_path = '/data/xiaoshuai/facial_lanmark/train_1226/val_1118_box_correct_square_20_d'
    mean_path = '/data/xiaoshuai/facial_lanmark/train_1226/val_1118_mean_face_20_d'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if not os.path.exists(mean_path):
        os.makedirs(mean_path)
    
    if input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): 
        # input single img path
        input_img_list = [input_path]
    else: 
        # input img folder
        if input_path.endswith('/'):  # solve when path ends with /
            input_path = input_path[:-1]
        
        # scan all the jpg and png images
        input_img_list = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))
    
    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError('No input image is found...\n')

    all_normalized_landmarks = []

    for i, img_path in tqdm(enumerate(input_img_list)):
        img_name = os.path.basename(img_path)
        # print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
        
        npy_name = os.path.join(npy_path, img_name.split(".")[0] + ".npy")
        
        image = cv2.imread(img_path)

        if not os.path.exists(npy_name):
            import pdb
            pdb.set_trace()
        
        infodict = np.load(npy_name, allow_pickle=True).tolist()
        
        # print(infodict.keys())
        # print(infodict["visibility"])
        # print(infodict["landmark_verified"])
        # print(infodict["DFSD_facebbox"])
        # print(infodict["landmarks208"])
        # import pdb
        # pdb.set_trace()
        
        landmarks235 = infodict["landmarks208"]

        bboxes = infodict["DFSD_facebbox"]
        gtbox = bboxes[0]
        
        # if "squarebbox" in infodict.keys():
        #     squarebox = infodict["squarebbox"][0]
        # else:
        #     x_min, y_min, size = generate_square_box(landmarks235, gtbox, (image.shape[1], image.shape[0]))
        
        #     squarebox = np.asarray([x_min, y_min, size, size], dtype=np.int32)
            
        #     infodict["squarebbox"] = [squarebox]
        #     np.save(file=os.path.join(output_path, os.path.basename(npy_name)), arr=infodict)

        x_min, y_min, size = generate_square_box(landmarks235, gtbox, (image.shape[1], image.shape[0]))
        
        squarebox = np.asarray([x_min, y_min, size, size], dtype=np.int32)

        infodict["squarebbox"] = [squarebox]
        np.save(file=os.path.join(output_path, os.path.basename(npy_name)), arr=infodict)
                
        # 归一化关键点
        normalized_landmarks = normalize_landmarks_by_squarebox(landmarks235, squarebox)
        all_normalized_landmarks.append(normalized_landmarks)
                
        
    all_normalized_landmarks = np.array(all_normalized_landmarks)
        
    mean_landmarks = np.mean(all_normalized_landmarks, axis=0)

    mean_landmarks_path = os.path.join(mean_path, "mean_landmarks.npy")
    np.save(mean_landmarks_path, mean_landmarks)
    print(f"平均关键点已保存到: {mean_landmarks_path}")
        
    output_image_path = os.path.join(mean_path, "mean_landmarks_on_96x96.png")
    draw_mean_landmarks_on_blank_image(mean_landmarks, output_image_path, image_size=96)


if __name__ == '__main__':
    main()