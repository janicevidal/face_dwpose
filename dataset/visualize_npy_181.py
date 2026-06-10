import os
import cv2
import glob
import numpy as np
import shutil

def main():    
    
    # input_path =  "/data/xiaoshuai/facial_landmark_181/train_0310/val/"
    # input_path = "/data/xiaoshuai/facial_lanmark/train_1226/val_1229/"
    # npy_path = '/data/xiaoshuai/facial_landmark_181/train_0326/val/'
    # output_path = '/data/xiaoshuai/facial_landmark_181/train_0326/val_visualize/'
    
    # dst_path = "/data/xiaoshuai/facial_landmark_181/train_0326/val_images/"
    
    # input_path = "/data/xiaoshuai/facial_lanmark/train_0126/images/"
    # npy_path = '/data/xiaoshuai/facial_landmark_181/prelable/train/'
    # output_path = '/data/xiaoshuai/facial_landmark_181/prelable/train_visualize/'
    
    # dst_path = "/data/xiaoshuai/facial_landmark_181/prelable/train_imgs/"
    
    # input_path = "/data/xiaoshuai/facial_lanmark/train_0126/images/"
    # npy_path = '/data/xiaoshuai/facial_landmark_181/train_0331/npys/'
    # output_path = '/data/xiaoshuai/facial_landmark_181/train_0331/train_visualize/'
    
    input_path = "/data/xiaoshuai/facial_landmark_181/train_0326/images/"
    npy_path = '/data/xiaoshuai/facial_landmark_181/train_0326/npys/'
    output_path = '/data/xiaoshuai/facial_landmark_181/train_0326/train_visualize/'
    
    # dst_path = "/data/xiaoshuai/facial_landmark_181/train_0331/images/"
    
    # if not os.path.exists(dst_path):
    #     os.makedirs(dst_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
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
    print('Total number of images:', test_img_num)
    if test_img_num == 0:
        raise FileNotFoundError('No input image is found...\n')

    for i, img_path in enumerate(input_img_list):
        img_name = os.path.basename(img_path)
        
        npy_name = os.path.join(npy_path, img_name.split(".")[0] + ".npy")
        
        print('Processing image:', npy_name)

        if not os.path.exists(npy_name):
            continue
            # import pdb
            # pdb.set_trace()
            
        print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
        
        infodict = np.load(npy_name, allow_pickle=True).tolist()
        
        landmarks235 = infodict["landmarks181"]

        bboxes = infodict["DFSD_facebbox"]
        gtbox = bboxes[0]
        x1, y1, w, h = gtbox
        
        image = cv2.imread(img_path)
        
        cv2.rectangle(image, (int(x1), int(y1), int(w), int(h)), (255, 255, 0), 1)
        
        label_offset=0
        
        for i in range(181):
            if i in [0, 1, 2, 3, 4, 24, 25, 26, 27, 131, 132, 138, 139]:
                cv2.circle(image, (round(landmarks235[i][0]), round(landmarks235[i][1])), 2, (0, 0, 255), -1)
            else:
                cv2.circle(image, (round(landmarks235[i][0]), round(landmarks235[i][1])), 2, (0, 255, 0), -1)
            
            # label_x = round(landmarks235[i][0]) + 5
            # label_y = round(landmarks235[i][1]) - 5
            # point_idx = label_offset + i
            # label_text = str(point_idx)
            
            # label_color=(255, 255, 255)
            
            # # 绘制黑色轮廓（粗线）
            # cv2.putText(image, label_text, (label_x, label_y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            # # 绘制数字（细线）
            # cv2.putText(image, label_text, (label_x, label_y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)
        
        print(f"Saving visualized image to: {output_path + img_name}")
        cv2.imwrite(output_path + img_name, image)
        
        
        # shutil.copy(img_path, dst_path)


if __name__ == '__main__':
    main()