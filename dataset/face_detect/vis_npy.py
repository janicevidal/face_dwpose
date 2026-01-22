import os
import cv2
import glob
import numpy as np

def main():    
    input_path = "/data/xiaoshuai/facial_lanmark/train_1226/val_1229/" 
    npy_path = "/data/xiaoshuai/facial_lanmark/train_1226/val_1229_box_correct_square_20_d/" 
    output_path = "/data/xiaoshuai/facial_lanmark/train_1226/visualize/val_1229/"
    
    # input_path = "/data/zhangxiaoshuai/FacialLandmark/1224/zx/" 
    # npy_path = "/data/zhangxiaoshuai/FacialLandmark/1224/zx/" 
    # output_path = '/data/zhangxiaoshuai/FacialLandmark/1224/visualize_zx/'
    
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
    if test_img_num == 0:
        raise FileNotFoundError('No input image is found...\n')

    for i, img_path in enumerate(input_img_list):
        img_name = os.path.basename(img_path)
        print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
        
        npy_name = os.path.join(npy_path, img_name.split(".")[0] + ".npy")

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
        x1, y1, w, h = gtbox
        
        image = cv2.imread(img_path)
        
        cv2.rectangle(image, (int(x1), int(y1), int(w), int(h)), (0, 0, 255), 2)
        
        for i in range(235):
            cv2.circle(image, (round(landmarks235[i][0]), round(landmarks235[i][1])), 2, (0, 255, 0), -1)
        
        if "squarebbox" in infodict.keys():
            square_bbox = infodict["squarebbox"][0]
            x1, y1, w, h = square_bbox
            
            cv2.rectangle(image, (int(x1), int(y1), int(w), int(h)), (0, 255, 255), 2)
        
        cv2.imwrite(output_path + img_name, image)


if __name__ == '__main__':
    main()