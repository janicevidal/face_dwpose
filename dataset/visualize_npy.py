import os
import cv2
import glob
import numpy as np

def main():    
    input_path = "/data/caiachang/video-ldms-ok/TEST/only1face/" 
    output_path = '/data/xiaoshuai/facial_lanmark/visualize_test/'
    
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
        
        npy_path = img_path.split(".")[0] + ".npy"

        if not os.path.exists(npy_path):
            import pdb
            pdb.set_trace()
        
        infodict = np.load(npy_path, allow_pickle=True).tolist()
        landmarks235 = infodict["landmarks208"]

        bboxes = infodict["DFSD_facebbox"]
        gtbox = bboxes[0]
        x1, y1, w, h = gtbox
        
        image = cv2.imread(img_path)
        
        cv2.rectangle(image, (int(x1), int(y1), int(w), int(h)), (0, 0, 255), 2)
        
        for i in range(235):
            cv2.circle(image, (round(landmarks235[i][0]), round(landmarks235[i][1])), 2, (0, 255, 0), -1)
        
        cv2.imwrite(output_path + img_name, image)


if __name__ == '__main__':
    main()