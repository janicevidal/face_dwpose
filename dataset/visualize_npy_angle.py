import os
import cv2
import glob
import json
import numpy as np
import math
from math import cos, sin


def parse_kuaishou_euler(jsonf):
    assert os.path.exists(jsonf), "file not exists: {}".format(jsonf)
    with open(jsonf, "r") as f:
        data = json.load(f)

    try:
        data = data["data"]
        faceInfos = data["face"]  # 列表类型 多个人脸
        
        # 这里取消assert 已经将不等于1的打印出来了 位于 ./注意人脸数目不等1的数据.py  ==> 自行处理之
        # assert len(faceInfos) == 1, "人脸数目不等于1~"
        faceInfo = faceInfos[0]
 
        # 每个人脸信息包括:  ['roll', 'yaw', 'pitch', 'age', 'rect', 'point']
        roll = faceInfo["roll"] 
        yaw = faceInfo["yaw"]
        pitch = faceInfo["pitch"]

        return (roll, yaw, pitch)
    
    except (KeyError, Exception) as e:
        print(f"Error: {jsonf} ==> Parsing json failed: {e}")
        return None

def parse_kuaishou_euler_bbox(jsonf):
    assert os.path.exists(jsonf), "file not exists: {}".format(jsonf)
    with open(jsonf, "r") as f:
        data = json.load(f)

    try:
        data = data["data"]
        faceInfos = data["face"]  # 列表类型 多个人脸
        
        # 这里取消assert 已经将不等于1的打印出来了 位于 ./注意人脸数目不等1的数据.py  ==> 自行处理之
        # assert len(faceInfos) == 1, "人脸数目不等于1~"
        faceInfo = faceInfos[0]
 
        # 每个人脸信息包括:  ['roll', 'yaw', 'pitch', 'age', 'rect', 'point']
        roll = faceInfo["roll"] 
        yaw = faceInfo["yaw"]
        pitch = faceInfo["pitch"]
        
        bbox = faceInfo["rect"] 

        return (roll, yaw, pitch, bbox["left"], bbox["top"], bbox["width"], bbox["height"])
    
    except (KeyError, Exception) as e:
        print(f"Error: {jsonf} ==> Parsing json failed: {e}")
        return None

def draw_axis(img, yaw_, pitch_, roll_, tdx=None, tdy=None, size = 100):

    # pitch = pitch * np.pi / 180
    # yaw = -(yaw * np.pi / 180)
    # roll = roll * np.pi / 180
    
    yaw_value = yaw_ * 180 / np.pi
    pitch_value = pitch_ * 180 / np.pi
    roll_value = roll_ * 180 / np.pi
    
    roll = -roll_
    pitch = -pitch_
    yaw = yaw_ 

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

def main():    
    # input_path = "/data/caiachang/video-ldms-ok/TEST/only1face/" 
    # output_path = '/data/xiaoshuai/facial_lanmark/visualize_test/'
    
    # input_path = "/data/xiaoshuai/facial_lanmark/ffhq-aug3d/euler-exp/" 
    # output_path = '/data/xiaoshuai/facial_lanmark/visualize_euler_exp/'
    
    # input_path = "/data/xiaoshuai/facial_lanmark/ffhq-aug3d/euler/" 
    # output_path = '/data/xiaoshuai/facial_lanmark/visualize_euler/'
    
    input_path = "/data/xiaoshuai/facial_lanmark/val_1118" 
    output_path = '/data/xiaoshuai/facial_lanmark/val_1118_visualize_angle/'
    
    EULER_PATH_MAP = {
        "ffhq": "/data/caiachang/ffhq/KuaiShou-ldms/rawData_mask",
        "only1face": "/data/caiachang/onlyOneFace/KuaiShou-ldms",
        "celeba": "/data/caiachang/CelebA/video-ldms/KuaiShou-ldms"
    }
    
    EULER_AUG3D_PATH ={
        "euler-exp":"/data/caiachang/video-ldms-ok/00Aug3d/ffhq/KuaiShou-pts/aug-all",
        "euler":"/data/caiachang/video-ldms-ok/00Aug3d/ffhq/KuaiShou-pts/aug-euler",
        "exp":"/data/caiachang/video-ldms-ok/00Aug3d/ffhq/KuaiShou-pts/aug-exp"
    }
    
    
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
        
        if "euler-exp" in img_name or "euler" in img_name or "exp" in img_name:
            prefix = img_name.split('_')[-1].split('.')[0]
            if prefix in EULER_AUG3D_PATH:
                euler_base_path = EULER_AUG3D_PATH[prefix]
                euler_path = os.path.join(euler_base_path, img_name.split('_')[1] + '.jpg.json')
                print(f"  Aug3d data detected, euler path: {euler_path}")
            
            if not os.path.exists(euler_path):
                print(f"Warning: No corresponding euler file for {img_name}, skipping...")
                import pdb
                pdb.set_trace()
        
            euler_info = parse_kuaishou_euler_bbox(euler_path)
                
                
        else:
            prefix = img_name.split('_')[0]
            if prefix in EULER_PATH_MAP:
                euler_base_path = EULER_PATH_MAP[prefix]
                euler_path = os.path.join(euler_base_path, os.path.splitext(img_name.split('_')[1])[0] + '.json')
            
            if not os.path.exists(euler_path):
                print(f"Warning: No corresponding euler file for {img_name}, skipping...")
                import pdb
                pdb.set_trace()
        
            euler_info = parse_kuaishou_euler(euler_path)
            
            # continue  # 直接跳过非aug3d的数据
        
        if euler_info is None:
            print(f"Warning: Parsing euler info failed for {img_name}, skipping...")
            import pdb
            pdb.set_trace()
        
        if len(np.asarray(euler_info)) == 3:
            roll, yaw, pitch = euler_info
        
        else:
            roll, yaw, pitch, x1, y1, w, h = euler_info
            
        roll_np, yaw_np, pitch_np = np.asarray(roll), np.asarray(yaw), np.asarray(pitch)
                    
        npy_path = img_path.split(".")[0] + ".npy"

        if not os.path.exists(npy_path):
            import pdb
            pdb.set_trace()
        
        infodict = np.load(npy_path, allow_pickle=True).tolist()
        
        # print(infodict.keys())
        # print(infodict["visibility"])
        # print(infodict["landmark_verified"])
        # print(infodict["DFSD_facebbox"])
        # print(infodict["landmarks208"])
        # import pdb
        # pdb.set_trace()
        
        landmarks235 = infodict["landmarks208"]

        if len(np.asarray(euler_info)) == 3:
            bboxes = infodict["DFSD_facebbox"]
            gtbox = bboxes[0]
            x1, y1, w, h = gtbox
        
        image = cv2.imread(img_path)
        
        tdx = int(x1 + w / 2)
        tdy = int(y1 + h / 2)
        
        draw_axis(image, yaw_np, pitch_np, roll_np, tdx=tdx, tdy=tdy, size=w//2)
        
        cv2.rectangle(image, (int(x1), int(y1), int(w), int(h)), (0, 0, 255), 2)
        
        # for i in range(235):
        #     cv2.circle(image, (round(landmarks235[i][0]), round(landmarks235[i][1])), 2, (0, 255, 0), -1)
        
        cv2.imwrite(output_path + img_name, image)


if __name__ == '__main__':
    main()