import os
import pickle
import numpy as np

def convert_235_to_181(face_235: np.ndarray, face_181: np.ndarray) -> np.ndarray:
    # 给定的映射关系
    mapping = [
        (0, 23), (2, 22), (4, 21), (6, 20), (8, 19), (10, 18), (12, 17), (14, 16), (16, 15), (18, 14),
        (20, 13), (22, 12), (24, 11), (26, 10), (28, 9), (30, 8), (32, 7), (34, 6), (36, 5),
        (37, 28), (38, 29), (39, 30), (40, 31), (41, 32), (42, 33), (43, 34), (44, 35), (45, 36), (46, 37), (47, 38),
        (48, 39), (49, 40), (50, 41), (51, 42), (52, 43), (53, 44), (54, 45), (55, 46), (56, 47),
        (57, 48), (58, 49), (59, 50), (60, 51), (61, 52), (62, 53), (63, 54), (64, 55), (65, 56), (66, 57), (67, 58),
        (68, 59), (69, 60), (70, 61), (71, 62), (72, 63), (73, 64), (74, 65), (75, 66), (76, 67),
        (77, 68), (78, 69), (79, 70), (80, 71), (81, 72), (82, 73), (83, 74), (84, 75), (85, 76), (86, 77), (87, 78), (88, 79), (89, 80),
        (90, 81), (91, 82), (92, 83), (93, 84), (94, 85), (95, 86), (96, 87), (97, 88), (98, 89), (99, 90), (100, 91),
        (101, 92), (102, 93), (103, 94), (104, 95), (105, 96), (106, 97), (107, 98), (108, 99), (109, 100), (110, 101), (111, 102), (112, 103), (113, 104),
        (114, 105), (115, 106), (116, 107), (117, 108), (118, 109), (119, 110), (120, 111), (121, 112), (122, 113), (123, 114), (124, 115),
        (125, 137), (126, 136), (127, 135), (128, 134), (129, 133),
        (132, 130), (133, 129), (134, 128), (135, 127), (136, 126),
        (138, 140), (139, 141), (140, 142),
        (141, 143), ((142, 143), 144), (144, 145), ((145, 146), 146), (147, 147), (150, 148),
        (153, 149), ((154, 155), 150), (156, 151), ((157, 158), 152), (159, 153),
        ((160, 161), 154), (162, 155), ((163, 164), 156), (165, 157), ((166, 167), 158), (168, 159),
        ((169, 170), 160), (171, 161), ((172, 173), 162), (174, 163), ((175, 176), 164), (177, 165),
        ((178, 179), 166), (180, 167), ((181, 182), 168), (183, 169),
        ((184, 185), 170), (186, 171), ((187, 188), 172), (189, 173),
        ((190, 191), 174), (192, 175), ((193, 194), 176), (195, 177),
        ((196, 197), 178), (198, 179), ((199, 200), 180),
        (201, 120), (202, 125), (215, 116), (203, 117), (207, 118), (211, 119), (231, 121), (219, 122), (223, 123), (227, 124),
        # add by 181
        # (235, 24), (236, 25), (237, 26), (238, 27), (239, 0), (240, 1), (241, 2), (242, 3), (243, 4),
        # ((37, 67), 138), ((77, 113), 139), ((130, 140), 132), ((140, 131), 131),
    ]

    for item in mapping:
        # 判断是否为平均映射（源为两个点）
        if isinstance(item[0], (tuple, list)):
            src_indices = item[0]      # 例如 (142, 143)
            target_idx = item[1]       # 例如 144
            # 取两个源点的平均值
            avg_point = (face_235[src_indices[0]] + face_235[src_indices[1]]) / 2.0
            face_181[target_idx] = avg_point
        else:
            # 直接映射
            src_idx = item[0]
            target_idx = item[1]
            face_181[target_idx] = face_235[src_idx]

    return face_181


if __name__ == '__main__':
    
    # pkl_file_path = "/home/zhangxiaoshuai/Project/face_dwpose/mmpose/kpts_181_mix_rtmw_l_val_results.pkl"
    # npy_235_dir = "/data/xiaoshuai/facial_lanmark/train_1226/val_1229_box_correct_square_20_d/"
    # save_dir = "/data/xiaoshuai/facial_landmark_181/prelable/val/"
    
    # pkl_file_path = "/home/zhangxiaoshuai/Project/face_dwpose/mmpose/kpts_181_mix_rtmw_l_train_results.pkl"
    # npy_235_dir = "/data/xiaoshuai/facial_lanmark/train_0126/box_correct_square_20_d"
    # save_dir = "/data/xiaoshuai/facial_landmark_181/prelable/train/"
    
    # pkl_file_path = "/home/zhangxiaoshuai/Project/face_dwpose/mmpose/kpts_181_mix_rtmw_l_train_results_0325.pkl"
    # npy_235_dir = "/data/xiaoshuai/facial_lanmark/train_0325/images_all_181_filter/"
    # save_dir = "/data/xiaoshuai/facial_landmark_181/prelable/train_0325/"
    
    pkl_file_path = "/home/zhangxiaoshuai/Project/face_dwpose/mmpose/kpts_181_mix_rtmw_l_train_results_0331.pkl"
    npy_235_dir = "/data/xiaoshuai/facial_lanmark/train_0126/box_correct_square_20_d"
    save_dir = "/data/xiaoshuai/facial_landmark_181/train_0331/npys/"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    for i in range(len(data)):    
        item = data[i]
        pred_instances = item['pred_instances']

        img_path = item['img_path']

        name = img_path.split('/')[-1].split('.')[0]

        npy_name = name + ".npy"
        ref_npy_path = os.path.join(npy_235_dir, npy_name)

        refdict = np.load(ref_npy_path, allow_pickle=True).tolist()

        landmarks235 = refdict["landmarks208"]
        bboxes = refdict["DFSD_facebbox"]

        keypoints = pred_instances['keypoints'][0]
        # bbox = pred_instances['bboxes'][0]
        # x1, y1, x2, y2 = bbox
        # w = x2 - x1
        # h = y2 - y1

        landmarks181 = convert_235_to_181(landmarks235, keypoints)
        visibility = np.asarray([2] * landmarks181.shape[0])
        
        
        if "squarebbox" in refdict:
            squarebox = refdict["squarebbox"]

            infodict = {
                "landmarks181": landmarks181,      # (181,2)
                # "DFSD_facebbox": np.array([[x1, y1, w, h]]).tolist(),
                "DFSD_facebbox": bboxes,
                "visibility": visibility, 
                "squarebbox": squarebox 
            }
        else:
            infodict = {
                "landmarks181": landmarks181,      # (181,2)
                "DFSD_facebbox": bboxes,
                "visibility": visibility, 
            }

        np.save(os.path.join(save_dir, npy_name) , infodict, allow_pickle=True)