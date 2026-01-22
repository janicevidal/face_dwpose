import os
import cv2
import glob
import numpy as np

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float32)  # shape (4, 2)
    points2 = points2.astype(np.float32)  # shape (4, 2)

    c1 = np.mean(points1, axis=0)  # shape (1, 2)
    c2 = np.mean(points2, axis=0)  # shape (1, 2)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)  # single value #standard deviation
    s2 = np.std(points2)  # single value #standard deviation
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)  # points1.T * points2 是2*2的矩阵   ==  2*4的矩阵和4*2的矩阵相乘
    R = (U * Vt).T  # shape (2, 2)
    M = (s2 / s1) * R  # shape (2, 2)
    B = c2.T - (s2 / s1) * R * c1.T  # shape (2, 1)
    M_inv = M.I  # shape (2, 2)
    B_inv = - B  # shape (2, 1)
    return np.hstack((M, B)), M, B, M_inv, B_inv

def compose_affine_transforms(M1, B1, M2, B2):
    """
    组合两个仿射变换: T_combined = T2 ∘ T1
    变换形式: y = Mx + B
    
    返回:
        MB_combined: 组合变换矩阵 (2x3)
        M_inv_combined: 组合逆变换的旋转缩放部分 (2x2)
        B_inv_combined: 组合逆变换的平移部分 (2x1)
    """
    # 组合变换: T_combined(x) = M2 * (M1 * x + B1) + B2 = (M2 * M1) * x + (M2 * B1 + B2)
    M_combined = M2 @ M1  # 2x2矩阵
    B_combined = M2 @ B1 + B2  # 2x1向量
    
    MB_combined = np.hstack((M_combined, B_combined.reshape(-1, 1)))
    
    # 计算逆变换的分量
    M_inv_combined = np.linalg.inv(M_combined)  # 2x2矩阵
    B_inv_combined = -B_combined  # 2x1向量
    
    return MB_combined, M_inv_combined, B_inv_combined

def main():    
    # input_path = "/data/xiaoshuai/facial_lanmark/datasets/lpff/images_all_1226/" 
    # npy_path = "/data/xiaoshuai/facial_lanmark/datasets/lpff/train_1226_box_correct_square/" 
    # output_path = "/data/xiaoshuai/facial_lanmark/train_1226/visualize/aligned_faces/"
    
    input_path = "/data/xiaoshuai/facial_lanmark/train_1226/val_1229/" 
    npy_path = "/data/xiaoshuai/facial_lanmark/train_1226/val_1229_box_correct_square_20_d/" 
    output_path = "/data/xiaoshuai/facial_lanmark/train_1226/visualize/aligned_faces/"
    
    mean_landmarks_path = "/data/xiaoshuai/facial_lanmark/train_1226/mean_face_20_d/mean_face_symmetric_centered.npy"
    mean_landmarks = np.load(mean_landmarks_path)
    
    mean_landmarks_path_2 = "/data/xiaoshuai/facial_lanmark/train_1226/mean_face_20_d/mean_landmarks.npy"
    mean_landmarks_2 = np.load(mean_landmarks_path_2)
    
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
    
    indices_5 = [139, 141, 159, 201, 202]
    indices_10 = [0, 9, 18, 27, 36, 139, 141, 159, 201, 202]
    indices_24 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 139, 141, 159, 201, 202]
    indices_19 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
    
    pts_dst_5 = mean_landmarks[indices_5]
    pts_dst_10 = mean_landmarks[indices_10]
    pts_dst_24 = mean_landmarks[indices_24]
    pts_dst_19 = mean_landmarks[indices_19]
    
    pts_dst_24_2 = mean_landmarks_2[indices_24]
    
    size = 128
    
    x_coords = mean_landmarks[:, 0] * size
    y_coords = mean_landmarks[:, 1] * size

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    rect_width = max_x - min_x
    rect_height = max_y - min_y

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    square_size = max(rect_width, rect_height)

    square_min_x = center_x - square_size / 2
    square_max_x = center_x + square_size / 2
    square_min_y = center_y - square_size / 2
    square_max_y = center_y + square_size / 2
        
    square_mean_box = np.array([
        [square_min_x, square_min_y],  # 左上角
        [square_max_x, square_min_y],  # 右上角
        [square_min_x, square_max_y],  # 左下角
        [square_max_x, square_max_y]   # 右下角
    ]).astype(np.float32)
    
    square_size_avg = (rect_width + rect_height) * 0.5

    square_min_x_avg = center_x - square_size_avg / 2
    square_max_x_avg = center_x + square_size_avg / 2
    square_min_y_avg = center_y - square_size_avg / 2
    square_max_y_avg = center_y + square_size_avg / 2
        
    square_mean_box_avg = np.array([
        [square_min_x_avg, square_min_y_avg],  # 左上角
        [square_max_x_avg, square_min_y_avg],  # 右上角
        [square_min_x_avg, square_max_y_avg],  # 左下角
        [square_max_x_avg, square_max_y_avg]   # 右下角
    ]).astype(np.float32)
    
    
    for i, img_path in enumerate(input_img_list):
        img_name = os.path.basename(img_path)
        print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
        
        npy_name = os.path.join(npy_path, img_name.split(".")[0] + ".npy")

        if not os.path.exists(npy_name):
            import pdb
            pdb.set_trace()
        
        infodict = np.load(npy_name, allow_pickle=True).tolist()
        
        landmarks235 = infodict["landmarks208"]

        image = cv2.imread(img_path)
        
        square_bbox = infodict["squarebbox"][0]
        x1, y1, w, h = square_bbox
        
        roi_face = image[int(y1):int(y1+h-1), int(x1):int(x1+w-1)]
        
        roi_face = cv2.resize(roi_face, (size, size))
        
        # 5 points alignment        
        pts_src_5 = landmarks235[indices_5]
        MB, M, B, M_inv, B_inv = transformation_from_points(np.matrix(pts_src_5), np.matrix(pts_dst_5 * size))
        dst_5 = cv2.warpAffine(image, MB, (size, size))
        
        landmarks_align = []
        for i in range(len(landmarks235)): 
            align_x = landmarks235[i][0] * MB[0, 0] + landmarks235[i][1] * MB[0, 1] + MB[0, 2]
            align_y = landmarks235[i][0] * MB[1, 0] + landmarks235[i][1] * MB[1, 1] + MB[1, 2]
            landmarks_align.append((align_x, align_y))
        
        for i in range(36):
            cv2.circle(dst_5, (round(landmarks_align[i][0]), round(landmarks_align[i][1])), 2, (0, 255, 0), -1)
        
        
        # 10 points alignment
        pts_src_10 = landmarks235[indices_10]
        MB, M, B, M_inv, B_inv = transformation_from_points(np.matrix(pts_src_10), np.matrix(pts_dst_10 * size))
        dst_10 = cv2.warpAffine(image, MB, (size, size))
        
        landmarks_align = []
        for i in range(len(landmarks235)): 
            align_x = landmarks235[i][0] * MB[0, 0] + landmarks235[i][1] * MB[0, 1] + MB[0, 2]
            align_y = landmarks235[i][0] * MB[1, 0] + landmarks235[i][1] * MB[1, 1] + MB[1, 2]
            landmarks_align.append((align_x, align_y))
        
        for i in range(36):
            cv2.circle(dst_10, (round(landmarks_align[i][0]), round(landmarks_align[i][1])), 2, (0, 255, 0), -1)
        
        # 24 points alignment    
        pts_src_24 = landmarks235[indices_24]
        MB, M, B, M_inv, B_inv = transformation_from_points(np.matrix(pts_src_24), np.matrix(pts_dst_24 * size))
        dst_24 = cv2.warpAffine(image, MB, (size, size))
        
        landmarks_align = []
        for i in range(len(landmarks235)): 
            align_x = landmarks235[i][0] * MB[0, 0] + landmarks235[i][1] * MB[0, 1] + MB[0, 2]
            align_y = landmarks235[i][0] * MB[1, 0] + landmarks235[i][1] * MB[1, 1] + MB[1, 2]
            landmarks_align.append((align_x, align_y))
        
        for i in range(36):
            cv2.circle(dst_24, (round(landmarks_align[i][0]), round(landmarks_align[i][1])), 2, (0, 255, 0), -1)
        
        # 19 points alignment
        pts_src_19 = landmarks235[indices_19]
        MB, M, B, M_inv, B_inv = transformation_from_points(np.matrix(pts_src_19), np.matrix(pts_dst_19 * size))
        dst_19 = cv2.warpAffine(image, MB, (size, size))
        
        landmarks_align = []
        for i in range(len(landmarks235)): 
            align_x = landmarks235[i][0] * MB[0, 0] + landmarks235[i][1] * MB[0, 1] + MB[0, 2]
            align_y = landmarks235[i][0] * MB[1, 0] + landmarks235[i][1] * MB[1, 1] + MB[1, 2]
            landmarks_align.append((align_x, align_y))
        
        for i in range(36):
            cv2.circle(dst_19, (round(landmarks_align[i][0]), round(landmarks_align[i][1])), 2, (0, 255, 0), -1)
        
        # 24 points alignment 2
        # pts_src_24 = landmarks235[indices_24]
        # MB, M, B, M_inv, B_inv = transformation_from_points(np.matrix(pts_src_24), np.matrix(pts_dst_24_2 * size))
        # dst_24_2 = cv2.warpAffine(image, MB, (size, size))
        
        # landmarks_align = []
        # for i in range(len(landmarks235)): 
        #     align_x = landmarks235[i][0] * MB[0, 0] + landmarks235[i][1] * MB[0, 1] + MB[0, 2]
        #     align_y = landmarks235[i][0] * MB[1, 0] + landmarks235[i][1] * MB[1, 1] + MB[1, 2]
        #     landmarks_align.append((align_x, align_y))
        
        # for i in range(36):
        #     cv2.circle(dst_24_2, (round(landmarks_align[i][0]), round(landmarks_align[i][1])), 2, (0, 255, 0), -1)
        
        # 24 points alignment box adjustment
        pts_src_24 = landmarks235[indices_24]
        MB, M1, B1, M_inv, B_inv = transformation_from_points(np.matrix(pts_src_24), np.matrix(pts_dst_24 * size))
        
        landmarks_align = []
        for i in range(len(landmarks235)): 
            align_x = landmarks235[i][0] * MB[0, 0] + landmarks235[i][1] * MB[0, 1] + MB[0, 2]
            align_y = landmarks235[i][0] * MB[1, 0] + landmarks235[i][1] * MB[1, 1] + MB[1, 2]
            landmarks_align.append((align_x, align_y))
        
        x_coords = np.array(landmarks_align)[:, 0]
        y_coords = np.array(landmarks_align)[:, 1]

        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)

        rect_width = max_x - min_x
        rect_height = max_y - min_y

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        square_size = max(rect_width, rect_height)

        square_min_x = center_x - square_size / 2
        square_max_x = center_x + square_size / 2
        square_min_y = center_y - square_size / 2
        square_max_y = center_y + square_size / 2
        
        square_box = np.array([
            [square_min_x, square_min_y],  # 左上角
            [square_max_x, square_min_y],  # 右上角
            [square_min_x, square_max_y],  # 左下角
            [square_max_x, square_max_y]   # 右下角
        ]).astype(np.float32)
        
        MB, M, B, M_inv, B_inv = transformation_from_points(np.matrix(square_box), np.matrix(square_mean_box))
        
        landmarks_align_new = []
        for i in range(len(landmarks_align)): 
            align_x = landmarks_align[i][0] * MB[0, 0] + landmarks_align[i][1] * MB[0, 1] + MB[0, 2]
            align_y = landmarks_align[i][0] * MB[1, 0] + landmarks_align[i][1] * MB[1, 1] + MB[1, 2]
            landmarks_align_new.append((align_x, align_y))
        
        MB, M, B, M_inv, B_inv = transformation_from_points(np.matrix(pts_src_24), np.matrix(np.array(landmarks_align_new)[indices_24]))
        
        dst_24_box = cv2.warpAffine(image, MB, (size, size))
        
        landmarks_align_final = []
        for i in range(len(landmarks235)): 
            align_x = landmarks235[i][0] * MB[0, 0] + landmarks235[i][1] * MB[0, 1] + MB[0, 2]
            align_y = landmarks235[i][0] * MB[1, 0] + landmarks235[i][1] * MB[1, 1] + MB[1, 2]
            landmarks_align_final.append((align_x, align_y))

        for i in range(36):
            cv2.circle(dst_24_box, (round(landmarks_align_final[i][0]), round(landmarks_align_final[i][1])), 2, (0, 255, 0), -1)
        
        
        # 验证
        # 计算从square_box到square_mean_box的变换（只有平移和缩放）
        box_center = np.mean(square_box, axis=0)
        mean_box_center = np.mean(square_mean_box, axis=0)
        
        box_size = np.max([square_box[1][0] - square_box[0][0], 
                          square_box[2][1] - square_box[0][1]])
        mean_box_size = np.max([square_mean_box[1][0] - square_mean_box[0][0], 
                               square_mean_box[2][1] - square_mean_box[0][1]])
        
        scale_box = mean_box_size / box_size
        M2 = np.array([[scale_box, 0], [0, scale_box]])
        B2 = mean_box_center - scale_box * box_center
        B2 = B2.reshape(-1, 1)  # 转换为列向量
        
        # 组合变换：MB_combined = MB2 ∘ MB1，同时获取逆矩阵
        MB_combined, M_inv_combined, B_inv_combined = compose_affine_transforms(M1, B1, M2, B2)
        
        print(np.allclose(MB_combined, MB, atol=0.0001))
            
            
            
        # 235 points alignment box adjustment all landmarks
        # MB, M, B, M_inv, B_inv = transformation_from_points(np.matrix(landmarks235), np.matrix(np.array(landmarks_align_new)))
        
        # dst_24_box_all = cv2.warpAffine(image, MB, (size, size))
        
        # landmarks_align_final_all = []
        # for i in range(len(landmarks235)): 
        #     align_x = landmarks235[i][0] * MB[0, 0] + landmarks235[i][1] * MB[0, 1] + MB[0, 2]
        #     align_y = landmarks235[i][0] * MB[1, 0] + landmarks235[i][1] * MB[1, 1] + MB[1, 2]
        #     landmarks_align_final_all.append((align_x, align_y))

        # for i in range(36):
        #     cv2.circle(dst_24_box_all, (round(landmarks_align_final_all[i][0]), round(landmarks_align_final_all[i][1])), 2, (0, 255, 0), -1)
        
        # 24 points alignment box adjustment average
        pts_src_24 = landmarks235[indices_24]
        MB, M, B, M_inv, B_inv = transformation_from_points(np.matrix(pts_src_24), np.matrix(pts_dst_24 * size))
        
        landmarks_align = []
        for i in range(len(landmarks235)): 
            align_x = landmarks235[i][0] * MB[0, 0] + landmarks235[i][1] * MB[0, 1] + MB[0, 2]
            align_y = landmarks235[i][0] * MB[1, 0] + landmarks235[i][1] * MB[1, 1] + MB[1, 2]
            landmarks_align.append((align_x, align_y))
        
        x_coords = np.array(landmarks_align)[:, 0]
        y_coords = np.array(landmarks_align)[:, 1]

        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)

        rect_width = max_x - min_x
        rect_height = max_y - min_y

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        square_size = (rect_width + rect_height) * 0.5

        square_min_x = center_x - square_size / 2
        square_max_x = center_x + square_size / 2
        square_min_y = center_y - square_size / 2
        square_max_y = center_y + square_size / 2
        
        square_box = np.array([
            [square_min_x, square_min_y],  # 左上角
            [square_max_x, square_min_y],  # 右上角
            [square_min_x, square_max_y],  # 左下角
            [square_max_x, square_max_y]   # 右下角
        ]).astype(np.float32)
        
        
        MB, M, B, M_inv, B_inv = transformation_from_points(np.matrix(square_box), np.matrix(square_mean_box_avg))
        
        landmarks_align_new = []
        for i in range(len(landmarks_align)): 
            align_x = landmarks_align[i][0] * MB[0, 0] + landmarks_align[i][1] * MB[0, 1] + MB[0, 2]
            align_y = landmarks_align[i][0] * MB[1, 0] + landmarks_align[i][1] * MB[1, 1] + MB[1, 2]
            landmarks_align_new.append((align_x, align_y))
        
        MB, M, B, M_inv, B_inv = transformation_from_points(np.matrix(pts_src_24), np.matrix(np.array(landmarks_align_new)[indices_24]))
        
        dst_24_box_ave = cv2.warpAffine(image, MB, (size, size))
        
        landmarks_align_final = []
        for i in range(len(landmarks235)): 
            align_x = landmarks235[i][0] * MB[0, 0] + landmarks235[i][1] * MB[0, 1] + MB[0, 2]
            align_y = landmarks235[i][0] * MB[1, 0] + landmarks235[i][1] * MB[1, 1] + MB[1, 2]
            landmarks_align_final.append((align_x, align_y))

        for i in range(36):
            cv2.circle(dst_24_box_ave, (round(landmarks_align_final[i][0]), round(landmarks_align_final[i][1])), 2, (0, 255, 0), -1)
        
        
        # concat_img = np.zeros((size, size*4+5*3, 3), dtype=np.uint8)
        # concat_img = np.zeros((size, size*5+5*4, 3), dtype=np.uint8)
        concat_img = np.zeros((size, size*6+5*5, 3), dtype=np.uint8)
        
        concat_img[:, :size] = roi_face
        concat_img[:, size+5:size*2+5] = dst_5
        concat_img[:, size*2+5*2:size*3+5*2] = dst_19
        concat_img[:, size*3+5*3:size*4+5*3] = dst_24
        concat_img[:, size*4+5*4:size*5+5*4] = dst_24_box
        concat_img[:, size*5+5*5:size*6+5*5] = dst_24_box_ave
        
        cv2.imwrite(output_path + img_name, concat_img)


if __name__ == '__main__':
    main()