import os
import glob
import cv2
import numpy as np

def find_extreme_points(landmarks):
    """
    找出关键点中x坐标最小/最大和y坐标最小/最大的点的索引
    """
    if len(landmarks) == 0:
        return None
    
    # 提取所有点的x坐标和y坐标
    x_coords = [point[0] for point in landmarks]
    y_coords = [point[1] for point in landmarks]
    
    # 找出x最小和最大的索引
    min_x_idx = np.argmin(x_coords)
    max_x_idx = np.argmax(x_coords)
    
    # 找出y最小和最大的索引
    min_y_idx = np.argmin(y_coords)
    max_y_idx = np.argmax(y_coords)
    
    return {
        'min_x': {
            'index': int(min_x_idx),
            'coordinate': landmarks[min_x_idx]
        },
        'max_x': {
            'index': int(max_x_idx),
            'coordinate': landmarks[max_x_idx]
        },
        'min_y': {
            'index': int(min_y_idx),
            'coordinate': landmarks[min_y_idx]
        },
        'max_y': {
            'index': int(max_y_idx),
            'coordinate': landmarks[max_y_idx]
        }
    }

def main():    
    # input_path = "/data/xiaoshuai/facial_lanmark/train_1226/val_1229/" 
    # npy_path = "/data/xiaoshuai/facial_lanmark/train_1226/val_1229_box_correct_square_20_d/" 
    # output_path = "/data/xiaoshuai/facial_lanmark/train_1226/visualize/val_1229/"
    
    input_path = "/data/xiaoshuai/facial_lanmark/train_1226/images" 
    npy_path = "/data/xiaoshuai/facial_lanmark/train_1226/box_correct_square_20_d" 
    output_path = '/data/xiaoshuai/facial_lanmark/train_1226'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 收集所有极值点的统计信息
    extreme_stats = {
        'min_x_indices': [],
        'max_x_indices': [],
        'min_y_indices': [],
        'max_y_indices': []
    }
    
    # 用于统计每个索引作为极值点出现的次数
    index_counts = {}
    
    if input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): 
        input_img_list = [input_path]
    else: 
        if input_path.endswith('/'):
            input_path = input_path[:-1]
        input_img_list = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))
    
    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError('No input image is found...\n')
    
    print(f"Found {test_img_num} images to process\n")
    
    for i, img_path in enumerate(input_img_list):
        img_name = os.path.basename(img_path)
        print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
        
        npy_name = os.path.join(npy_path, img_name.split(".")[0] + ".npy")
        
        if not os.path.exists(npy_name):
            print(f"Warning: {npy_name} does not exist, skipping...")
            continue
        
        infodict = np.load(npy_name, allow_pickle=True).tolist()
        landmarks235 = infodict["landmarks208"]
        
        # 找出极值点
        extremes = find_extreme_points(landmarks235)
        
        if extremes:
            # 收集统计信息
            extreme_stats['min_x_indices'].append(extremes['min_x']['index'])
            extreme_stats['max_x_indices'].append(extremes['max_x']['index'])
            extreme_stats['min_y_indices'].append(extremes['min_y']['index'])
            extreme_stats['max_y_indices'].append(extremes['max_y']['index'])
            
            # 更新索引计数
            for key in ['min_x', 'max_x', 'min_y', 'max_y']:
                idx = extremes[key]['index']
                if idx not in index_counts:
                    index_counts[idx] = 0
                index_counts[idx] += 1
            
            # 可视化（可选）
            # bboxes = infodict["DFSD_facebbox"]
            # gtbox = bboxes[0]
            # x1, y1, w, h = gtbox
            
            # image = cv2.imread(img_path)
            # cv2.rectangle(image, (int(x1), int(y1), int(w), int(h)), (0, 0, 255), 2)
            
            # # 画所有关键点
            # for i in range(235):
            #     cv2.circle(image, (round(landmarks235[i][0]), round(landmarks235[i][1])), 2, (0, 255, 0), -1)
            
            # # 用不同颜色标记极值点
            # colors = {
            #     'min_x': (0, 0, 255),    # 红色 - x最小
            #     'max_x': (255, 0, 0),    # 蓝色 - x最大
            #     'min_y': (0, 255, 255),  # 黄色 - y最小
            #     'max_y': (255, 0, 255)   # 紫色 - y最大
            # }
            
            # for key, color in colors.items():
            #     idx = extremes[key]['index']
            #     point = extremes[key]['coordinate']
            #     cv2.circle(image, (round(point[0]), round(point[1])), 5, color, -1)
            #     cv2.putText(image, f'{key}:{idx}', 
            #                (round(point[0])+5, round(point[1])+5),
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # cv2.imwrite(output_path + img_name, image)
    
    
    
    # 统计各类型极值点出现次数
    min_x_counter = {}
    for idx in extreme_stats['min_x_indices']:
        min_x_counter[idx] = min_x_counter.get(idx, 0) + 1
    
    max_x_counter = {}
    for idx in extreme_stats['max_x_indices']:
        max_x_counter[idx] = max_x_counter.get(idx, 0) + 1
    
    min_y_counter = {}
    for idx in extreme_stats['min_y_indices']:
        min_y_counter[idx] = min_y_counter.get(idx, 0) + 1
    
    max_y_counter = {}
    for idx in extreme_stats['max_y_indices']:
        max_y_counter[idx] = max_y_counter.get(idx, 0) + 1
    
    # 找出出现次数大于5的索引（各类型分开统计）
    min_x_frequent_indices = [idx for idx, count in min_x_counter.items() if count > 5]
    max_x_frequent_indices = [idx for idx, count in max_x_counter.items() if count > 5]
    min_y_frequent_indices = [idx for idx, count in min_y_counter.items() if count > 5]
    max_y_frequent_indices = [idx for idx, count in max_y_counter.items() if count > 5]
    
    # 综合四个最大最小点，找出出现次数大于5的索引（去重合并）
    all_frequent_indices = set()
    all_frequent_indices.update(min_x_frequent_indices)
    all_frequent_indices.update(max_x_frequent_indices)
    all_frequent_indices.update(min_y_frequent_indices)
    all_frequent_indices.update(max_y_frequent_indices)
    
    # 转换为排序后的列表
    sorted_all_frequent_indices = sorted(list(all_frequent_indices))
    
    print(f"\n综合结果（去重合并，共{len(sorted_all_frequent_indices)}个索引）：")
    print("索引列表:", sorted_all_frequent_indices)
    
    # 按类型分组输出
    print(f"\nX坐标最小点出现次数大于5的索引（{len(min_x_frequent_indices)}个）：")
    print("索引列表:", sorted(min_x_frequent_indices))
    
    print(f"\nX坐标最大点出现次数大于5的索引（{len(max_x_frequent_indices)}个）：")
    print("索引列表:", sorted(max_x_frequent_indices))
    
    print(f"\nY坐标最小点出现次数大于5的索引（{len(min_y_frequent_indices)}个）：")
    print("索引列表:", sorted(min_y_frequent_indices))
    
    print(f"\nY坐标最大点出现次数大于5的索引（{len(max_y_frequent_indices)}个）：")
    print("索引列表:", sorted(max_y_frequent_indices))
                                         
    # 打印统计结果
    print("\n" + "="*60)
    print("统计结果：")
    print("="*60)
    
    print("\n1. X坐标最小点的可能位置（按出现频率排序）：")
    min_x_counter = {}
    for idx in extreme_stats['min_x_indices']:
        min_x_counter[idx] = min_x_counter.get(idx, 0) + 1
    
    sorted_min_x = sorted(min_x_counter.items(), key=lambda x: x[1], reverse=True)
    for idx, count in sorted_min_x:
        percentage = (count / test_img_num) * 100
        print(f"   索引 {idx}: 出现 {count} 次 ({percentage:.1f}%)")
    
    print("\n2. X坐标最大点的可能位置（按出现频率排序）：")
    max_x_counter = {}
    for idx in extreme_stats['max_x_indices']:
        max_x_counter[idx] = max_x_counter.get(idx, 0) + 1
    
    sorted_max_x = sorted(max_x_counter.items(), key=lambda x: x[1], reverse=True)
    for idx, count in sorted_max_x:
        percentage = (count / test_img_num) * 100
        print(f"   索引 {idx}: 出现 {count} 次 ({percentage:.1f}%)")
    
    print("\n3. Y坐标最小点的可能位置（按出现频率排序）：")
    min_y_counter = {}
    for idx in extreme_stats['min_y_indices']:
        min_y_counter[idx] = min_y_counter.get(idx, 0) + 1
    
    sorted_min_y = sorted(min_y_counter.items(), key=lambda x: x[1], reverse=True)
    for idx, count in sorted_min_y:
        percentage = (count / test_img_num) * 100
        print(f"   索引 {idx}: 出现 {count} 次 ({percentage:.1f}%)")
    
    print("\n4. Y坐标最大点的可能位置（按出现频率排序）：")
    max_y_counter = {}
    for idx in extreme_stats['max_y_indices']:
        max_y_counter[idx] = max_y_counter.get(idx, 0) + 1
    
    sorted_max_y = sorted(max_y_counter.items(), key=lambda x: x[1], reverse=True)
    for idx, count in sorted_max_y:
        percentage = (count / test_img_num) * 100
        print(f"   索引 {idx}: 出现 {count} 次 ({percentage:.1f}%)")
    
    print("\n5. 最常作为极值点的关键点索引（总出现次数）：")
    sorted_indices = sorted(index_counts.items(), key=lambda x: x[1], reverse=True)
    for idx, count in sorted_indices[:10]:  # 显示前10个
        print(f"   索引 {idx}: 总共出现 {count} 次")
    
    # 保存统计结果到文件
    stats_file = os.path.join(output_path, "extreme_points_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("极端关键点位置统计\n")
        f.write("="*60 + "\n\n")
        
        f.write("X坐标最小点统计:\n")
        for idx, count in sorted_min_x:
            f.write(f"  索引 {idx}: 出现 {count} 次\n")
        
        f.write("\nX坐标最大点统计:\n")
        for idx, count in sorted_max_x:
            f.write(f"  索引 {idx}: 出现 {count} 次\n")
        
        f.write("\nY坐标最小点统计:\n")
        for idx, count in sorted_min_y:
            f.write(f"  索引 {idx}: 出现 {count} 次\n")
        
        f.write("\nY坐标最大点统计:\n")
        for idx, count in sorted_max_y:
            f.write(f"  索引 {idx}: 出现 {count} 次\n")
    
    print(f"\n统计结果已保存到: {stats_file}")

if __name__ == '__main__':
    main()