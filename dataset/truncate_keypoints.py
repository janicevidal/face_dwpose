import json

def select_keypoints_by_indices(input_json_path, output_json_path, indices):
    """
    根据给定的索引列表（从0开始），从每个标注的keypoints中提取对应点，
    重新构建keypoints，并更新num_keypoints为len(indices)，其他字段保持不变。
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    for ann in coco_data.get('annotations', []):
        keypoints = ann.get('keypoints', [])
        num_points_original = len(keypoints) // 3

        new_keypoints = []
        for idx in indices:
            if idx < num_points_original:
                start = idx * 3
                new_keypoints.extend(keypoints[start:start+3])
            else:
                # 如果索引超出原有点数，补零（与原生成逻辑一致）
                new_keypoints.extend([0, 0, 0])
                print(f"警告: 标注 {ann.get('id')} 索引 {idx} 超出范围，已补零")
        ann['keypoints'] = new_keypoints
        ann['num_keypoints'] = len(indices)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)  # 如需压缩可删除indent


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("用法: python truncate_keypoints.py <输入文件.json> <输出文件.json>")
        sys.exit(1)
    
    selected_indices = list(range(203)) + [207, 215, 223, 231]
    
    select_keypoints_by_indices(sys.argv[1], sys.argv[2], selected_indices)