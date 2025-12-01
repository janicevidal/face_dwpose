import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import glob
import pandas as pd
from datetime import datetime

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def parse_kuaishou_euler(jsonf):
    """解析快手欧拉角数据"""
    assert os.path.exists(jsonf), "file not exists: {}".format(jsonf)
    with open(jsonf, "r") as f:
        data = json.load(f)

    try:
        data = data["data"]
        faceInfos = data["face"]  # 列表类型 多个人脸
        
        if len(faceInfos) == 0:
            print(f"Warning: No face detected in {jsonf}")
            return None
            
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
    """解析快手欧拉角数据和边界框"""
    assert os.path.exists(jsonf), "file not exists: {}".format(jsonf)
    with open(jsonf, "r") as f:
        data = json.load(f)

    try:
        data = data["data"]
        faceInfos = data["face"]  # 列表类型 多个人脸
        
        if len(faceInfos) == 0:
            print(f"Warning: No face detected in {jsonf}")
            return None
            
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

def collect_euler_angles(input_path):
    """收集所有样本的欧拉角数据"""
    rolls, yaws, pitches = [], [], []
    
    # 路径映射
    EULER_PATH_MAP = {
        "ffhq": "/data/caiachang/ffhq/KuaiShou-ldms/rawData_mask",
        "only1face": "/data/caiachang/onlyOneFace/KuaiShou-ldms",
        "celeba": "/data/caiachang/CelebA/video-ldms/KuaiShou-ldms"
    }
    
    EULER_AUG3D_PATH = {
        "euler-exp": "/data/caiachang/video-ldms-ok/00Aug3d/ffhq/KuaiShou-pts/aug-all",
        "euler": "/data/caiachang/video-ldms-ok/00Aug3d/ffhq/KuaiShou-pts/aug-euler",
        "exp": "/data/caiachang/video-ldms-ok/00Aug3d/ffhq/KuaiShou-pts/aug-exp"
    }
    
    # 获取所有图像文件
    if input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): 
        input_img_list = [input_path]
    else: 
        if input_path.endswith('/'):
            input_path = input_path[:-1]
        input_img_list = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))
    
    print(f"找到 {len(input_img_list)} 个图像文件")
    
    valid_count = 0
    for i, img_path in enumerate(input_img_list):
        img_name = os.path.basename(img_path)
        if (i + 1) % 1000 == 0:
            print(f'处理进度: [{i+1}/{len(input_img_list)}]')
        
        # 根据图像名称确定对应的json文件路径
        if "euler-exp" in img_name or "euler" in img_name or "exp" in img_name:
            prefix = img_name.split('_')[-1].split('.')[0]
            if prefix in EULER_AUG3D_PATH:
                euler_base_path = EULER_AUG3D_PATH[prefix]
                euler_path = os.path.join(euler_base_path, img_name.split('_')[1] + '.jpg.json')
                euler_info = parse_kuaishou_euler_bbox(euler_path)
        else:
            prefix = img_name.split('_')[0]
            if prefix in EULER_PATH_MAP:
                euler_base_path = EULER_PATH_MAP[prefix]
                euler_path = os.path.join(euler_base_path, os.path.splitext(img_name.split('_')[1])[0] + '.json')
                euler_info = parse_kuaishou_euler(euler_path)
        
        if euler_info is not None:
            if len(euler_info) == 3:
                roll, yaw, pitch = euler_info
            else:
                roll, yaw, pitch, _, _, _, _ = euler_info
            
            # roll_deg = math.degrees(roll) if abs(roll) > 2 * math.pi else roll
            # yaw_deg = math.degrees(yaw) if abs(yaw) > 2 * math.pi else yaw
            # pitch_deg = math.degrees(pitch) if abs(pitch) > 2 * math.pi else pitch
            
            yaw_deg = yaw * 180 / np.pi
            pitch_deg = pitch * 180 / np.pi
            roll_deg = roll * 180 / np.pi
            
            rolls.append(roll_deg)
            yaws.append(yaw_deg)
            pitches.append(pitch_deg)
            valid_count += 1
    
    print(f"成功解析 {valid_count} 个有效样本")
    return np.array(rolls), np.array(yaws), np.array(pitches)

def analyze_euler_angles(rolls, yaws, pitches):
    """分析欧拉角数据"""
    print("\n=== 欧拉角统计分析 ===")
    
    # 基本统计
    angles_data = {
        'Roll': rolls,
        'Yaw': yaws, 
        'Pitch': pitches
    }
    
    for name, data in angles_data.items():
        print(f"\n{name}角度统计:")
        print(f"  最小值: {np.min(data):.2f}°")
        print(f"  最大值: {np.max(data):.2f}°")
        print(f"  平均值: {np.mean(data):.2f}°")
        print(f"  标准差: {np.std(data):.2f}°")
        print(f"  中位数: {np.median(data):.2f}°")
    
    return angles_data

def create_histograms(angles_data, bin_width=5, save_dir="./euler_analysis_results"):
    """创建并保存直方图"""
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建主图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    # 存储详细的区间统计信息
    detailed_stats = {}
    
    for idx, (name, data) in enumerate(angles_data.items()):
        ax = axes[idx]
        
        # 计算合适的bins
        data_min = np.min(data)
        data_max = np.max(data)
        bins = np.arange(
            np.floor(data_min / bin_width) * bin_width,
            np.ceil(data_max / bin_width) * bin_width + bin_width,
            bin_width
        )
        
        # 绘制直方图
        n, bins, patches = ax.hist(data, bins=bins, alpha=0.7, color=colors[idx], 
                                  edgecolor='black', linewidth=0.5)
        
        # 设置图表属性
        ax.set_xlabel(f'{name}角度 (°)', fontsize=12)
        ax.set_ylabel('出现次数', fontsize=12)
        ax.set_title(f'{name}角度分布直方图\n(区间宽度: {bin_width}°)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息文本框
        stats_text = f'样本数: {len(data)}\n最小值: {np.min(data):.1f}°\n最大值: {np.max(data):.1f}°\n平均值: {np.mean(data):.1f}°'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
        
        # 旋转x轴标签以避免重叠
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # 保存详细的区间统计
        detailed_stats[name] = {
            'bins': bins,
            'counts': n,
            'statistics': {
                'count': len(data),
                'min': np.min(data),
                'max': np.max(data),
                'mean': np.mean(data),
                'std': np.std(data),
                'median': np.median(data)
            }
        }
    
    plt.tight_layout()
    
    # 保存主图
    main_plot_path = os.path.join(save_dir, "euler_angles_histograms.png")
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
    print(f"主直方图已保存: {main_plot_path}")
    plt.show()
    
    # 创建单独的子图并保存
    for name, data in angles_data.items():
        fig_single, ax_single = plt.subplots(figsize=(10, 6))
        
        data_min = np.min(data)
        data_max = np.max(data)
        bins = np.arange(
            np.floor(data_min / bin_width) * bin_width,
            np.ceil(data_max / bin_width) * bin_width + bin_width,
            bin_width
        )
        
        n, bins, patches = ax_single.hist(data, bins=bins, alpha=0.7, color=colors[list(angles_data.keys()).index(name)], 
                                         edgecolor='black', linewidth=0.5)
        
        ax_single.set_xlabel(f'{name}角度 (°)', fontsize=12)
        ax_single.set_ylabel('出现次数', fontsize=12)
        ax_single.set_title(f'{name}角度分布直方图', fontsize=14, fontweight='bold')
        ax_single.grid(True, alpha=0.3)
        
        stats_text = f'样本数: {len(data)}\n最小值: {np.min(data):.1f}°\n最大值: {np.max(data):.1f}°\n平均值: {np.mean(data):.1f}°\n标准差: {np.std(data):.1f}°'
        ax_single.text(0.02, 0.98, stats_text, transform=ax_single.transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      fontsize=10)
        
        single_plot_path = os.path.join(save_dir, f"{name.lower()}_histogram.png")
        plt.savefig(single_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig_single)
        print(f"单独直方图已保存: {single_plot_path}")
    
    return detailed_stats

def export_statistics_to_csv(angles_data, detailed_stats, save_dir="./euler_analysis_results"):
    """导出统计表格到CSV文件"""
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 导出基本统计信息
    basic_stats_data = []
    for name, data in angles_data.items():
        stats = detailed_stats[name]['statistics']
        basic_stats_data.append({
            '角度类型': name,
            '样本数量': stats['count'],
            '最小值 (°)': round(stats['min'], 2),
            '最大值 (°)': round(stats['max'], 2),
            '平均值 (°)': round(stats['mean'], 2),
            '标准差 (°)': round(stats['std'], 2),
            '中位数 (°)': round(stats['median'], 2)
        })
    
    basic_stats_df = pd.DataFrame(basic_stats_data)
    basic_stats_path = os.path.join(save_dir, "euler_angles_basic_statistics.csv")
    basic_stats_df.to_csv(basic_stats_path, index=False, encoding='utf-8-sig')
    print(f"基本统计信息已导出: {basic_stats_path}")
    
    # 2. 导出详细的区间统计信息
    for name in angles_data.keys():
        bin_stats = detailed_stats[name]
        bins = bin_stats['bins']
        counts = bin_stats['counts']
        
        interval_data = []
        for i in range(len(counts)):
            interval_data.append({
                '角度类型': name,
                '区间起始 (°)': round(bins[i], 2),
                '区间结束 (°)': round(bins[i+1], 2),
                '区间中心 (°)': round((bins[i] + bins[i+1]) / 2, 2),
                '样本数量': int(counts[i]),
                '占比 (%)': round(counts[i] / bin_stats['statistics']['count'] * 100, 2)
            })
        
        interval_df = pd.DataFrame(interval_data)
        interval_path = os.path.join(save_dir, f"{name.lower()}_interval_statistics.csv")
        interval_df.to_csv(interval_path, index=False, encoding='utf-8-sig')
        print(f"区间统计信息已导出: {interval_path}")
    
    # 3. 导出原始数据样本
    sample_data = []
    max_len = max(len(angles_data['Roll']), len(angles_data['Yaw']), len(angles_data['Pitch']))
    
    for i in range(min(1000, max_len)):  # 最多导出1000个样本
        sample_data.append({
            '样本索引': i,
            'Roll角度 (°)': round(angles_data['Roll'][i], 2) if i < len(angles_data['Roll']) else None,
            'Yaw角度 (°)': round(angles_data['Yaw'][i], 2) if i < len(angles_data['Yaw']) else None,
            'Pitch角度 (°)': round(angles_data['Pitch'][i], 2) if i < len(angles_data['Pitch']) else None
        })
    
    sample_df = pd.DataFrame(sample_data)
    sample_path = os.path.join(save_dir, "euler_angles_sample_data.csv")
    sample_df.to_csv(sample_path, index=False, encoding='utf-8-sig')
    print(f"样本数据已导出: {sample_path}")
    
    # 4. 创建汇总报告
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_report = f"""欧拉角统计分析报告
生成时间: {timestamp}
总样本数量: {len(angles_data['Roll'])}

基本统计摘要:
{basic_stats_df.to_string(index=False)}

分析完成!
"""
    
    report_path = os.path.join(save_dir, "analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary_report)
    print(f"分析报告已保存: {report_path}")

def main():
    # 设置输入路径和输出目录
    # input_path = "/data/xiaoshuai/facial_lanmark/val_1118"
    # output_dir = "./euler_analysis_val_results"
    input_path = "/data/xiaoshuai/facial_lanmark/train_1121"
    output_dir = "./euler_analysis_train_results"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有欧拉角数据
    print("开始收集欧拉角数据...")
    rolls, yaws, pitches = collect_euler_angles(input_path)
    
    if len(rolls) == 0:
        print("未找到有效的欧拉角数据！")
        return
    
    # 分析数据
    angles_data = analyze_euler_angles(rolls, yaws, pitches)
    
    # 创建并保存直方图
    print("\n生成直方图中...")
    detailed_stats = create_histograms(angles_data, bin_width=5, save_dir=output_dir)
    
    # 导出统计表格
    print("\n导出统计表格中...")
    export_statistics_to_csv(angles_data, detailed_stats, save_dir=output_dir)
    
    print(f"\n=== 分析完成 ===")
    print(f"所有结果已保存到: {os.path.abspath(output_dir)}")
    print("包含文件:")
    print("  - euler_angles_histograms.png (主直方图)")
    print("  - roll_histogram.png, yaw_histogram.png, pitch_histogram.png (单独直方图)")
    print("  - euler_angles_basic_statistics.csv (基本统计)")
    print("  - roll_interval_statistics.csv 等 (区间统计)")
    print("  - euler_angles_sample_data.csv (样本数据)")
    print("  - analysis_report.txt (分析报告)")

if __name__ == '__main__':
    main()