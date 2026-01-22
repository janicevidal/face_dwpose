import matplotlib.pyplot as plt
import numpy as np

# 定义所有模板
src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]], dtype=np.float32)

src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]], dtype=np.float32)

src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]], dtype=np.float32)

src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]], dtype=np.float32)

src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]], dtype=np.float32)

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

# 创建图像
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
templates = [src1, src2, src3, src4, src5, arcface_src]
titles = ['src1 (左侧)', 'src2 (正面偏左)', 'src3 (完全正面)', 
          'src4 (正面偏右)', 'src5 (右侧)', 'ArcFace标准模板']

# 点的标签和颜色
point_labels = ['左眼', '右眼', '鼻尖', '左嘴角', '右嘴角']
colors = ['red', 'green', 'blue', 'orange', 'purple']

# 绘制每个模板
for idx, (ax, points, title) in enumerate(zip(axes.flat, templates, titles)):
    # 设置图像范围
    ax.set_xlim(0, 112)
    ax.set_ylim(112, 0)  # 注意：y轴反向，因为图像坐标原点在左上角
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # 绘制点
    for i, (x, y) in enumerate(points):
        ax.scatter(x, y, color=colors[i], s=100, zorder=5)
        ax.text(x+1, y-2, point_labels[i], fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # 连接点形成人脸轮廓（按顺序连接）
        if i < 2:  # 连接双眼
            if i == 0:
                ax.plot([points[0][0], points[1][0]], 
                       [points[0][1], points[1][1]], 'k--', alpha=0.5)
        if i == 2:  # 从鼻尖到双眼
            ax.plot([points[2][0], points[0][0]], 
                   [points[2][1], points[0][1]], 'k--', alpha=0.3)
            ax.plot([points[2][0], points[1][0]], 
                   [points[2][1], points[1][1]], 'k--', alpha=0.3)
        if i == 3:  # 从左嘴角到鼻尖和右嘴角
            ax.plot([points[3][0], points[2][0]], 
                   [points[3][1], points[2][1]], 'k--', alpha=0.3)
            ax.plot([points[3][0], points[4][0]], 
                   [points[3][1], points[4][1]], 'k--', alpha=0.5)
    
    # 添加模板序号
    ax.text(5, 10, f'模板 {idx+1}', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # 显示坐标网格
    ax.set_xticks(np.arange(0, 113, 10))
    ax.set_yticks(np.arange(0, 113, 10))
    
    # 对于最后一行，添加x轴标签
    if idx >= 3:
        ax.set_xlabel('X坐标', fontsize=10)
    if idx % 3 == 0:
        ax.set_ylabel('Y坐标', fontsize=10)

plt.suptitle('人脸对齐关键点模板比较 (图像尺寸: 112×112)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("template_comparison.png")

# 额外创建一个汇总图，将所有模板叠加在一张图上比较
plt.figure(figsize=(10, 10))
plt.xlim(0, 112)
plt.ylim(112, 0)
plt.gca().set_aspect('equal')
plt.grid(True, alpha=0.3)
plt.title('所有模板关键点位置叠加对比', fontsize=14, fontweight='bold')

# 为每个模板使用不同的标记
markers = ['o', 's', '^', 'D', 'v', '*']
template_names = ['src1', 'src2', 'src3', 'src4', 'src5', 'ArcFace']

for idx, (points, name, marker) in enumerate(zip(templates, template_names, markers)):
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    plt.scatter(x_coords, y_coords, s=80, marker=marker, label=name, alpha=0.8)
    
    # 连接点（仅显示轮廓，不区分具体点）
    plt.plot(x_coords[[0, 1]], y_coords[[0, 1]], '--', alpha=0.3)  # 双眼连线
    plt.plot(x_coords[[3, 4]], y_coords[[3, 4]], '--', alpha=0.3)  # 嘴角连线

plt.xlabel('X坐标', fontsize=12)
plt.ylabel('Y坐标', fontsize=12)
plt.legend(loc='best')
plt.xticks(np.arange(0, 113, 10))
plt.yticks(np.arange(0, 113, 10))
plt.tight_layout()
# plt.show()
plt.savefig("template_comparison_all.png")

# 打印每个模板的统计信息
print("="*60)
print("关键点模板统计信息 (坐标范围在0-112之间):")
print("="*60)

for idx, (points, name) in enumerate(zip(templates, template_names)):
    print(f"\n{name}:")
    print(f"  左眼: ({points[0, 0]:.2f}, {points[0, 1]:.2f})")
    print(f"  右眼: ({points[1, 0]:.2f}, {points[1, 1]:.2f})")
    print(f"  鼻尖: ({points[2, 0]:.2f}, {points[2, 1]:.2f})")
    print(f"  左嘴角: ({points[3, 0]:.2f}, {points[3, 1]:.2f})")
    print(f"  右嘴角: ({points[4, 0]:.2f}, {points[4, 1]:.2f})")
    
    # 计算双眼距离
    eye_distance = np.sqrt((points[1, 0] - points[0, 0])**2 + (points[1, 1] - points[0, 1])**2)
    print(f"  双眼距离: {eye_distance:.2f} 像素")
    
    # 计算嘴巴宽度
    mouth_width = np.sqrt((points[4, 0] - points[3, 0])**2 + (points[4, 1] - points[3, 1])**2)
    print(f"  嘴巴宽度: {mouth_width:.2f} 像素")