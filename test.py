import numpy as np

def get_affine_transform_ls(points_from, points_to):
    """最小二乘法（C++风格的实现）"""
    num_point = len(points_from)
    ma = np.zeros((4, 4), dtype=np.float32)
    mb = np.zeros(4, dtype=np.float32)
    
    for i in range(num_point):
        x_from, y_from = points_from[i]
        x_to, y_to = points_to[i]
        
        ma[0, 0] += x_from * x_from + y_from * y_from
        ma[0, 2] += x_from
        ma[0, 3] += y_from
        
        mb[0] += x_from * x_to + y_from * y_to
        mb[1] += x_from * y_to - y_from * x_to
        mb[2] += x_to
        mb[3] += y_to
    
    # 设置对称元素
    ma[1, 1] = ma[0, 0]
    ma[2, 1] = ma[1, 2] = -ma[0, 3]
    ma[3, 1] = ma[1, 3] = ma[2, 0] = ma[0, 2]
    ma[2, 2] = ma[3, 3] = float(num_point)
    ma[3, 0] = ma[0, 3]
    
    # 解线性方程组
    mm = np.linalg.solve(ma, mb)
    
    # 构建2x3变换矩阵
    tm = np.array([
        [mm[0], -mm[1], mm[2]],
        [mm[1], mm[0], mm[3]]
    ], dtype=np.float32)
    
    return tm

def transformation_from_points(points1, points2):
    """均值+SVD方法"""
    points1 = points1.astype(np.float32)
    points2 = points2.astype(np.float32)
    
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1_centered = points1 - c1
    points2_centered = points2 - c2
    
    s1 = np.std(points1_centered)
    s2 = np.std(points2_centered)
    points1_normalized = points1_centered / s1
    points2_normalized = points2_centered / s2
    
    # 计算旋转矩阵
    cov_matrix = points1_normalized.T @ points2_normalized
    U, S, Vt = np.linalg.svd(cov_matrix)
    
    # 确保行列式为正值（防止反射）
    if np.linalg.det(U @ Vt) < 0:
        Vt[-1, :] *= -1
    
    R = U @ Vt
    
    # 恢复缩放和平移
    M = (s2 / s1) * R
    B = c2.reshape(2, 1) - M @ c1.reshape(2, 1)
    
    # 组合成2x3矩阵
    affine_matrix = np.hstack((M, B))
    return affine_matrix

# 测试数据
# 生成一些测试点
np.random.seed(26)
points_from = np.random.randn(10, 2).astype(np.float32)

# 应用一个已知的相似变换
theta = np.pi / 6  # 30度
scale = 1.5
tx, ty = 10.0, 5.0

# 相似变换矩阵
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])
M = scale * R
t = np.array([[tx], [ty]])

# 生成目标点
points_to = (M @ points_from.T + t).T

# 计算变换矩阵
tm_ls = get_affine_transform_ls(points_from, points_to)
tm_svd = transformation_from_points(points_from, points_to)

print("最小二乘法结果:")
print(tm_ls)
print("\n均值+SVD方法结果:")
print(tm_svd)
print("\n差异:")
print(np.abs(tm_ls - tm_svd))