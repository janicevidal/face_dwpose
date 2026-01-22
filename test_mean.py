import numpy as np

points1 = np.array([[34.3054, 30.7081], [61.1536, 30.7081], [36.2007, 60.3844], [59.2562, 60.3844]], dtype=np.float32)


points1 = points1.astype(np.float32)  # shape (4, 2)

c1 = np.mean(points1, axis=0)  # shape (1, 2)
print(c1)
points1 -= c1

s1 = np.std(points1)  # single value #standard deviation

print(s1)

points1 /= s1
print(points1)

# path = '/data/xiaoshuai/facial_lanmark/train_1226/mean_face_20_d/mean_landmarks.npy'
path = '/data/xiaoshuai/facial_lanmark/train_1226/mean_face_20_d/mean_face_symmetric_centered.npy'
mean_face = np.load(path)

mean_face = mean_face.astype(np.float32)

# idx = list(range(0, 37, 2))+ [141, 159, 201, 202]
idx = list(range(0, 37, 2))+ [201, 202, 141, 159]

points1 = mean_face[idx].astype(np.float32) * 96.0

c1 = np.mean(points1, axis=0)  # shape (1, 2)
print(c1)
points1 -= c1

s1 = np.std(points1)  # single value #standard deviation

print(s1)

points1 /= s1
print(points1)


idx = [201, 202, 141, 159]

points1 = mean_face[idx].astype(np.float32) * 96.0

c1 = np.mean(points1, axis=0)  # shape (1, 2)
print(c1)
points1 -= c1

s1 = np.std(points1)  # single value #standard deviation

print(s1)

points1 /= s1
print(points1)



mean_face_96 = mean_face.astype(np.float32) * 96.0

x_coords = mean_face_96[:, 0]  # 所有x坐标
y_coords = mean_face_96[:, 1]  # 所有y坐标

expanded_landmarks = np.concatenate([x_coords, y_coords])



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
    
print(square_mean_box)

mean_box_center = np.mean(square_mean_box, axis=0)
print(mean_box_center)

mean_box_size = np.max([square_mean_box[1][0] - square_mean_box[0][0], square_mean_box[2][1] - square_mean_box[0][1]])
print(mean_box_size)





mean_face_96_5 = mean_face[[201, 202, 141, 159]].astype(np.float32) * 96.0

x_coords = mean_face_96_5[:, 0]  # 所有x坐标
y_coords = mean_face_96_5[:, 1]  # 所有y坐标

expanded_landmarks = np.concatenate([x_coords, y_coords])


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
    
print(square_mean_box)

mean_box_center = np.mean(square_mean_box, axis=0)
print(mean_box_center)

mean_box_size = np.max([square_mean_box[1][0] - square_mean_box[0][0], square_mean_box[2][1] - square_mean_box[0][1]])
print(mean_box_size)



print("float mean_landmarks[470] = {", end="")
for i, value in enumerate(expanded_landmarks):
    print(f"{value:}f", end="")
    if i < len(expanded_landmarks) - 1:
        print(", ", end="")
print("};")