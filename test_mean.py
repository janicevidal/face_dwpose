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

print("pupil points in mean_face:")
idx = [207, 201, 215]
points1 = mean_face[idx].astype(np.float32) * 96.0
print(points1)

idx = [223, 202, 231]
points2 = mean_face[idx].astype(np.float32) * 96.0
print(points2)

idx = list(range(203, 219, 1))
points3 = mean_face[idx].astype(np.float32) * 96.0
print(points3)

idx = list(range(219, 235, 1))
points4 = mean_face[idx].astype(np.float32) * 96.0
print(points4)

import pdb; pdb.set_trace()


# idx = list(range(0, 37, 2))+ [141, 159, 201, 202]
idx = list(range(0, 37, 2))+ [201, 202, 141, 159]
# idx = list(range(0, 37, 2))

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

# expanded_landmarks = np.concatenate([x_coords, y_coords])


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
#     print(f"{value:}f", end="")
    print(f"{value:.2f}", end=" ")  
    if i < len(expanded_landmarks) - 1:
        print(", ", end="")
print("};")




avgFaceSdAlign181 = 25.973375
avgFaceMeanAlign181 = np.array([63.607624, 86.27017])

avgFaceAlign181 = np.array([
    [1.6210643, -0.9600147],    # 6 point in 181 points
    [1.5416889, -0.1931234],    # 8 point in 181 points
    [1.2599964,  0.5209421],    # 10 point in 181 points
    [0.7298336,  1.0626760],    # 12 point in 181 points
    [-0.0157396,  1.3538444],   # 14 point in 181 points
    [-0.7354493,  1.0527896],   # 16 point in 181 points
    [-1.2433237,  0.5107294],   # 18 point in 181 points
    [-1.5216014, -0.1959554],   # 20 point in 181 points
    [-1.6066356, -0.9556512],   # 22 point in 181 points
    [-0.7304242, -1.3178347],   # 120 point in 181 points
    [0.7182326, -1.3200892],    # 125 point in 181 points
    [-0.6219274,  0.2229699],   # 143 point in 181 points
    [0.6042836,  0.2187181]     # 145 point in 181 points
])

std9points = np.array([
    [23.379078, 37.308876],     # 24 on 181
    [27.409477, 23.992020],     # 25 on 181 
    [35.637268, 13.367502],     # 26 on 181         
    [48.479927, 7.4739660],     # 27 on 181 
    [63.471820, 5.8822320],     # 0 on 181 
    [78.601040, 7.555958],     # 1 on 181 
    [91.603660, 13.559926],     # 2 on 181 
    [100.03552, 24.186983],     # 3 on 181 
    [104.019104,  37.309320]    # 4 on 181 
])


avgFaceAlign181_ = avgFaceAlign181.astype(np.float32) * avgFaceSdAlign181 + avgFaceMeanAlign181
print(avgFaceAlign181_)

all_points = np.concatenate([avgFaceAlign181_, std9points], axis=0)

blank_image = np.ones((128, 128, 3), dtype=np.uint8) * 255

import cv2    
for i, (x, y) in enumerate(all_points):
    cv2.circle(blank_image, (int(x+0.5), int(y+0.5)), 1, (0, 0, 0), 0)
    
    cv2.imwrite('test.png', blank_image)

# [[105.71213526  61.33534792]
#  [103.65048975  81.25410345]
#  [ 96.33398432  99.80079439]
#  [ 82.56386561 113.87145212]
#  [ 63.19881347 121.43407991]
#  [ 44.50552363 113.61466768]
#  [ 31.31431193  99.5355356 ]
#  [ 24.08649821  81.18054703]
#  [ 21.87787623  61.44868257]
#  [ 44.63604159  52.04155519]
#  [ 82.26254839  51.98299578]
#  [ 47.45407099  92.06145122]
#  [ 79.30290789  91.95101717]]