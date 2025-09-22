
import os
import json
import numpy as np

from glob import glob

ALIGN_POINT_NUM = 235
MIN_FACE_SIZE = 40


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep




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



# 检查数据是否可用于训练 过滤掉错误的文件、不满足条件的
def check_data(image_file, npy_file, euler_file=None):
    
    # 文件不存在
    if not os.path.exists(image_file) or not os.path.exists(npy_file):
        return False
    
    if euler_file is not None:
        if not os.path.exists(euler_file):
            return False
        
        eulerInfo = parse_kuaishou_euler(euler_file)
        if eulerInfo is None:
            return False

    infodict = np.load(npy_file, allow_pickle=True).tolist()
    landmarks235 = infodict["landmarks208"]
    if len(landmarks235) != ALIGN_POINT_NUM:
        return False

    bboxes = infodict["DFSD_facebbox"]
    if bboxes.shape[0] != 1:
        return False
    
    gtbox = bboxes[0]
    if gtbox is None:
        return False

    # 判断该图人脸大小是否合适 MIN_FACE_SIZE=40
    _, _, w, h = gtbox
    if min(h, w) <= MIN_FACE_SIZE:
        return False

    return True




# 加载图片数据 返回路径
def load_test_data():
    DATA_ROOT = "/data/caiachang/video-ldms-ok/TEST"
    FOLDERS = ["ffhq", "only1face", "celeba"]

    # 遍历所有文件
    image_files = []
    npy_files = []


    for folder in FOLDERS:
        dataset_root = os.path.join(DATA_ROOT, folder)

        crt_image_files = sorted(glob(os.path.join(dataset_root, "*.jpg")))
        crt_npy_files = []

        # 仍然检查其他文件必须存在
        for imgf in crt_image_files:
            bname = os.path.basename(imgf)
            nname = os.path.splitext(bname)[0]

            npyf = os.path.join(dataset_root, nname + ".npy")
            if not os.path.exists(npyf):
                print(f"imgf={imgf}, npy file not exist: {npyf}")
                quit()

            crt_npy_files.append(npyf)

        assert len(crt_npy_files) == len(crt_image_files)

        image_files.extend(crt_image_files)
        npy_files.extend(crt_npy_files)

    assert len(image_files) == len(npy_files)

    # TODO 数据检查 剔除不满足条件的
    image_files_checked = []
    npy_files_checked = []
    for imgf, npyf in zip(image_files, npy_files):
        if check_data(imgf, npyf):
            image_files_checked.append(imgf)
            npy_files_checked.append(npyf)

    return image_files_checked, npy_files_checked


