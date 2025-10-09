
import os 
import numpy as np

from utils import load_test_data


# -0.0151

def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark 
    """
    N = preds.shape[0]
    L = preds.shape[1]

    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = 34  # meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        elif L == 235:
            interocular = np.linalg.norm(pts_gt[201, ] - pts_gt[202, ])
            # interocular = np.linalg.norm(pts_gt[77, ] - pts_gt[113, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt,
                                        axis=1)) / (interocular * L)

    return rmse


def eval(pred_folder_names):
    _, gt_npy_files = load_test_data()

    for pred_folder_name in pred_folder_names:
        print(f"Processing {pred_folder_name} ...")

        pred_dir = os.path.join("/data/xiaoshuai/facial_lanmark/vis", pred_folder_name)
        # pred_dir = os.path.join("/home/zhangxiaoshuai/Project/video-align235-deploy/results", pred_folder_name)
        
        no_predfs = []

        dists = []
        dists_gt = []

        for gt_npy_file in gt_npy_files:
            bname = os.path.basename(gt_npy_file)
            nname = os.path.splitext(bname)[0]

            pred_npyf_name = nname + ".jpg.npy"
            pred_npyf = os.path.join(pred_dir, pred_npyf_name)

            if not os.path.exists(pred_npyf):
                no_predfs.append(gt_npy_file)
                continue

            infodict = np.load(gt_npy_file, allow_pickle=True).tolist()
            gt_landmarks235 = infodict["landmarks208"]  # FIXME 实际是235个点

            pred_landmarks235 = np.load(pred_npyf, allow_pickle=True)
            dists.append(pred_landmarks235)
            dists_gt.append(gt_landmarks235)

        dists = np.array(dists)
        dists_gt = np.array(dists_gt)
        nme = compute_nme(dists, dists_gt)

        if "flip" in pred_folder_name:
            print(f"{pred_folder_name} mean dist:", np.array(nme).mean())
        else:
            print(f"{pred_folder_name} mean dist:", np.array(nme).mean())
        print("no pred files:", no_predfs)
        print("\n\n")


if __name__ == "__main__":

    # eval(["test"])  # zxs 0.0363(mnn) 0.0340(onnx)
    # eval(["test_ipr"])  # zxs_ipr 0.031686 (onnx)
    # eval(["test_ipr_finetune"])  # zxs_ipr 0.031434 (onnx)
    # eval(["test_ipr_beta_finetune"])  # zxs_ipr 0.031421 (onnx)
    eval(["test_ipr_beta_scalenorm_finetune_420"])  # zxs_ipr 0.031323 (onnx)
    eval(["test_ipr_beta_scalenorm_finetune"])  # zxs_ipr 0.031342 (onnx)
    eval(["test_ipr_beta_finetune_420"])  # zxs_ipr 0.031382 (onnx)
    # eval(["test_ipr_358"])  # zxs_ipr 0.03172(onnx)
    # eval(["test_two"])  # zxs 0.0373
    
    # eval(["ghostnetv2_dot_25x"]) # 0.0536
    # eval(["ghostnetv2_dot_50x"])  # 0.0465
    # eval(["ghostnetv2_dot_75x"])  # 0.0432
    # eval(["pts235-ghostv2-new"])  # 0.02998