import time
import cv2
import numpy as np
import MNN

from face_align import norm_crop
from landmark_track import GroupTrack

class SeqBoxMatcher(object):
    """Late fusion for video object detection"""

    def __init__(
        self,
        iou_thresh: float = 0.5,
        semantic_thresh: float = 0.1,
        seq_thresh: int = 2,
    ) -> None:
        super(SeqBoxMatcher, self).__init__()

        self.iou_thresh = iou_thresh
        self.semantic_thresh = semantic_thresh
        self.seq_thresh = seq_thresh
        self.period = 500
        self.reset()

    def reset(
        self,
    ) -> None:
        self.reference = []

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> list:
        def matrix_iou(a, b):
            """return iou of a and b, numpy version for data augenmentation"""

            lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
            rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
            area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
            area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
            area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
            return area_i / (area_a[:, np.newaxis] + area_b - area_i)

        new_boxes = np.hstack((boxes, scores, np.ones((boxes.shape[0], 1)))).astype(
            np.float32
        )

        if new_boxes.shape[0] == 0:
            self.reset()
            return new_boxes

        if len(self.reference) > 0:
            iou_matrix = matrix_iou(new_boxes[:, :4], self.reference[:, :4])
            iou_matrix_ = iou_matrix
            
            iou_matrix[iou_matrix < self.iou_thresh] = 0
            prob_matrix = np.dot(new_boxes[:, 4:-1], self.reference[:, 4:-1].T)
            prob_matrix[prob_matrix < self.semantic_thresh] = 0
            sim_matrix = iou_matrix * prob_matrix
            pairs = {}
            for index in sim_matrix.reshape(-1).argsort()[::-1]:
                i, j = np.unravel_index(index, sim_matrix.shape)
                if sim_matrix[i, j] == 0:
                    break
                if (i in pairs) or (j in list(pairs.values())):
                    continue
                pairs[i] = j
            for i, j in pairs.items():
                numb = self.reference[j, -1] % self.period
                
                new_boxes[i, 4:-1] = np.add(
                    np.dot(
                        self.reference[j, 4:-1],
                        numb / (numb + 1),
                    ),
                    np.dot(new_boxes[i, 4:-1], 1 / (numb + 1)),
                )
                new_boxes[i, -1] += self.reference[j, -1]
                
                # smooth box
                alpha = iou_matrix_[i][j] * iou_matrix_[i][j] * iou_matrix_[i][j] * iou_matrix_[i][j]
                new_boxes[i,:4] = alpha * self.reference[j,:4] + (1 - alpha) * new_boxes[i,:4]
                
                # alpha = 0.3
                # new_boxes[i,:4] = alpha * new_boxes[i,:4] + (1 - alpha) * self.reference[j,:4]
        
        # fliter boxes
        keep = np.ones(new_boxes.shape[0], dtype=bool)
        for i, val in enumerate(new_boxes):
            if (val[-2] < 0.7 and val[-1] < self.seq_thresh) or val[-2] < 0.55:
                keep[i] = False
        
        self.reference = new_boxes[keep, ]

        return self.reference

    
def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.


    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def MNNDataType2NumpyDataType(data_type):
    if data_type == MNN.Halide_Type_Uint8:
        return np.uint8
    elif data_type == MNN.Halide_Type_Double:
        return np.float64
    elif data_type == MNN.Halide_Type_Int:
        return np.int32
    elif data_type == MNN.Halide_Type_Int64:
        return np.int64
    else:
        return np.float32


def createTensor(tensor):
    shape = tensor.getShape()
    data_type = tensor.getDataType()
    dtype = MNNDataType2NumpyDataType(data_type)
    data = np.ones(shape, dtype=dtype)
    return MNN.Tensor(shape, tensor.getDataType(), data, tensor.getDimensionType())
    
    
class SSFD:
    def __init__(self, model_file=None, session=None, nms_thresh=0.4):
        self.model_file = model_file
        self.session = session
        self.taskname = 'detection'
        self.batched = False
        if self.session is None:
            config = {}
            config['precision'] = 'low'
            # config['backend'] = "OPENCL"
            # self.net.setCacheFile(".cachefile")
            self.net = MNN.Interpreter(self.model_file)
            self.session = self.net.createSession(config)
        self.center_cache = {}
        self.nms_thresh = nms_thresh
        self._init_vars()

    def _init_vars(self):

        self.input_size = None
        self.use_kps = False
        self._num_anchors = 1

        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

    def prepare(self, **kwargs):
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print(
                    'warning: det_size is already set in scrfd model, ignore')
            else:
                self.input_size = input_size

    def forward(self, img, thresh):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / 128,
            input_size,
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        
        # input
        inputTensor = self.net.getSessionInput(self.session)
        
        if (inputTensor.getShape()[-1] != input_size[0] or inputTensor.getShape()[-2] != input_size[1]):
            self.net.resizeTensor(inputTensor, (1, 3, input_size[1], input_size[0]))
            self.net.resizeSession(self.session)

        inputHost = MNN.Tensor(inputTensor.getShape(), inputTensor.getDataType(), blob, inputTensor.getDimensionType())
        inputTensor.copyFrom(inputHost)
        
        # infer
        self.net.runSession(self.session)
        outputTensorAll = self.net.getSessionOutputAll(self.session)
        
        # for key in outputTensorAll.keys():
        #     print(key, "shape:", outputTensorAll[key].getShape())

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        # output
        for stride in self._feat_stride_fpn:
            height = (-1) * (((-1) * input_height) // stride)  # Round up
            width = (-1) * (((-1) * input_width) // stride)  # Round up
            K = height * width
            
            score_oname = 'score_' + str(stride)
            bbox_oname  = 'bbox_' + str(stride)
            
            score_tensor = outputTensorAll[score_oname]
            output_host = createTensor(score_tensor)
            score_tensor.copyToHostTensor(output_host)
            score_numpy = output_host.getNumpyData()
            
            bbox_tensor = outputTensorAll[bbox_oname]
            output_host = createTensor(bbox_tensor)
            bbox_tensor.copyToHostTensor(output_host)
            bbox_numpy = output_host.getNumpyData()
            
            scores = (score_numpy.transpose(0, 2, 3, 1)).reshape(1, K * self._num_anchors, 1)[0]
            bbox_preds = (bbox_numpy.transpose(0, 2, 3, 1)).reshape(1, K * self._num_anchors, 4)[0]
            bbox_preds = bbox_preds * stride
            
            if self.use_kps:
                kps_oname  = 'kps_' + str(stride)
            
                kps_tensor = outputTensorAll[kps_oname]
                output_host = createTensor(kps_tensor)
                kps_tensor.copyToHostTensor(output_host)
                kps_numpy = output_host.getNumpyData()
            
                kps_preds =(kps_numpy.transpose(0, 2, 3, 1)).reshape(1, K * self._num_anchors, 10)[0] * stride

            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1],
                                          axis=-1).astype(np.float32)
                
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] *
                                              self._num_anchors,
                                              axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self,
               img,
               thresh=0.5,
               input_size=None,
               max_num=0,
               metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
            
        # print(img.shape[0], img.shape[1], new_width, new_height)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        
        # cv2.imwrite('test.jpg', det_img)

        scores_list, bboxes_list, kpss_list = self.forward(det_img, thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
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

    def get_align(self, image, kpss):
        """
        从图像中生成align后的人脸图像
        :param image: nparray, 原始图
        :param kpss:  人脸关键点坐标列表
        :return: aligned 人脸 112x112
        """
        aligns = []
        for pts in kpss:
            align = norm_crop(image, pts)  # 得到112x112的对齐图像
            aligns.append(align)
        return aligns


if __name__ == '__main__':
    detector = SSFD(
        model_file=
        'mnn_vsproject/mnn_face_detect/model/SSFD_lite_320x192_kpts.mnn')
        
    detector.prepare()

    cap = cv2.VideoCapture(0)
    
    box_matcher = SeqBoxMatcher()
    lmk_tracker = GroupTrack()
    
    print('Press "Esc", "q" or "Q" to exit.')

    while True:
        # Capture read
        ret, frame = cap.read()
        
        if not ret:
            break

        start_time = time.time()

        # bboxes, kpss = detector.detect(frame, 0.5, input_size=(256, 160))
        bboxes, kpss = detector.detect(frame, 0.5, input_size=(320, 192))
        
        # box match smooth    
        new_boxes = box_matcher.update(bboxes[:, :4], bboxes[:, 4:])
        
        # landmark smooth
        kpss = lmk_tracker.calculate(frame, kpss)
        
        elapsed_time = time.time() - start_time
        
        # print(elapsed_time)
        
        # draw aligned face
        if kpss is not None:
            img_align = detector.get_align(frame, kpss)
            h_, w_, _ = frame.shape
            bottem = np.full([112, w_, 3], 255, dtype=np.uint8)
            frame = np.vstack((frame, bottem))
            
            max_draw_num = int(w_ / 112)
                
            for i in range(min(len(img_align), max_draw_num)):
                aimg = img_align[i]
                
                pos = 112 * i
                frame[h_:(h_ + 112), pos:(pos + 112)] = aimg

        # draw boxes kpts
        for i in range(new_boxes.shape[0]):
            bbox = new_boxes[i]
            x1, y1, x2, y2, _, freq = bbox.astype(np.int32)
            
            if kpss is not None:
                kps = kpss[i]

                # align = norm_crop(frame, kps.reshape((5, 2)))
                # cv2.imshow('align ' + str(i), align)
                
                # draw kps
                for kp in kps:
                    kp_ = kp.astype(np.int32)
                    cv2.circle(frame, tuple(kp_), 2, (0, 0, 255), 2)
             
            # draw box score
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            label_text = f'{bbox[-2]:.02f}'
            ret, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - ret[1] - baseline), (x1 + ret[0], y1), (255, 255, 255), -1)
            cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
          
        # Inference elapsed time
        cv2.putText(
            frame,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
        
        cv2.imshow('SSFD', frame)
        
    cap.release()
    cv2.destroyAllWindows()