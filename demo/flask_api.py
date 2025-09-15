import argparse
import datetime
import cv2

from ssfd_ort_infer import *
from landmark_track import GroupTrack

from flask import Flask, render_template, Response


class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        success, image = self.video.read()
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        # ret, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()
        return image

app = Flask(__name__)

@app.route('/')  # 主页
def index():
    # 具体格式保存在index.html文件中
    return render_template('index.html')

def ssfd(camera):
    detector = SSFD(
        model_file=
        'onnx/scrfd_lite_320_kpts_tood_resume_tal_anchor_mod_shape192x320.onnx')
    
    detector.prepare(-1)
    
    box_matcher = SeqBoxMatcher()
    lmk_tracker = GroupTrack()
    
    while True:
        frame = camera.get_frame()
        # cv2.imshow('fourcc', frame)

        ta = datetime.datetime.now()
        bboxes, kpss = detector.detect(frame, 0.5, input_size = (320, 192))

        # box match smooth    
        new_boxes = box_matcher.update(bboxes[:, :4], bboxes[:, 4:])
        
        # landmark smooth
        kpss = lmk_tracker.calculate(frame, kpss)
        
        tb = datetime.datetime.now()
        print('all cost:', (tb - ta).total_seconds() * 1000)
        
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
                    
        for i in range(new_boxes.shape[0]):
            bbox = new_boxes[i]
            x1, y1, x2, y2, _, freq = bbox.astype(np.int32)
            
            if kpss is not None:
                kps = kpss[i]
                
                # draw kps
                for kp in kps:
                    kp_ = kp.astype(np.int32)
                    cv2.circle(frame, tuple(kp_), 2, (0, 0, 255), 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            label_text = f'{bbox[-2]:.02f}'
            ret, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - ret[1] - baseline), (x1 + ret[0], y1), (255, 255, 255), -1)
            cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    if model == 'ssfd':
        return Response(ssfd(VideoCamera()),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Detection')
    parser.add_argument('--model', type=str, default='ssfd')
    args = parser.parse_args()
    model = args.model
    app.run(host='0.0.0.0', debug=True, port=9009)