import argparse
import datetime
import cv2

from scrfd_onnx import *
from prealign_landmark_onnx_infer import *
from landmark_track import GroupTrack

from flask import Flask, render_template, Response


class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(0)
        
        desired_width = 1280
        desired_height = 720

        # 设置分辨率
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

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
    detector = SCRFD(
        model_file=
        '/Users/user/Project/FacialLandmark/demo_web/onnx/scrfd_500m_bnkps_shape640x640.onnx')
    
    landmarker = PreAlignDWPOSE(
        model_file=
        '/Users/user/Project/FacialLandmark/demo_web/onnx/end2end_slim.onnx')
    
    # landmarker = PreAlignDWPOSE(
    #     model_file=
    #     '/Users/user/Project/FacialLandmark/demo_web/onnx/end2end_slim_simcc.onnx')
    
    detector.prepare()
    
    # box_matcher = SeqBoxMatcher()
    lmk_tracker_five = GroupTrack()
    lmk_tracker = GroupTrack()
    
    while True:
        frame = camera.get_frame()
        # cv2.imshow('fourcc', frame)

        ta = datetime.datetime.now()
        bboxes, kpss = detector.detect(frame, 0.5, input_size = (640, 640))

        # box match smooth    
        # new_boxes = box_matcher.update(bboxes[:, :4], bboxes[:, 4:])
        
        # landmark smooth
        kpss_five = lmk_tracker_five.calculate(frame, kpss)
        
        tb = datetime.datetime.now()
        print('all cost:', (tb - ta).total_seconds() * 1000)
        
        # draw aligned face
        # if kpss is not None:
        #     img_align = detector.get_align(frame, kpss)
        #     h_, w_, _ = frame.shape
        #     bottem = np.full([112, w_, 3], 255, dtype=np.uint8)
        #     frame = np.vstack((frame, bottem))
            
        #     max_draw_num = int(w_ / 112)
                
        #     for i in range(min(len(img_align), max_draw_num)):
        #         aimg = img_align[i]
                
        #         pos = 112 * i
        #         frame[h_:(h_ + 112), pos:(pos + 112)] = aimg
        
        pts_235 = []            
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            landms = kpss_five[i]
            
            four_landmarks = np.asarray(
                [landms[0][0], landms[0][1], landms[1][0], landms[1][1], landms[3][0], landms[3][1], landms[4][0], landms[4][1]],
                dtype=np.float32
            )
            
            # kpss,_ = landmarker.infer(frame, four_landmarks, head="simcc")
            kpss,_ = landmarker.infer(frame, four_landmarks, head="ipr")
            
            pts_235.append(kpss[0])
        
        
        kpss_final = lmk_tracker.calculate(frame, np.array(pts_235))
        
        for i in range(kpss_final.shape[0]):
            kps = kpss_final[i]

            for kp in kps:
                # kp_ = kp.astype(np.int32)
                kp_ = np.around(kp).astype(np.int32)
                cv2.circle(frame, tuple(kp_), 2, (0, 0, 255), 2)
                    
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
    parser = argparse.ArgumentParser(description='Face Landmark Detection')
    parser.add_argument('--model', type=str, default='ssfd')
    args = parser.parse_args()
    model = args.model
    app.run(host='0.0.0.0', debug=True, port=8008)