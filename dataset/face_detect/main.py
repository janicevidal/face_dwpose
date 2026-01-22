import argparse
import cv2
import os
import time
from timer import Timer

from nets.nn import FaceDetector

clock = Timer()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_id', default='', help='Camera ID or image/video file path')
    parser.add_argument('--model', default='weights/SCRFD_500M.onnx', help='Model file path')
    parser.add_argument('--output_dir', default='./results', help='Output directory to save results')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    detector = FaceDetector(onnx_file=args.model)
    detector.prepare(-1)

    is_image = os.path.isfile(args.cam_id) and args.cam_id.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    # clock = Timer()

    if is_image:
        # Load image
        frame = cv2.imread(args.cam_id)
        if frame is None:
            print(f"Error loading image file: {args.cam_id}")
            return

        start = time.time()
        # clock.tic()
        processed_frame = process_frame(frame, detector)
        # clock.toc()
        print('Thời gian xử lý: ', time.time()-start)
        # print('Thời gian xử lý TB: ', clock.average_time)

        output_path = os.path.join(args.output_dir, os.path.basename(args.cam_id))
        cv2.imwrite(output_path, processed_frame)
        print(f"Processed image saved to {output_path}")
    else:
        # Open video stream
        cam_id = int(args.cam_id) if args.cam_id.isdigit() else args.cam_id
        stream = cv2.VideoCapture(args.cam_id)

        if not stream.isOpened():
            print("Error opening video stream or file")
            return

        width = stream.get(cv2.CAP_PROP_FRAME_WIDTH)  # width of input frame
        height = stream.get(cv2.CAP_PROP_FRAME_HEIGHT)  # height of input frame
        fps = stream.get(cv2.CAP_PROP_FPS)  # fps of video

        if cam_id is not None:
            output_video_path = os.path.join(args.output_dir, 'video.mp4')
        else:
            output_video_path = os.path.join(args.output_dir, os.path.basename(args.cam_id))
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

        while True:
            success, frame = stream.read()
            if not success:
                break

            start = time.time()
            clock.tic()
            processed_frame = process_frame(frame, detector)
            clock.toc()
            print('Thời gian xử lý: ', time.time() - start)
            print('Thời gian xử lý TB: ', clock.average_time)
            out.write(processed_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        stream.release()
        cv2.destroyAllWindows()


def process_frame(frame, detector):
    clock.tic()
    # boxes, _ = detector.detect(frame, input_size=(512, 512))
    boxes, _ = detector.detect(frame, input_size=(640, 640))
    clock.toc()
    print('Thời gian xử lý TB: ', clock.average_time)
    boxes = boxes.astype('int32')
    for box in boxes:
        x_min, y_min, x_max, y_max, _ = box
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 255), 1)

        cv2.line(frame, (int(x_min), int(y_min)), (int(x_min + 15), int(y_min)), (255, 0, 255), 3)
        cv2.line(frame, (int(x_min), int(y_min)), (int(x_min), int(y_min + 15)), (255, 0, 255), 3)

        cv2.line(frame, (int(x_max), int(y_max)), (int(x_max - 15), int(y_max)), (255, 0, 255), 3)
        cv2.line(frame, (int(x_max), int(y_max)), (int(x_max), int(y_max - 15)), (255, 0, 255), 3)

        cv2.line(frame, (int(x_max - 15), int(y_min)), (int(x_max), int(y_min)), (255, 0, 255), 3)
        cv2.line(frame, (int(x_max), int(y_min)), (int(x_max), int(y_min + 15)), (255, 0, 255), 3)

        cv2.line(frame, (int(x_min), int(y_max - 15)), (int(x_min), int(y_max)), (255, 0, 255), 3)
        cv2.line(frame, (int(x_min), int(y_max)), (int(x_min + 15), int(y_max)), (255, 0, 255), 3)

    # cv2.imshow('Video', frame)
    # cv2.waitKey(0)
    return frame


if __name__ == '__main__':
    main()