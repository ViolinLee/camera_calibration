import os
import cv2
import sys


def capture_stereo(left_cam_number, right_cam_number, left_capture_dir, right_capture_dir):
    left_capture_dir = os.path.join(os.getcwd(), left_capture_dir)
    if not os.path.exists(left_capture_dir):
        os.mkdir(left_capture_dir)

    right_capture_dir = os.path.join(os.getcwd(), right_capture_dir)
    if not os.path.exists(right_capture_dir):
        os.mkdir(right_capture_dir)

    cap_left = cv2.VideoCapture(left_cam_number)
    cap_right = cv2.VideoCapture(right_cam_number)

    cnt = 0
    while True:
        if not (cap_left.grab() and cap_right.grab()):
            break

        left_ret, left_frame = cap_left.retrieve()
        right_ret, right_frame = cap_right.retrieve()
        if left_ret and right_ret:
            cv2.imshow('capL', left_frame)
            cv2.imshow('capR', right_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                left_filename = "left" + str(cnt) + ".jpg"
                left_filepath = os.path.join(left_capture_dir, left_filename)
                cv2.imwrite(left_filepath, left_frame)
                print("Left Image saved: ", left_filepath)

                right_filename = "right" + str(cnt) + ".jpg"
                right_filepath = os.path.join(right_capture_dir, right_filename)
                cv2.imwrite(right_filepath, right_frame)
                print("Right Image saved: ", right_filepath)

        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python capture_images_mono.py cam_number capture_dir prefix")
        sys.exit(1)
