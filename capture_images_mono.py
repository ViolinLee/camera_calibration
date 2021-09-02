import os
import cv2
import sys


def capture_mono(cam_number, capture_dir, prefix):
    capture_dir = os.path.join(os.getcwd(), capture_dir)
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)

    cap = cv2.VideoCapture(cam_number)

    cnt = 0
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', frame)
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            elif key == ord('c'):
                filename = prefix + str(cnt) + ".jpg"
                filepath = os.path.join(capture_dir, filename)
                cv2.imwrite(filepath, frame)
                print("Image saved:", filepath)

                cnt += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        # For example: python capture_images_mono.py 0 captures/mono_captures mono
        print("Usage: python capture_images_mono.py cam_number capture_dir prefix")
        sys.exit(1)

    cam_number = int(sys.argv[1])
    capture_dir = sys.argv[2]
    prefix = sys.argv[3]
    capture_mono(cam_number, capture_dir, prefix)
