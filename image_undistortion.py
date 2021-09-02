import argparse
import os
import cv2
import glob
import time
import numpy as np


def undistort(captures_savedir, mtx, dist, is_remapping):
    images = glob.glob(captures_savedir + '/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        h, w, _ = img.shape
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # alpha==1
        x, y, w, h = roi

        # undistort image
        if not is_remapping:
            # method 1
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            dst = dst[y:y+h, x:x+w]
            cv2.imshow("dst", dst)
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            # method 2
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
            dst2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            # crop the image
            dst = dst2[y:y+h, x:x+w]
            cv2.imshow("dst", dst)
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("captures_savedir", default="calibration_captures", type=str)
    parser.add_argument("results_savedir", default="calibration_results", type=str)
    parser.add_argument("--save_filename", default="calibration.yml")
    parser.add_argument("is_remapping", default=False, type=bool)
    args = parser.parse_args()

    # configuration from parser
    working_dir = os.getcwd()
    captures_dir = os.path.join(working_dir, args.captures_savedir)
    results_dir = os.path.join(working_dir, args.results_savedir)
    save_path = os.path.join(results_dir, args.save_filename)
    is_remapping = args.is_remapping

    # load calibration coefficient
    cv_file = cv2.FileStorage(save_path, cv2.FILE_STORAGE_READ)
    mtx = cv_file.getNode("K").mat()
    dist = cv_file.getNode("D").mat()
    cv_file.release()

    # undistort
    undistort(captures_dir, mtx, dist, is_remapping)
