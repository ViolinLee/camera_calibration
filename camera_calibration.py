# https://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html

import argparse
import os
import cv2
import glob
import numpy as np


def calibrate(captures_savedir, squre_size, p_height, p_width):
    # terminate criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points on real patterns, like (0,0,0), (1,0,0)...,(6,5,0)
    objp = np.zeros((p_height*p_width, 3), np.float32)
    objp[:, :, 2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * squre_size

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(captures_savedir + '/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (p_height, p_width), None)
        # If found, add object pints, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (p_height, p_width), corners2, ret)
            cv2.imshow("img", img)

            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()

    """Starting Calibration"""
    # return value (boolean), camera matrix, distortion coefficients, rotation and translation vectors
    gray = cv2.imread(images[0])
    ret, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("camera matrix: ", cam_mtx)
    print("distortion coefficients: ", dist)
    print("rotation vectors: ", rvecs)
    print("translation vectors: ", tvecs)

    return ret, cam_mtx, dist, rvecs, tvecs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures_savedir", default="mono_captures", type=str)
    parser.add_argument("--results_savedir", default="calibration_results", type=str)
    parser.add_argument("--save_filename", default="mono_calibration.yml", type=str)
    parser.add_argument('--square_size', default=2.5, type=float, help='chessboard square size')
    parser.add_argument("--pattern_height", default="7", type=str)
    parser.add_argument("--pattern_width", default="7", type=str)
    args = parser.parse_args()

    # configuration from parser
    working_dir = os.getcwd()
    captures_dir = os.path.join(working_dir, "captures")
    mono_captures_dir = os.path.join(captures_dir, args.captures_savedir)
    results_dir = os.path.join(working_dir, args.results_savedir)
    save_path = os.path.join(results_dir, args.save_filename)
    square_size = args.square_size
    p_height = args.pattern_heihgt
    p_width = args.pattern_width

    # calibrate
    ret, cam_mtx, dist, rvecs, tvecs = calibrate(mono_captures_dir, square_size, p_height, p_width)

    # save calibration result
    cv_file = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("cam_mtx", cam_mtx)
    cv_file.write("dist", dist)
    cv_file.release()
