# https://docs.opencv.org/4.5.2/d9/d0c/group__calib3d.html#ga91018d80e2a93ade37539f01e6f07de5

import os
import numpy as np
import cv2
import glob
import argparse
import sys
from utils.load_coefficients import load_cv2file

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def stereo_calibrate(left_dir, left_calib_path, right_dir, right_calib_path, square_size, p_width, p_height):
    objp, leftp, rightp, img_size = load_image_points(left_dir, right_dir, square_size, p_width, p_height)
    cam_mtx1, dist1 = load_cv2file(left_calib_path, ["cam_mtx", "dist"])
    cam_mtx2, dist2 = load_cv2file(right_calib_path, ["cam_mtx", "dist"])

    flag = 0
    # flag |= cv2.CALIB_FIX_INTRINSIC
    # flag |= cv2.CALIB_FIX_PRINCIPAL_POINT
    flag |= cv2.CALIB_USE_INTRINSIC_GUESS
    # flag |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flag |= cv2.CALIB_FIX_ASPECT_RATIO
    # flag |= cv2.CALIB_ZERO_TANGENT_DIST
    # flag |= cv2.CALIB_RATIONAL_MODEL
    # flag |= cv2.CALIB_SAME_FOCAL_LENGTH
    # flag |= cv2.CALIB_FIX_K3
    # flag |= cv2.CALIB_FIX_K4
    # flag |= cv2.CALIB_FIX_K5

    # Stereo Calibration
    ret, cam_mtx1, dist1, cam_mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(objp, leftp, rightp, cam_mtx1, dist1, cam_mtx2, dist2, img_size)
    print("stereo calibration rms: ", ret)
    print("rotation matrix (from first camera's coordinate system to second camera's coordinate system): ", R)
    print("translation matrix (from first camera's coordinate system to second camera's coordinate system): ", T)
    print("Essential Matrix: ", E)
    print("Fundamental matrix: ", F)

    # Stereo Rectify
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cam_mtx1, dist1, cam_mtx2, dist2, img_size, R, T,
                                                               flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)
    print("3x3 rectification transform (rotation matrix) for the first camera: ", R1)
    print("3x3 rectification transform (rotation matrix) for the second camera: ", R2)
    print("3x4 projection matrix in the new (rectified) coordinate systems for the first camera: ", P1)
    print("3x4 projection matrix in the new (rectified) coordinate systems for the second camera: ", P2)
    print("4×4 disparity-to-depth mapping matrix: ", Q)

    return cam_mtx1, dist1, cam_mtx2, dist2, R, T, E, F, R1, R2, P1, P2, Q, roi1, roi2


def load_image_points(left_dir, right_dir, square_size, width=9, height=6):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size  # Create real world coords. Use your metric.

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    left_imgpoints = []  # 2d points in image plane.
    right_imgpoints = []  # 2d points in image plane.

    # Get images for left and right directory. Since we use prefix and formats, both image set can be in the same dir.
    left_images = glob.glob(left_dir).sort()
    right_images = glob.glob(right_dir).sort()

    if len(left_images) != len(right_images):
        print("Numbers of left and right images are not equal. They should be pairs.")
        sys.exit(-1)
    else:
        pair_images = zip(left_images, right_images)  # Pair the images for single loop handling

    for left_img, right_img in pair_images:
        # right object points
        right = cv2.imread(right_img)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, (width, height),
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        # left object points
        left = cv2.imread(left_img)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        ret_left, corners_left = cv2.findChessboardCorners(gray_right, (width, height),
                                                             cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        if ret_left and ret_right:
            objpoints.append(objp)
            # Right points
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (5, 5), (-1, -1), criteria)
            right_imgpoints.append(corners2_right)
            # Left points
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (5, 5), (-1, -1), criteria)
            left_imgpoints.append(corners2_left)
        else:
            print("Chessboard couldn't detected. Image pair: ", left_img, " and ", right_img)
            continue

        img_size = gray_left.shape

        return objpoints, left_imgpoints, right_imgpoints, img_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures_prefix", default="stereo_captures_", type=str)
    parser.add_argument("--results_savedir", default="calibration_results", type=str)
    parser.add_argument("--save_filename", default="stereo_calibration.yml")
    parser.add_argument('--square_size', default=2.5, type=float, help='chessboard square size')
    parser.add_argument("--pattern_height", default="7", type=str)
    parser.add_argument("--pattern_width", default="7", type=str)
    args = parser.parse_args()

    # configuration from parser
    working_dir = os.getcwd()
    captures_dir = os.path.join(working_dir, "captures")
    left_dir = os.path.join(captures_dir, args.captures_prefix + "left")
    right_dir = os.path.join(captures_dir, args.captures_prefix + "right")
    results_dir = os.path.join(working_dir, args.results_savedir)
    save_path = os.path.join(results_dir, args.save_filename)
    square_size = args.square_size
    p_height = args.pattern_heihgt
    p_width = args.pattern_width

    # calibrate
    cam_mtx1, dist1, cam_mtx2, dist2, R, T, E, F, R1, R2, P1, P2, Q, roi1, roi2 = stereo_calibrate(left_dir, right_dir, square_size, p_height, p_width)

    # save calibration result
    cv_file = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("cam_mtx1", cam_mtx1)
    cv_file.write("dist1", dist1)
    cv_file.write("cam_mtx2", cam_mtx2)
    cv_file.write("dist2", dist2)
    cv_file.write("R", R)
    cv_file.write("T", T)
    cv_file.write("E", E)
    cv_file.write("F", F)
    cv_file.write("R1", R1)
    cv_file.write("R2", R2)
    cv_file.write("P1", P1)
    cv_file.write("P2", P2)
    cv_file.write("Q", Q)
    cv_file.release()
