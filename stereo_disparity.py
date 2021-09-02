import numpy as np
import cv2
import argparse
import sys
from utils.load_coefficients import load_cv2file


def depth_image(left_img, right_img):
    # SGBM configuration
    window_size = 3

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    lmbda = 80000
    sigma = 1.3
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(left_img, right_img).astype(np.int16)
    dispr = right_matcher.compute(right_img, left_img).astype(np.int16)

    filteredImg = wls_filter.filter(displ, left_img, None, dispr)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    return filteredImg


def stereo_disparity(calibration_file, is_realtime, left_source, right_source):
    # load stereo calibration result
    cam_mtx1, dist1, cam_mtx2, dist2, R, T, E, F, R1, R2, P1, P2, Q, roi1, roi2 = load_cv2file(calibration_file,
                                                                                               ['cam_mtx1', 'dist1',
                                                                                                'cam_mtx2', 'dist2',
                                                                                                'R', 'T', 'E', 'F',
                                                                                                'R1', 'R2', 'P1', 'P2',
                                                                                                'Q', 'roi1', 'roi2'])

    # is camera stream or video
    if is_realtime:
        cap_left = cv2.VideoCapture(left_source, cv2.CAP_V4L2)
        cap_right = cv2.VideoCapture(right_source, cv2.CAP_V4L2)
    else:
        cap_left = cv2.VideoCapture(left_source)
        cap_right = cv2.VideoCapture(right_source)

    if cap_left.isOpened() and cap_right.isOpened():  # If we can't get images from both sources, error
        while True:
            if not (cap_left.grab() and cap_right.grab()):
                break

            left_ret, left_frame = cap_left.retrieve()
            right_ret, right_frame = cap_right.retrieve()
            if left_ret and right_ret:
                height, width, channel = left_frame.shaped  # for remapping

                # Undistortion and Rectification
                left_map_x, left_map_y = cv2.initUndistortRectifyMap(cam_mtx1, dist1, R1, P1, (width, height), cv2.CV_32FC1)
                left_rectified = cv2.remap(left_frame, left_map_x, left_map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

                right_map_x, right_map_y = cv2.initUndistortRectifyMap(cam_mtx2, dist2, R2, P2, (width, height), cv2.CV_32FC1)
                right_rectified = cv2.remap(right_frame, right_map_x, right_map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

                # Grayscale for disparity map.
                left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

                # Get the disparity image
                disparity_image = depth_image(left_gray, right_gray)

                cv2.imshow('Disparity', disparity_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    else:
        sys.exit(-9)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--calibration_file', type=str, required=True)
    parser.add_argument('--is_realtime', type=int, required=True, help='Is it camera stream or video')
    parser.add_argument('--left_source', type=str, required=True, help='Left video or v4l2 device name')
    parser.add_argument('--right_source', type=str, required=True, help='Right video or v4l2 device name')
    args = parser.parse_args()

    # configuration from argument parser
    calibration_file = args.calibration_file
    is_realtime = args.is_realtime
    left_source = int(args.left_source) if is_realtime == 1 else args.left_source
    right_source = int(args.right_source) if is_realtime == 1 else args.right_source

    stereo_disparity(calibration_file, is_realtime, left_source, right_source)
