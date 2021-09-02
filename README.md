###Script
1.capture_images_mono.py: prepare captured images for single camera calibration

2.camera_calibration.py: find intrinsic and extrinsic properties of a camera.

3.image_distortion.py: undistort images based on camera calibration results.

4.capture_images_stereo.py: prepare captured images for stereo camera calibration

5.stereo_calibration.py: find transform matrix from first camera to second camera

6.stereo_disparity.py: generate depth image from stereo disparity


###Notes (Main cv2 function):
1. cv2.findChessboardCorners, cv2.cornerSubPix
Finds the positions of internal corners of the chessboard. 
2. cv2.calibrateCamera
Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
3. stero calibration
3.1 cv2.stereoCalibrate:   
Calibrates a stereo camera set up. This function finds the intrinsic parameters for each of the two cameras and the extrinsic parameters between the two cameras.
3.2 cv2.stereoRectify:   
The function computes the rotation matrices for each camera that (virtually) make both camera image planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies the dense stereo correspondence problem. The function takes the matrices computed by stereoCalibrate as input. As output, it provides two rotation matrices and also two projection matrices in the new coordinates. The function distinguishes the following two cases: 1) Horizontal stereo and 2) ertical stereo.
3.3 cv2.StereoSGBM_create
3.4 cv2.ximgproc.createRightMatcher
3.5 cv2.ximgproc.createDisparityWLSFilter
3.6 cv2.initUndistortRectifyMap

###reference: 
https://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html
