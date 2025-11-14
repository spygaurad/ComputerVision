import cv2
import numpy as np
from calibration_preprocessing import get_intrinsic_parameters
from camera_utils import estimate_relative_pose  # after you paste it there
from camera_utils import visualize_warp_matplotlib  # put the warp fn here

# Load your two images (camera moved ~10 cm horizontally, etc.)
img1 = cv2.imread("outputs/undistorted_test_captures/undistorted_chess_1763058672.jpg")
img2 = cv2.imread("outputs/undistorted_test_captures/undistorted_chess_1763058677.jpg")

rms, K, dist = get_intrinsic_parameters()

# get pose + inlier matches
F, E, R, t, inlier_mask, pts1_all, pts2_all = estimate_relative_pose(img1, img2, K)

# Keep only inliers
pts1 = pts1_all[inlier_mask.ravel() == 1]
pts2 = pts2_all[inlier_mask.ravel() == 1]
