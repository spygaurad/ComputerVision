import cv2
from camera_utils import calibrate_camera_from_chessboard, draw_cube_on_chessboard
import numpy as np
from glob import glob
from calibration_preprocessing import get_intrinsic_parameters

rms, K, dist = get_intrinsic_parameters()

test_images = glob("outputs/undistorted_test_captures/*.jpg")
for path in test_images:
    file_name = path.split("/")[-1]
    test_img = cv2.imread(path)
    cube_img, rvec, tvec = draw_cube_on_chessboard(
        test_img,
        camera_matrix=K,
        dist_coeffs=dist,
        pattern_size=(8, 6),
        square_size=0.024,  # same unit as Phase 1
    )
    cv2.imwrite("outputs/ar_cube_overlayed/AR_cube_" + file_name, cube_img)
    print(f"Saved augmented reality image to: outputs/ar_cube_overlayed/AR_cube_{file_name}")