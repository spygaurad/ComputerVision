from glob import glob
from pathlib import Path
from camera_utils import calibrate_camera_from_chessboard, undistort_image
import cv2

def get_intrinsic_parameters():
    # 1. Collect chessboard images (e.g., .jpg in folder)
    images = sorted(glob("images/captures/*.jpg"))

    # 2. Run calibration
    # Suppose your board has 9 inner corners along width, 6 along height, and each square is 24 mm
    rms, K, dist, rvecs, tvecs = calibrate_camera_from_chessboard(
        images,
        pattern_size=(8, 6),
        square_size=0.024,  # meters, or 24 mm if you're thinking in SI
        debug_show=False,
    )
    return rms, K, dist

def generate_undistorted_samples():
    rms, K, dist = get_intrinsic_parameters()

    test_images = glob("images/test_captures/*.jpg")
    for path in test_images:
        img = cv2.imread(path)
        undistorted = undistort_image(img, K, dist, alpha=0.0)
        out_path = f"outputs/undistorted_test_captures/undistorted_{Path(path).name}"
        cv2.imwrite(out_path, undistorted)
        print(f"Saved undistorted image to: {out_path}")

if __name__ == "__main__":
    generate_undistorted_samples()