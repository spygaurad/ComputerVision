import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from typing import Tuple
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def calibrate_camera_from_chessboard(
    image_paths: List[str],
    pattern_size: Tuple[int, int],
    square_size: float,
    debug_show: bool = False,
):
    """
    Calibrate a pinhole camera using multiple chessboard images (Phase 1).

    Parameters
    ----------
    image_paths : list of str
        Paths to chessboard images (15–20 views from different poses).
    pattern_size : (cols, rows)
        Number of internal corners per chessboard row and column, e.g. (9, 6).
        This is the same tuple you pass to cv2.findChessboardCorners().
    square_size : float
        Size of one square edge in *real-world units* (e.g. 0.024 for 24 mm).
        Only the relative scale matters for calibration; pick any consistent unit.
    debug_show : bool, default False
        If True, shows detected corners for visual verification.

    Returns
    -------
    retval : float
        Overall RMS re-projection error from cv2.calibrateCamera().
    camera_matrix : np.ndarray, shape (3, 3)
        Intrinsic matrix K.
    dist_coeffs : np.ndarray, shape (N,)
        Distortion coefficients (k1, k2, p1, p2, k3, ...).
    rvecs : list of np.ndarray
        Rotation vectors for each input view (extrinsics).
    tvecs : list of np.ndarray
        Translation vectors for each input view (extrinsics).
    """

    # --- 1. Prepare the known 3D coordinates of chessboard corners in the pattern frame ---
    # pattern_size = (num_corners_x, num_corners_y)
    num_corners_x, num_corners_y = pattern_size

    # Create one template of 3D points for the chessboard (z = 0)
    objp = np.zeros((num_corners_x * num_corners_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_corners_x, 0:num_corners_y].T.reshape(-1, 2)
    objp *= square_size  # scale by the physical square size

    # Lists to store 3D points (in world coords) and 2D points (in image coords) for all images
    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    img_size = None

    # Criteria for refining corner locations (sub-pixel accuracy)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-6,
    )

    # --- 2. Loop over images and detect chessboard corners ---
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[WARN] Could not read image: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (width, height)

        # Try to find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if not ret:
            print(f"[INFO] Chessboard NOT found in: {path}")
            continue

        # Refine corner locations to subpixel accuracy
        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria,
        )

        objpoints.append(objp)
        imgpoints.append(corners_refined)

        if debug_show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners_refined, ret)
            cv2.imshow("Detected Corners", vis)
            cv2.waitKey(500)

    if debug_show:
        cv2.destroyAllWindows()

    if len(objpoints) < 3:
        raise RuntimeError(
            f"Not enough valid chessboard detections ({len(objpoints)}). "
            "Make sure you have good views (15–20 recommended)."
        )

    # --- 3. Calibrate the camera ---
    # This returns: rms_error, K, dist, rvecs, tvecs
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_size,
        None,  # initial camera matrix (None = estimate)
        None,  # initial distortion (None = estimate)
    )

    print("[INFO] Calibration RMS re-projection error:", retval)
    print("[INFO] Camera matrix K:\n", camera_matrix)
    print("[INFO] Distortion coefficients:\n", dist_coeffs.ravel())

    return retval, camera_matrix, dist_coeffs, rvecs, tvecs


def undistort_image(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    alpha: float = 0.0,
):
    """
    Undistort an image using the calibration result.

    Parameters
    ----------
    img : np.ndarray
        Input distorted BGR image.
    camera_matrix : np.ndarray
        Intrinsic matrix K from calibrate_camera_from_chessboard().
    dist_coeffs : np.ndarray
        Distortion coefficients from calibrateCamera().
    alpha : float in [0, 1]
        Free scaling parameter:
        - 0.0 = crop to valid region (no black borders)
        - 1.0 = keep all pixels (may include black borders)

    Returns
    -------
    undistorted : np.ndarray
        Undistorted BGR image.
    """

    h, w = img.shape[:2]

    # Compute a new optimal camera matrix based on the free scaling parameter alpha
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
    )

    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_K)

    # Optionally crop to the ROI (region of interest)
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        undistorted = undistorted[y : y + h_roi, x : x + w_roi]

    return undistorted


def draw_cube_on_chessboard(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    pattern_size: Tuple[int, int],
    square_size: float,
):
    """
    Phase 2 option: Calibrated Augmented Reality.

    Uses the calibrated intrinsics (K) and a known chessboard pattern
    to estimate the camera pose (R, T) with cv2.solvePnP, then overlays
    a virtual 3D cube on top of the board.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image containing the same chessboard pattern used
        in calibration.
    camera_matrix : np.ndarray
        Intrinsic matrix K from cv2.calibrateCamera.
    dist_coeffs : np.ndarray
        Distortion coefficients from calibration.
    pattern_size : (cols, rows)
        Number of inner corners in the chessboard (same as Phase 1).
    square_size : float
        Physical size of a chessboard square (same units as Phase 1).

    Returns
    -------
    img_out : np.ndarray
        Image with a projected cube drawn on the chessboard.
    rvec : np.ndarray
        3×1 rotation vector (Rodrigues) giving camera pose.
    tvec : np.ndarray
        3×1 translation vector giving camera pose.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cols, rows = pattern_size

    # --- 1. Detect chessboard corners in the image ---
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not ret:
        raise RuntimeError("Chessboard could not be found in the image.")

    # Sub-pixel refine corners
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-6,
    )
    corners_refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )

    # --- 2. Prepare 3D object points for the chessboard plane (z = 0) ---
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size  # scale by physical size

    # --- 3. Estimate pose with solvePnP (gives R, T) ---
    ret, rvec, tvec = cv2.solvePnP(
        objp,
        corners_refined,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ret:
        raise RuntimeError("solvePnP failed to estimate camera pose.")

    # --- 4. Define a cube in 3D anchored on the chessboard ---
    # Cube base: same size as a 2×2 square region on the board.
    s = square_size
    cube_pts_3d = np.float32([
        [0, 0,      0],  # bottom square
        [2*s, 0,    0],
        [2*s, 2*s,  0],
        [0, 2*s,    0],
        [0, 0,   -2*s],  # top square (negative z = up from board)
        [2*s, 0,  -2*s],
        [2*s, 2*s,-2*s],
        [0, 2*s,  -2*s],
    ])

    # --- 5. Project 3D cube vertices to the image ---
    imgpts, _ = cv2.projectPoints(
        cube_pts_3d, rvec, tvec, camera_matrix, dist_coeffs
    )
    imgpts = imgpts.reshape(-1, 2).astype(int)

    img_out = img.copy()

    # helper lambdas for drawing
    def draw_poly(img_, pts, closed=True):
        pts = pts.reshape(-1, 1, 2)
        cv2.polylines(img_, [pts], closed, (0, 255, 0), 2)

    # bottom, top, and vertical edges
    bottom = imgpts[0:4]
    top = imgpts[4:8]

    # bottom square (on board)
    draw_poly(img_out, bottom)
    # top square
    draw_poly(img_out, top)
    # vertical edges
    for b, t in zip(bottom, top):
        cv2.line(img_out, tuple(b), tuple(t), (255, 0, 0), 2)

    return img_out, rvec, tvec

import cv2
import numpy as np
from typing import Tuple


def draw_chess_piece_on_chessboard(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    pattern_size: Tuple[int, int],
    square_size: float,
    piece_rgba: np.ndarray,
    square_x: int,
    square_y: int,
):
    """
    Overlay a 1x1 chess piece PNG on a chessboard square using pose estimation.

    - Uses the calibrated intrinsics (K) and known chessboard pattern to estimate
      camera pose (R, T) via cv2.solvePnP.
    - Then projects a 1x1-square region on the board and warps the chess PNG
      onto that quad with alpha blending.

    Parameters
    ----------
    img : np.ndarray
        Input BGR image containing the same chessboard pattern used in calibration.
    camera_matrix : np.ndarray
        Intrinsic matrix K from cv2.calibrateCamera.
    dist_coeffs : np.ndarray
        Distortion coefficients from calibration.
    pattern_size : (cols, rows)
        Number of inner corners in the chessboard (same as in calibration),
        e.g. (8, 6) means 8 corners horizontally, 6 vertically.
    square_size : float
        Physical size of a chessboard square (same units as calibration).
    piece_rgba : np.ndarray
        Chess piece image with alpha channel (H x W x 4), e.g. from
        cv2.imread("piece.png", cv2.IMREAD_UNCHANGED).
    square_x : int
        Square index along x (0 .. num_squares_x-1).
    square_y : int
        Square index along y (0 .. num_squares_y-1).

    Returns
    -------
    img_out : np.ndarray
        BGR image with the chess piece overlaid.
    rvec : np.ndarray
        3x1 rotation vector (Rodrigues).
    tvec : np.ndarray
        3x1 translation vector.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cols, rows = pattern_size  # cols = x, rows = y (inner corners)

    # --- 1. Detect chessboard corners in the image ---
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not ret:
        raise RuntimeError("Chessboard could not be found in the image.")

    # Refine corners to subpixel accuracy
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-6,
    )
    corners_refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )

    # --- 2. Prepare 3D object points for the chessboard plane (z = 0) ---
    # One 3D point per inner corner.
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    # --- 3. Estimate pose with solvePnP ---
    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners_refined,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed to estimate camera pose.")

    # --- 4. Define the 3D quad corresponding to the chosen 1x1 square ---
    # Number of actual squares = (cols-1) x (rows-1)
    num_squares_x = cols - 1
    num_squares_y = rows - 1

    # Clamp so we don't go out-of-bounds
    square_x = int(np.clip(square_x, 0, num_squares_x - 1))
    square_y = int(np.clip(square_y, 0, num_squares_y - 1))

    s = square_size
    # Four corners of that square on the board (z=0 plane)
    # (x,y) in board coordinates
    square_pts_3d = np.float32([
        [square_x * s,         square_y * s,         0.0],  # top-left
        [(square_x + 1) * s,   square_y * s,         0.0],  # top-right
        [(square_x + 1) * s,   (square_y + 1) * s,   0.0],  # bottom-right
        [square_x * s,         (square_y + 1) * s,   0.0],  # bottom-left
    ])

    # --- 5. Project these 3D square corners into the image ---
    imgpts, _ = cv2.projectPoints(
        square_pts_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    dst_quad = imgpts.reshape(-1, 2).astype(np.float32)  # (4,2)

    # --- 6. Prepare the chess piece image and alpha ---
    if piece_rgba.shape[2] == 4:
        piece_bgr = piece_rgba[:, :, :3]
        piece_alpha = piece_rgba[:, :, 3]
    else:
        piece_bgr = piece_rgba
        piece_alpha = np.full(piece_bgr.shape[:2], 255, dtype=np.uint8)

    h_piece, w_piece = piece_bgr.shape[:2]

    # Source quad: corners of the piece image
    src_quad = np.float32([
        [0,        0],         # top-left
        [w_piece,  0],         # top-right
        [w_piece,  h_piece],   # bottom-right
        [0,        h_piece],   # bottom-left
    ])

    # --- 7. Compute homography and warp piece & alpha to the board square ---
    H, _ = cv2.findHomography(src_quad, dst_quad)
    if H is None:
        raise RuntimeError("Homography for chess piece overlay failed.")

    h_img, w_img = img.shape[:2]
    warped_piece = cv2.warpPerspective(piece_bgr, H, (w_img, h_img))
    warped_alpha = cv2.warpPerspective(piece_alpha, H, (w_img, h_img))

    # Normalize alpha to [0, 1]
    alpha_f = warped_alpha.astype(np.float32) / 255.0
    alpha_f = np.expand_dims(alpha_f, axis=2)  # (H, W, 1)

    img_out = img.copy().astype(np.float32)
    warped_piece_f = warped_piece.astype(np.float32)

    # --- 8. Alpha blend onto the board ---
    img_out = alpha_f * warped_piece_f + (1.0 - alpha_f) * img_out
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)

    return img_out, rvec, tvec


def estimate_relative_pose(
    img1: np.ndarray,
    img2: np.ndarray,
    K: np.ndarray,
    use_orb: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the relative pose (R, T) between two camera positions using
    feature matching + 8-point algorithm, as described in the assignment.

    Steps
    -----
    1. Detect and describe keypoints in both images (ORB by default).
    2. Match descriptors and apply Lowe's ratio test.
    3. Compute the Fundamental matrix F with the 8-point algorithm:
       F = findFundamentalMat(pts1, pts2, FM_RANSAC).
    4. Convert F to the Essential matrix E using intrinsics K:
       E = Kᵀ F K.
    5. Decompose E with recoverPose(E, pts1, pts2, K) to obtain R, T.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Two grayscale or BGR images taken from different, static camera poses.
    K : np.ndarray, shape (3, 3)
        Camera intrinsic matrix from calibration.
    use_orb : bool, default True
        If True, uses ORB (fast, free). If False, uses SIFT (if available).

    Returns
    -------
    F : np.ndarray, shape (3, 3)
        Fundamental matrix.
    E : np.ndarray, shape (3, 3)
        Essential matrix.
    R : np.ndarray, shape (3, 3)
        Relative rotation from camera 1 to camera 2.
    t : np.ndarray, shape (3, 1)
        Relative translation direction from camera 1 to camera 2 (scale unknown).
    inlier_mask : np.ndarray, shape (N, 1)
        Mask of inlier matches used for F and pose estimation.
    """
    # --- Ensure grayscale ---
    if img1.ndim == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1.copy()

    if img2.ndim == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2.copy()

    # --- 1. Detect and describe features ---
    if use_orb:
        detector = cv2.ORB_create(nfeatures=2000)
        norm_type = cv2.NORM_HAMMING
    else:
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2

    kpts1, desc1 = detector.detectAndCompute(gray1, None)
    kpts2, desc2 = detector.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None or len(kpts1) < 8 or len(kpts2) < 8:
        raise RuntimeError("Not enough features found in one or both images.")

    # --- 2. Match descriptors + Lowe's ratio test ---
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        raise RuntimeError(f"Not enough good matches for 8-point algorithm "
                           f"({len(good_matches)} found).")

    pts1 = np.float32([kpts1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kpts2[m.trainIdx].pt for m in good_matches])

    # --- 3. Fundamental matrix F using 8-point + RANSAC ---
    # FM_RANSAC uses 8-point internally with robust outlier rejection.
    F, inlier_mask = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.999
    )

    if F is None or F.shape != (3, 3):
        raise RuntimeError("Failed to compute a valid Fundamental matrix F.")

    inliers1 = pts1[inlier_mask.ravel() == 1]
    inliers2 = pts2[inlier_mask.ravel() == 1]

    # --- 4. Essential matrix E = Kᵀ F K ---
    E = K.T @ F @ K

    # --- 5. Decompose E to get (R, T) ---
    # recoverPose expects normalized image coordinates (pixel coords + K).
    _, R, t, pose_mask = cv2.recoverPose(E, inliers1, inliers2, K)

    return F, E, R, t, inlier_mask, pts1, pts2
