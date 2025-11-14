import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import os

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

    num_corners_x, num_corners_y = pattern_size

    # Template 3D points for the chessboard (z = 0 plane in board frame)
    objp = np.zeros((num_corners_x * num_corners_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_corners_x, 0:num_corners_y].T.reshape(-1, 2)
    objp *= square_size  # scale by the physical square size

    
    objpoints = []  # 3D points in board coordinates
    imgpoints = []  # 2D detected corner in image coordinates

    img_size = None

    # Corner refinement criteria (sub-pixel accuracy)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-6,
    )

    # Detect Chessboard corners in each image
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[WARN] Could not read image: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (width, height)

        # Find the chessboard corners
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
        )

    # Global camera calibration over all views
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
    Calibrated Augmented Reality.

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

    # Detect chessboard corners in the image 
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

    # Prepare 3D object points for the chessboard plane 
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size  # scale by physical size

    # Estimate pose with solvePnP (gives R, T) 
    ret, rvec, tvec = cv2.solvePnP(
        objp,
        corners_refined,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ret:
        raise RuntimeError("solvePnP failed to estimate camera pose.")

    # Defining a cube in 3D anchored on the chessboard
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

    # Project 3D cube vertices to the image
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

    # Prepare 3D object points for the chessboard plane (z = 0)
    # One 3D point per inner corner.
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    # Estimate pose with solvePnP
    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners_refined,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed to estimate camera pose.")

    # Defining the 3D quad corresponding to the chosen 1x1 square
    # Number of actual squares = (cols-1) x (rows-1)
    num_squares_x = cols - 1
    num_squares_y = rows - 1

    # Clamping so we don't go out-of-bounds
    square_x = int(np.clip(square_x, 0, num_squares_x - 1))
    square_y = int(np.clip(square_y, 0, num_squares_y - 1))

    s = square_size
    # Four corners of the square on the board (z=0 plane)
    # (x,y) in board coordinates
    square_pts_3d = np.float32([
        [square_x * s,         square_y * s,         0.0],  # top-left
        [(square_x + 1) * s,   square_y * s,         0.0],  # top-right
        [(square_x + 1) * s,   (square_y + 1) * s,   0.0],  # bottom-right
        [square_x * s,         (square_y + 1) * s,   0.0],  # bottom-left
    ])

    # Projecting these 3D square corners into the image
    imgpts, _ = cv2.projectPoints(
        square_pts_3d,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    dst_quad = imgpts.reshape(-1, 2).astype(np.float32)  # (4,2)

    # Prepare the chess piece image and alpha
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

    # Compute homography and warp piece & alpha to the board square
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

    # Alpha blend onto the board
    img_out = alpha_f * warped_piece_f + (1.0 - alpha_f) * img_out
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)

    return img_out, rvec, tvec




def stitch_two_images(img1, img2, ratio_thresh=0.75, reproj_thresh=4.0):
    """
    Stitch two overlapping images into a single panorama.

    Parameters
    ----------
    img1 : np.ndarray
        First image (e.g., left image), BGR or RGB as read by cv2.
    img2 : np.ndarray
        Second image (e.g., right image), BGR or RGB as read by cv2.
    ratio_thresh : float, optional
        Lowe's ratio threshold for filtering matches (default 0.75).
    reproj_thresh : float, optional
        RANSAC reprojection threshold for homography estimation.

    Returns
    -------
    panorama : np.ndarray
        The stitched panorama image (same color format as input).
    """

    # --- 1. Convert to grayscale for feature detection ---
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    # --- 2. Detect keypoints + descriptors with ORB ---
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        raise ValueError("Could not find enough features in one of the images.")

    # --- 3. Match descriptors using Brute-Force + Hamming distance ---
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # --- 4. Apply Lowe's ratio test to keep only good matches ---
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError(
            f"Not enough good matches ({len(good_matches)}) to compute homography."
        )

    # --- 5. Extract matched keypoints and compute Homography with RANSAC ---
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    if H is None:
        raise ValueError("Homography computation failed.")

    # --- 6. Compute size of the output canvas (handle negative coords) ---
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Corners of img1 in its own coordinate system
    corners_img1 = np.float32(
        [[0, 0], [w1, 0], [w1, h1], [0, h1]]
    ).reshape(-1, 1, 2)
    # Warp them into img2's coordinate system
    warped_corners_img1 = cv2.perspectiveTransform(corners_img1, H)

    # Corners of img2 (already in its own coordinate system)
    corners_img2 = np.float32(
        [[0, 0], [w2, 0], [w2, h2], [0, h2]]
    ).reshape(-1, 1, 2)

    # Combine all corners to find overall bounds
    all_corners = np.concatenate((warped_corners_img1, corners_img2), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation to shift everything so minimum coords are at (0, 0)
    translation = [-xmin, -ymin]
    T = np.array(
        [[1, 0, translation[0]],
         [0, 1, translation[1]],
         [0, 0, 1]],
        dtype=np.float64,
    )

    # H_translated = T @ H     # Final homography with translation


    # --- 7. Warp img1 using the composed homography (T * H) ---
    pano_width = xmax - xmin
    pano_height = ymax - ymin

    panorama = cv2.warpPerspective(img1, T @ H, (pano_width, pano_height))

    # --- 8. Paste img2 into the panorama at the translated location ---
    x_offset, y_offset = translation
    panorama[y_offset:y_offset + h2, x_offset:x_offset + w2] = img2

    return panorama, H


def show_and_save_stitch(img1, img2, stitched, output_path):
    """
    Plot the two input images and the final stitched image side-by-side
    and save the figure to the given output path.

    Parameters
    ----------
    img1 : np.ndarray
        First image (BGR as from cv2.imread).
    img2 : np.ndarray
        Second image (BGR as from cv2.imread).
    stitched : np.ndarray
        Stitched panorama image (BGR).
    output_path : str
        Path to save the resulting figure (e.g., 'results/stitched.png').
    """

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert BGR (OpenCV default) to RGB for matplotlib
    def bgr_to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img

    img1_rgb = bgr_to_rgb(img1)
    img2_rgb = bgr_to_rgb(img2)
    stitched_rgb = bgr_to_rgb(stitched)

    # Create figure with 3 subplots
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img1_rgb)
    plt.title("Image 1")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img2_rgb)
    plt.title("Image 2")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(stitched_rgb)
    plt.title("Stitched Panorama")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()

