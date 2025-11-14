import cv2
import time
from pathlib import Path

# --- SETTINGS ---
PATTERN_SIZE = (8, 6)          # 9x7 squares -> 8x6 inner corners = 48
NEEDED_CORNERS = PATTERN_SIZE[0] * PATTERN_SIZE[1]  # 48
TOTAL_TO_SAVE = 30             # stop after saving this many frames
SAVE_DIR = Path("images")    # folder to store images
CAM_INDEX = 0                  # change to 1 if you have multiple cameras
# ----------------

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Try to open the webcam
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    # Some systems prefer these backends; try again if first attempt fails
    cap.release()
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)  # Windows
if not cap.isOpened():
    print("❌ Could not open webcam. Try a different CAM_INDEX (0/1) or backend.")
    raise SystemExit

# Optional: set resolution (comment out if unnecessary)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Chessboard Capture", cv2.WINDOW_NORMAL)

saved = 0
last_save_t = 0.0

# Small termination helper so you don't save duplicates too fast
def ok_to_save(min_gap=0.6):
    return (time.time() - last_save_t) > min_gap

# Main loop
while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️ Failed to read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # More robust SB detector (OpenCV 4.5+); falls back to classic if not available
    use_sb = hasattr(cv2, "findChessboardCornersSB")
    if use_sb:
        flags = (cv2.CALIB_CB_EXHAUSTIVE |
                 cv2.CALIB_CB_ACCURACY |
                 cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret, corners = cv2.findChessboardCornersSB(gray, PATTERN_SIZE, flags=flags)
    else:
        ret, corners = cv2.findChessboardCorners(
            gray, PATTERN_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )

    total = len(corners) if ret else 0

    # Draw & refine if found
    if ret:
        if not use_sb:
            # refine for classic detector
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(frame, PATTERN_SIZE, corners, ret)

    # HUD text
    cv2.putText(frame, f"Corners: {total}/{NEEDED_CORNERS}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255 if ret else 0, 255 if ret else 0), 2)
    cv2.putText(frame, f"Saved: {saved}/{TOTAL_TO_SAVE}  |  Press 'q' to quit",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save when full pattern seen
    if ret and total == NEEDED_CORNERS and ok_to_save():
        filename = SAVE_DIR / f"chess_{int(time.time())}.jpg"
        cv2.imwrite(str(filename), frame)
        last_save_t = time.time()
        saved += 1
        print(f"✅ Saved {filename}  ({saved}/{TOTAL_TO_SAVE})")
        if saved >= TOTAL_TO_SAVE:
            cv2.putText(frame, "Done! Collected all frames.", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Chessboard Capture", frame)
            cv2.waitKey(1000)
            break

    # Show window
    cv2.imshow("Chessboard Capture", frame)

    # Keep UI responsive; press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
