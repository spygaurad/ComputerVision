import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    RTCConfiguration,
    WebRtcMode,
)
import av
import os
from datetime import datetime

from camera_utils import draw_cube_on_chessboard
from project2.phase1_calibration_preprocessing import get_intrinsic_parameters


# ---------------------------------------------------------
# Load intrinsic parameters once
# ---------------------------------------------------------
rms, K, dist = get_intrinsic_parameters()

PATTERN_SIZE = (8, 6)    # inner corners
SQUARE_SIZE = 0.024      # meters or same unit as calibration


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Calibrated AR Webcam", layout="wide")
st.title("Calibrated Augmented Reality (Live)")

st.caption(
    "Point your calibrated camera at the same chessboard pattern used in Phase 1."
)


# ---------------------------------------------------------
# Video transformer — outputs ONLY the AR-overlaid frame
# ---------------------------------------------------------
class ARVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_ar = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        h, w = img.shape[:2]
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
        undistorted = cv2.undistort(img, K, dist, None, newK)

        # Try AR overlay
        try:
            cube_img, rvec, tvec = draw_cube_on_chessboard(
                undistorted,
                camera_matrix=K,
                dist_coeffs=dist,
                pattern_size=PATTERN_SIZE,
                square_size=SQUARE_SIZE,
            )
        except RuntimeError:
            cube_img = undistorted.copy()
            cv2.putText(
                cube_img,
                "Chessboard not found",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        self.last_ar = cube_img.copy()
        return av.VideoFrame.from_ndarray(cube_img, format="bgr24")


# ---------------------------------------------------------
# Webcam widget
# ---------------------------------------------------------
st.markdown("### Start Webcam")

webrtc_ctx = webrtc_streamer(
    key="ar-webcam",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=ARVideoTransformer,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


# ---------------------------------------------------------
# Save AR frame
# ---------------------------------------------------------
st.markdown("### Save Current AR Frame")

if st.button("💾 Save AR Frame"):
    vt = webrtc_ctx.video_transformer

    if vt and vt.last_ar is not None:
        os.makedirs("./outputs/ar_live", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"./outputs/ar_live/ar_frame_{ts}.png"

        if cv2.imwrite(path, vt.last_ar):
            st.success("Saved!")
            st.caption(path)
        else:
            st.error("Failed to save image.")

    else:
        st.warning("Start the webcam and point it at the chessboard first.")
