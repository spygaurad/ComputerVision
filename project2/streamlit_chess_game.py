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

from camera_utils import draw_chess_piece_on_chessboard
from calibration_preprocessing import get_intrinsic_parameters


PIECE_PATH = "images/king.png"  # put your file here
piece_rgba = cv2.imread(PIECE_PATH, cv2.IMREAD_UNCHANGED)
if piece_rgba is None:
    raise RuntimeError(f"Could not load chess piece image: {PIECE_PATH}")


# ---------------------------------------------------------
# Load intrinsic parameters once
# ---------------------------------------------------------
rms, K, dist = get_intrinsic_parameters()

PATTERN_SIZE = (8, 6)    # inner corners
SQUARE_SIZE = 0.024      # meters or same unit as calibration

NUM_SQ_X = PATTERN_SIZE[0] - 1
NUM_SQ_Y = PATTERN_SIZE[1] - 1

# ---------------------------------------------------------
# Streamlit UI (compact layout)
# ---------------------------------------------------------
# st.set_page_config(page_title="Calibrated AR Webcam", layout="centered")

# Tiny CSS tweak to reduce padding and keep things compact
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 800px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Initialize piece position (center of board)
if "piece_x" not in st.session_state:
    st.session_state.piece_x = NUM_SQ_X // 2
if "piece_y" not in st.session_state:
    st.session_state.piece_y = NUM_SQ_Y // 2

# ---------------------------------------------------------
# Video transformer — outputs ONLY the AR-overlaid frame
# ---------------------------------------------------------
class ARVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_ar = None
        # defaults; will be overwritten from the main script
        self.piece_x = NUM_SQ_X // 2
        self.piece_y = NUM_SQ_Y // 2

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        h, w = img.shape[:2]
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
        undistorted = cv2.undistort(img, K, dist, None, newK)

        # use the transformer’s own attributes
        px = self.piece_x
        py = self.piece_y

        try:
            overlay_img, rvec, tvec = draw_chess_piece_on_chessboard(
                undistorted,
                camera_matrix=K,      # or newK
                dist_coeffs=dist,
                pattern_size=PATTERN_SIZE,
                square_size=SQUARE_SIZE,
                piece_rgba=piece_rgba,
                square_x=px,
                square_y=py,
            )
        except RuntimeError:
            overlay_img = undistorted.copy()
            cv2.putText(
                overlay_img,
                "Chessboard not found",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        self.last_ar = overlay_img
        return av.VideoFrame.from_ndarray(overlay_img, format="bgr24")


# ---------------------------------------------------------
# Webcam widget (small YouTube-style size)
# ---------------------------------------------------------
# st.markdown("#### Live View")

webrtc_ctx = webrtc_streamer(
    key="ar-webcam",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=ARVideoTransformer,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},   # YouTube-ish size
            "height": {"ideal": 360},
            "frameRate": {"ideal": 15},
        },
        "audio": False,
    },
    async_processing=True,
)

# ---------------------------------------------------------
# Controls directly below video (D-pad style)
# ---------------------------------------------------------
st.markdown("#### Controls")

# --- Compact D-Pad CSS ---
st.markdown("""
<style>
/* remove vertical spacing */
.row-widget.stButton > button {
    padding: 0.25rem 0.7rem !important;
    font-size: 0.9rem !important;
}
/* remove column padding */
div[data-testid="column"] {
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Compact D-Pad Layout ---
pad_up = st.columns([1, 1, 1])
with pad_up[1]:
    if st.button("⬆️"):
        st.session_state.piece_y = max(0, st.session_state.piece_y - 1)

pad_mid = st.columns([1, 1, 1])
with pad_mid[0]:
    if st.button("⬅️"):
        st.session_state.piece_x = max(0, st.session_state.piece_x - 1)
with pad_mid[1]:
    if st.button("⬇️"):
        st.session_state.piece_y = min(NUM_SQ_Y - 1, st.session_state.piece_y + 1)
with pad_mid[2]:
    if st.button("➡️"):
        st.session_state.piece_x = min(NUM_SQ_X - 1, st.session_state.piece_x + 1)

st.caption(
    f"Current square: x={st.session_state.piece_x}, "
    f"y={st.session_state.piece_y} (0-based indices)"
)

# 🔁 NOW sync transformer AFTER controls so it sees updated state
if webrtc_ctx and webrtc_ctx.video_transformer:
    webrtc_ctx.video_transformer.piece_x = st.session_state.piece_x
    webrtc_ctx.video_transformer.piece_y = st.session_state.piece_y


# ---------------------------------------------------------
# Save AR frame (compact)
# ---------------------------------------------------------
col_save, col_spacer = st.columns([1, 1])
with col_save:
    if st.button("💾 Save Current AR Frame"):
        vt = webrtc_ctx.video_transformer if webrtc_ctx else None

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
