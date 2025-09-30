import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import av
import os
from datetime import datetime
from convolution_implementation import get_filters

st.set_page_config(page_title="Webcam Filters", layout="wide")
st.title("Webcam with Real-time Convolution Filters")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.filter_name = 'sobel_vertical'
        self.filters = get_filters()
        self.last_original = None
        self.last_filtered = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        kernel = self.filters[self.filter_name]
        filtered = cv2.filter2D(img, -1, kernel)

        if 'sobel' in self.filter_name or 'emboss' in self.filter_name:
            filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

        filtered = np.clip(filtered, 0, 255).astype(np.uint8)
        self.last_original = img.copy()
        self.last_filtered = filtered.copy()

        return av.VideoFrame.from_ndarray(np.hstack([img, filtered]), format="bgr24")

# --- Sidebar: quick controls ---
st.sidebar.title("Select Filter:")
selected_filter = st.sidebar.selectbox("Choose", list(get_filters().keys()), index=1)
st.sidebar.markdown("###### After changing filter, restart the webcam stream to apply.")

# --- Main: place Start/Stop and Save side-by-side ---
col_left, col_right = st.columns([3, 1])

with col_left:
    st.markdown("### Start Webcam")
    st.markdown("###### Click `Select Device` if you have multiple webcams")
    webrtc_ctx = webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VideoTransformer,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# keep transformer in sync with selected filter
if webrtc_ctx.video_transformer:
    webrtc_ctx.video_transformer.filter_name = selected_filter

with col_right:
    st.markdown("##### Save Original and Convolved Frame to outputs folder")
    st.markdown("### ")  # small spacer to align with Start button height
    save_clicked = st.button("💾 Save Frame", use_container_width=True)

    if save_clicked:
        vt = webrtc_ctx.video_transformer
        if vt and vt.last_original is not None and vt.last_filtered is not None:
            os.makedirs("./outputs/out_conv", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"./outputs/out_conv/frame_{ts}_{selected_filter}"
            orig_path = f"{base}_orig.png"
            filt_path = f"{base}_filtered.png"
            ok1 = cv2.imwrite(orig_path, vt.last_original)
            ok2 = cv2.imwrite(filt_path, vt.last_filtered)
            if ok1 and ok2:
                st.success("Saved!")
                st.caption(f"• {orig_path}\n\n• {filt_path}")
            else:
                st.error("Failed to save images.")
        else:
            st.warning("Start the stream first.")

st.caption("Live feed: left = original, right = filtered")
