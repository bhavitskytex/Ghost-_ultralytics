
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import logging
import numpy as np
import cv2
import pyrealsense2 as rs
import warnings
import time
import threading

from liveinference_utlis import RtspServer

# Suppress warnings
logging.getLogger('ultralytics').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
print("LIB IMPORTED")

# Initialize GStreamer
Gst.init(None)

rtsp_frames = RtspServer()
rtsp_thread = threading.Thread(target=rtsp_frames.run_rtsp, args=("8554", "10.10.10.147"))
rtsp_thread.start()


def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    print("PIPELINE SETTED UP")
    return pipeline

def get_frames(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    color_image = np.asanyarray(color_frame.get_data())
    return color_image


def push_frames():
    pipeline = setup_realsense()

    while True:
        frame = get_frames(pipeline)

        if frame is None:
            print("No frame to read ")

        rtsp_frames.add_frame(frame)


push_frames()