import logging
import warnings
import numpy as np
import cv2
import pyrealsense2 as rs
import random
from ultralytics import YOLO




# Suppress warnings
logging.getLogger('ultralytics').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

print("LIB IMPORTED")

# Setup RealSense pipeline
def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    pipeline.start(config)
    print("PIPELINE SET UP")
    return pipeline

# Get frame from RealSense
def get_frames(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    color_image = np.asanyarray(color_frame.get_data())
    return color_image

# Paths
ENGINE_PATH = "/home/bhavit/Desktop/Ghost_ultralytics_v2/yolo11n.engine"
LABELS_PATH = "/home/bhavit/Desktop/Ghost_ultralytics_v2/labels.names"
IMG_H, IMG_W = 1024, 1024

# Load class labels
with open(LABELS_PATH, "r") as f:
    names = [line.strip() for line in f.readlines()]
print("Labels:", names)

# Load model
model = YOLO(ENGINE_PATH)
print("MODEL LOADED")

def detect():
    pipeline = setup_realsense()
    while True:
        frame = get_frames(pipeline)
        if frame is None:
            print("No frame received.")
            continue

        print("Original frame shape:", frame.shape)
        # frame = letterbox(frame, (1024, 1024), stride=32, auto=False)[0]
        # frame_resized = cv2.resize(frame, (IMG_W, IMG_H))


        # # Run inference (Ultralytics handles letterboxing internally)
        results = model(frame, verbose=True)
        # print(results)

   
# Run detection loop
detect()
