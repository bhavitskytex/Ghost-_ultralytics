
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from ultralytics import YOLO
from bytetracker_utils import BYTETracker
import logging
from pathlib import Path
import numpy as np
import cv2
import pyrealsense2 as rs
import struct
import socket
import pickle
import random
import warnings
from v10utils import plot_one_box ,plot_target
import time
import threading
import queue  # Added for thread-safe queue
from liveinference_utlis import RtspServer

# Suppress warnings
logging.getLogger('ultralytics').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
print("LIB IMPORTED")

# Initialize GStreamer
Gst.init(None)

rtsp_frames = RtspServer()
rtsp_thread = threading.Thread(target=rtsp_frames.run_rtsp, args=("7554", "10.10.10.147"))
rtsp_thread.start()

# Paths for model and labels
PT_FILE = "//home/bhavit/Desktop/ghost_gimbal_ultralytics/yolo11n.pt"
LABELS_NAMES = "/home/bhavit/Desktop/ghost_gimbal_ultralytics/coco.names"
TRACK = True
RED = (0, 0, 255)

# Load class names
names = [label.strip() for label in open(LABELS_NAMES)]

# Initialize the YOLO model
device = 'cuda:0'
model = YOLO(PT_FILE).to(device=device)
print("MODEL LOADED-------", model.device)
fp16 = True

# Global variables
clicked_cord = None  # Stores user's click coordinates
selected_track_id = None  # Stores selected object ID
frames_without_target = 0  # Counter for frames where target is missing
MAX_MISSING_FRAMES = 300  # Number of frames to wait before resetting tracking
points_queue = queue.Queue()  # Thread-safe queue for click coordinates

# Threaded function to receive points with debounce
def receive_points_thread():
    MCAST_GRP = '224.1.1.1'
    MCAST_PORT = 5004
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', MCAST_PORT))
    mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    
    last_point = None  # Track the last received point for debouncing
    
    while True:
        data = sock.recv(10240)
        X_Y_points = pickle.loads(data)
        
        # Only queue the point if it's different from the last one
        if X_Y_points != last_point:
            points_queue.put(X_Y_points)
            last_point = X_Y_points
            print("New click received:", X_Y_points)

# Start the point receiver thread
point_thread = threading.Thread(target=receive_points_thread, daemon=True)
point_thread.start()



def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
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



def detect():
    global clicked_cord, selected_track_id, frames_without_target
    frame_count = 0
    tracker = BYTETracker()  # Optionally configure with parameters
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    pipeline = setup_realsense()
    start_time = time.time()

    while True:
        frame = get_frames(pipeline)
        # ret, frame = cap.read()
        if frame is None:
            print("No frames to read...")
            break

        # Check for new click from the queue
        try:
            clicked_cord = points_queue.get_nowait()
            print("New click from queue:", clicked_cord)
        except queue.Empty:
            pass  # No new click available

        frame_count += 1
        results = model(frame, device=device)
        DET = results[0]

        if len(DET) != 0:
            track_dets = []
            for i in range(len(results[0].boxes)):
                box = results[0].boxes[i]
                clsID = int(box.cls.cpu().numpy()[0])
                conf = box.conf.cpu().numpy()[0]
                conf = float(f'{conf:.2f}')
                bb = box.xyxy.cpu().numpy()  # Shape (1,4)
                x1, y1, x2, y2 = bb[0]  # Unpack the coordinates
                track_dets.append([x1, y1, x2, y2, conf, clsID])

            tracker_detection = np.array(track_dets)
            tracker_detection = tracker.update(tracker_detection)  # Update tracker



            if clicked_cord is not None:
                click_x, click_y = clicked_cord
                min_distance = float('inf')
                for det in tracker_detection:
                    # Assume first 5 values are x1,y1,x2,y2,track_id
                    x1, y1, x2, y2, track_id = det[:5]
                    if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        distance = np.sqrt((click_x - center_x) ** 2 + (click_y - center_y) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            selected_track_id = int(track_id)
                clicked_cord = None  # Reset after processing
                
            # Check if selected object is present in current frame
            target_found = False

            # Draw bounding boxes
            for det in tracker_detection:
                # print(det)
                x1, y1, x2, y2, track_id,class_id, conf_score = det
                class_id = int(class_id)
                class_names = names[int(class_id)]
                label = f"{int(track_id)} {conf_score:.2f} {class_names}"
                # print("DETECTIONS", det)
                
                # If we have a selected track ID
                if selected_track_id is not None:
                    # If this detection matches our selected track ID
                    if int(track_id) == selected_track_id:
                        target_found = True
                        plot_one_box([x1, y1, x2, y2], frame, label=label, color=RED, line_thickness=2)
               
                else:
                    # No selected object, draw all detections
                    plot_one_box([x1, y1, x2, y2], frame, label=label, color=colors[class_id], line_thickness=2)

            # Update counter for missing target
            if selected_track_id is not None and not target_found:
                frames_without_target += 1
            else:
                frames_without_target = 0
            
            # If target is missing for too many frames, reset tracking
            if frames_without_target >= MAX_MISSING_FRAMES:
                # print(f"Object {selected_track_id} lost for {MAX_MISSING_FRAMES} frames, resuming normal tracking...")
                selected_track_id = None
                frames_without_target = 0
        else:
            # No detections at all
            if selected_track_id is not None:
                frames_without_target += 1
                if frames_without_target >= MAX_MISSING_FRAMES:
                    # print(f"No detections for {MAX_MISSING_FRAMES} frames, resuming normal tracking...")
                    selected_track_id = None
                    frames_without_target = 0

        # Add status text
        status_text = "Tracking All Objects"
        if selected_track_id is not None:
            status_text = f"Tracking Object ID: {selected_track_id}"
        
        # Add FPS counter and status
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        
        # Display the frame
        rtsp_frames.add_frame(frame)
    #     cv2.imshow("FEED" ,frame)
    #     time.sleep(0.05)
    #     if cv2.waitKey(1) & 0XFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()

detect()
























