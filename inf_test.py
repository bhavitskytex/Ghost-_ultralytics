import logging
from  v10utils import plot_one_box
import numpy as np 
import cv2 
import random
import gi 
import threading
import warnings
import pyrealsense2 as rs

from ultralytics import YOLO
from bytetracker_utils import BYTETracker
from collections import deque
from  liveinference_utlis import  RtspServer

gi.require_version('Gst','1.0')
from gi.repository import Gst
import time

logging.getLogger('ultralytics').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
print("LIB IMPORTED")

Gst.init(None)



PT_FILE = "/home/bhavit/Desktop/Ghost_ultralytics_v2/yolo11n.pt"
LABELS_NAMES = "/home/bhavit/Desktop/Ghost_ultralytics_v2/coco.names"
TRACK = True
RED = (0, 0, 255) 
GREEN = (0,255,0)
YELLOW = (0 ,255,255)
BLUE = (255 ,0, 0)

# global variables 
selected_track_id = None
x_point = None
y_point = None


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


def click_event(event,x,y,flags, params):
    global x_point ,y_point
    if event == cv2.EVENT_LBUTTONDOWN:
        x_point = x 
        y_point = y


# ext_patch = deque(maxlen=1)
# ext_patch.append(np.zeros((640, 480, 3), np.uint8))

def show_target(target, ext_patch=None):
    if ext_patch is None:
        print("Bkl")

    if target is None or target.size == 0:
        print("Invalid target image. Skipping display.")
        return
    target = cv2.resize(target, (224,224))
    cv2.imshow("Target", target)
    cv2.waitKey(1)  # Use 1 instead of 0 for real-time display


# tar = threading.Thread(target=show_target)
# tar.start()

names = [label.strip() for label in  open(LABELS_NAMES)]

# Initialize the YOLO model
device = 'cuda:0'
model = YOLO(PT_FILE).to(device=device)


def detect():
    global selected_track_id ,x_point ,y_point
    pipeline = setup_realsense()
    frame_count = 0
    colors = [[random.randint(0,255) for _ in range(3)] for _ in names]
    start_time = time.time()
    tracker = BYTETracker() 
    cv2.namedWindow('Detected Frame')

    while True:
        frame = get_frames(pipeline)

        if frame is  None:
            print("NO FRAMES TO READ...")
            break

        # Detect objects in the frame
        frame_count += 1
        results = model(frame)
        DET = results[0]

        if len(DET) != 0:
            track_dets = []
            for i in range(len(results[0].boxes)):
                box = results[0].boxes[i]
                clsID = int(box.cls.cpu().numpy()[0]) 
                label = results[0].names[clsID]       
                conf = box.conf.cpu().numpy()[0]    
                conf = float(f'{conf:.2f}')
                bb = box.xyxy.cpu().numpy()      
                x1, y1, x2, y2 = bb[0][0], bb[0][1], bb[0][2], bb[0][3] 
                LABELS = f"{label} {conf}"
                track_dets.append([x1, y1, x2, y2, conf, clsID])

            tracker_detection = np.array(track_dets)
            tracker_detection = tracker.update(tracker_detection)  # Update tracker


            if x_point is not None and y_point is not None:
                min_distance = float('inf')
                for det in tracker_detection:
                    x1, y1, x2, y2, track_id = det[:5]
                    if x1 <= x_point <= x2 and y1 <= y_point <= y2:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        distance = np.sqrt((x_point - center_x) ** 2 + (y_point - center_y) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            selected_track_id = int(track_id)
                
                x_point = None 
                y_point = None
            
            for det in tracker_detection:
                # print(det)
                x1, y1, x2, y2, track_id,class_id, conf_score = det
                class_id = int(class_id)
                class_names = names[int(class_id)]
                label = f"{int(track_id)} {conf_score:.2f} {class_names}"
                if selected_track_id is not None:
                    # If this detection matches our selected track ID
                    if int(track_id) == selected_track_id:
                        patch = frame[int(y1):int(y2),int(x1):int(x2)]
                        # ext_patch.append(patch)
                        show_target(patch)
                        plot_one_box([x1,y1,x2,y2],frame,label=LABELS,color=RED,line_thickness=2)
            


            elapsed_time = time.time() - start_time
            fps = frame_count/elapsed_time
            cv2.putText(frame ,f"FPS:{fps:.2f}" ,(50,50),
                        cv2.FONT_HERSHEY_SIMPLEX ,1,(0,0,255),4)
                
        # Display the frame with detection results
        resize_frame = cv2.resize(frame,(1920,1080))
        cv2.imshow('Detected Frame', resize_frame)
        cv2.setMouseCallback('Detected Frame', click_event)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

detect()





# SIMILARITY_THRESHOLD = 0.75  # tune this based on your extractor

# if selected_track_id is None and person1_features is not None:
#     for det in tracker_detections:
#         x1, y1, x2, y2, det_track_id, class_id, conf_score = det
#         det_patch = pframe[y1:y2, x1:x2]
        
#         det_feat = feature_extraction.extract_feature(det_patch)
#         if det_feat is None:
#             continue
        
#         det_feat = det_feat.flatten()
#         similarity = np.dot(person1_features, det_feat) / (
#             np.linalg.norm(person1_features) * np.linalg.norm(det_feat)
#         )
        
#         print(f"Similarity with track {det_track_id}: {similarity:.3f}")
        
#         if similarity > SIMILARITY_THRESHOLD:
#             print(f"Re-identification successful! Reassigning to track ID {det_track_id}")
#             selected_track_id = int(det_track_id)
#             break  # stop after first valid match
