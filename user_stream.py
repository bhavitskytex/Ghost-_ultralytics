import cv2
from collections import deque
import gi
gi.require_version('Gst', '1.0')
import numpy as np
from gi.repository import Gst
from  v10utils import plot_target
import threading
import socket
import pickle
import time


# Initialize GStreamer
Gst.init(None)

# # Coordinates deque with a maximum length of 1
# coordinates = deque(maxlen=1)

# Define the GStreamer pipeline
frame_capture_pipeline = (
    'rtspsrc location=rtsp://10.10.10.147:8554/stream1  latency=1 ! '
    'rtph264depay ! '
    'h264parse ! '
    'avdec_h264 ! '          
    'videoconvert ! '  
    'video/x-raw,format=RGB! '  
    'appsink emit-signals=False sync=true max-buffers=1 drop=true name=sink'
)



def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        group = '224.1.1.1'
        port = 5004

        # 2-hop restriction in network
        ttl = 2

        sock = socket.socket(socket.AF_INET,
                            socket.SOCK_DGRAM,
                            socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP,
                        socket.IP_MULTICAST_TTL,
                        ttl)

        data = pickle.dumps((x, y))
        sock.sendto(data, (group, port))
        print(f"Sent click coordinates: ({x}, {y})")
          # Small delay to prevent flooding the network
        print(f"Click registered at: ({x}, {y})")


# Function to grab frames from appsink
def grab_frame(appsink):
    sample = appsink.emit('pull-sample')  
    if sample:
        buf = sample.get_buffer()  
        caps = sample.get_caps() 
        
        # Extract width and height from caps
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')
        
        # Extract data from the buffer
        data = buf.extract_dup(0, buf.get_size())
        rgb_frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        
        # Convert RGB to BGR using OpenCV
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
 
        return bgr_frame
    
    return None

# Create and configure the pipeline
pipeline = Gst.parse_launch(frame_capture_pipeline)
appsink = pipeline.get_by_name('sink')
appsink.set_property('emit_signals', False)
appsink.set_property('sync', False)

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)
print("RTSP stream started, connecting...")

def stream():
    window_name = "RTSP stream"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)    
    print("Click on objects in the stream to track them")

    while True:
        frame = grab_frame(appsink)
        if frame is None:
            print("NO FRAMES TO DETECT")
            break
        


        frame = plot_target(frame)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()

# Start the main stream loop
stream()


# gst-launch-1.0 -v rtspsrc location=rtsp://196.21.92.82/axis-media/media.amp ! rtpjitterbuffer ! rtph264depay ! h264parse ! avdec_h264 ! queue ! autovideosink sync=true
