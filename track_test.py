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
from v10utils import plot_one_box
import time
import threading
import queue  # Added for thread-safe queue
from liveinference_utlis import TestRtspMediaFactory

# Suppress warnings
logging.getLogger('ultralytics').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
print("LIB IMPORTED")