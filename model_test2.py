import logging
import warnings
import numpy as np
from ultralytics import YOLO
import torch

# Suppress warnings
logging.getLogger('ultralytics').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

print("LIB IMPORTED")

# Engine file path
ENGINE_FILE = "/home/bhavit/Desktop/Ghost_ultralytics_v2/best_fixed_np.engine"
IMG_H, IMG_W = 1024, 1024  

if not torch.cuda.is_available():
    print("CUDA NOT AVAILABLE ::::::::::::::")

print(f"Loading engine: {ENGINE_FILE}")
model = YOLO(ENGINE_FILE)
print("Engine loaded.")


# dummy_input = np.random.randint(0, 255, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
# dummy_input = dummy_input.transpose(2, 0, 1)
# dummy_input = np.expand_dims(dummy_input, axis=0)


dummy_input = np.random.randint(0, 255, size=(IMG_H, IMG_W, 3), dtype=np.uint8)

# # Optional: Convert to torch.Tensor if needed explicitly
# dummy_tensor = torch.from_numpy(dummy_input).to("cuda")

print(f"Performing first inference with imgsz=({IMG_H},{IMG_W})...")
results = model(dummy_input, imgsz=(IMG_H, IMG_W), verbose=True)
print("Inference successful.")
print(results[0].boxes)
