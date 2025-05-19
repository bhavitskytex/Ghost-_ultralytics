from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/home/bhavit/Desktop/Ghost_ultralytics_v2/best_fixed_np.pt")

# Export the model to TensorRT format
model.export(half=True ,format="engine" )  # creates 'yolo11n.engine'