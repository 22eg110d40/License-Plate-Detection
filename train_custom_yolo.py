from ultralytics import YOLO

print("Initializing YOLOv5 training...")
# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov5s.pt') 

# Train the model
# We use 5 epochs for a quick demonstration. In reality, it needs 50-100.
results = model.train(
    data='sample_videos/user_images/YOLO_dataset/data.yaml',
    epochs=25,
    imgsz=640,
    project='runs/train',
    name='custom_plate_detector',
    exist_ok=True
)
print("Training complete! Best weights saved to runs/train/custom_plate_detector/weights/best.pt")
