# License Plate Detection, Blurring & Super-Resolution Pipeline

This project provides a complete end-to-end pipeline for processing video files containing vehicles with license plates. It combines YOLOv5 object detection, license plate blurring for privacy protection, and GAN-based super-resolution enhancement.

## 🎯 Features

- **License Plate Detection**: Using YOLOv5 for accurate plate detection
- **Privacy Protection**: Automatic blurring of detected license plates
- **Super-Resolution**: GAN-based enhancement for low-quality images
- **Video Processing**: Complete pipeline for processing video files
- **Hit-and-Run Cases**: Purpose-built for surveillance video analysis

***

## 📁 Project Structure

```
License_plate_detection_and_super_resolution-main/
├── Video_Processing_Pipeline.ipynb          # NEW: Complete video processing pipeline
├── YOLOv5_object_detection.ipynb            # Train YOLO model for detection
├── GAN_Super-Resolution.ipynb               # Train GAN for super-resolution
├── README.md                                # This file
├── sample_videos/                           # Generated sample videos
├── processed_videos/                        # Output videos with blurred plates
└── extracted_frames/                        # Temporary frame storage
```

***

## 🚀 Quick Start

### Option 1: Use Pre-built Pipeline (Recommended)

1. Open `Video_Processing_Pipeline.ipynb`
2. Run all cells sequentially
3. The notebook will:
   - Create a sample video with license plates
   - Extract frames from video
   - Detect and blur license plates
   - Apply enhancement
   - Reconstruct processed video

### Option 2: Train Models First

1. **Train YOLO Detection Model**:
   - Open `YOLOv5_object_detection.ipynb`
   - Follow instructions to train on your dataset
   - Save trained weights

2. **Train GAN Super-Resolution Model** (optional):
   - Open `GAN_Super-Resolution.ipynb`
   - Train on license plate images
   - Save trained model

3. **Process Videos**:
   - Open `Video_Processing_Pipeline.ipynb`
   - Update paths to your trained models
   - Process your videos

### Option 3: Train on a Custom Dataset (e.g. Roboflow)

If you want to train the YOLOv5 model on your own annotated license plate dataset, follow these steps:

1. **Get a Dataset:** Create a free account at [Roboflow Universe](https://universe.roboflow.com/) and find a "License Plate" dataset.
2. **Download Code:** Click "Download Dataset", select "YOLOv5 PyTorch", and choose "Show download code". Copy the provided Python snippet.
3. **Update Notebook:** Open `YOLOv5_object_detection.ipynb` and locate the cell (around step 3) that downloads the dataset. Replace it with your Roboflow snippet:
   ```python
   !pip install roboflow
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY_HERE")
   project = rf.workspace("workspace-name").project("project-name")
   dataset = project.version(1).download("yolov5")
   ```
4. **Train using GPU:** Upload the notebook to Google Colab, set the Runtime to **GPU**, and run all cells.
5. **Download Weights:** When finished, download the `best.pt` file from `/content/yolov5/runs/train/.../weights/best.pt`. Move it to your local project folder and use it in your detection scripts!

***

## 📝 Dataset

* Dataset will be provided on the basis of request.
* For custom datasets, use labelme tool for annotation
* Annotated JSON files contain license plate bounding boxes

***

## 🔧 Preprocessing

1. **Data Collection**: Gather vehicle images with visible license plates
2. **Train/Test Split**: Divide dataset for model training and validation
3. **Annotation**: Use labelme tool to mark license plate regions
4. **Output**: JSON files with annotated bounding boxes

***

## 🎨 Model Architecture

### Object Detection: YOLOv5
YOLO (You Only Look Once) is a real-time object detection algorithm that:
- Divides input image into grids
- Predicts bounding boxes and class probabilities
- YOLOv5 offers improved accuracy and efficiency

### Super-Resolution: GAN
Generative Adversarial Network (GAN) architecture for:
- Upscaling low-resolution license plate images
- Generating high-resolution, realistic images
- Generator creates enhanced images
- Discriminator evaluates authenticity

***

## 🎬 Video Processing Pipeline

The new `Video_Processing_Pipeline.ipynb` notebook provides:

1. **Sample Video Generation**: Create test videos with simulated license plates
2. **Frame Extraction**: Extract individual frames from video files
3. **Plate Detection**: Detect license plates using YOLO or color detection
4. **Privacy Blurring**: Apply Gaussian or pixelation blur to plates
5. **Enhancement**: CLAHE and other enhancement techniques
6. **Super-Resolution**: Optional upscaling for blurred regions
7. **Video Reconstruction**: Combine processed frames back to video
8. **Quality Verification**: Compare input/output quality metrics

***

## 💼 Use Cases

### Hit-and-Run Case Processing
- Process surveillance footage
- Blur license plates for public sharing
- Maintain original evidence separately
- Generate processing reports with timestamps

### Privacy-Compliant Surveillance
- Share footage publicly while protecting privacy
- Comply with GDPR and privacy regulations
- Maintain audit trails

***

## 🛠️ Technical Details

### Supported Blur Methods
- **Gaussian Blur**: Smooth blurring effect
- **Pixelation**: Mosaic-style obscuring

### Detection Methods
- **YOLOv5 (Production)**: Trained deep learning model
- **Color-Based (Demo)**: Simple yellow color detection

### Enhancement Techniques
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Super-Resolution**: GAN-based upscaling
- **Frame Enhancement**: Brightness and clarity improvements

***

## 📊 Output Formats

- **Processed Videos**: MP4 format with blurred plates
- **Frame Comparisons**: Side-by-side before/after images
- **Processing Reports**: JSON format with statistics
- **Quality Metrics**: Resolution, FPS, file size comparisons

***

## ⚙️ Configuration Options

```python
process_video_pipeline(
    input_video="input.mp4",
    output_video="output.mp4",
    blur_intensity=25,         # Odd number, 15-51 recommended
    blur_method='gaussian',    # 'gaussian' or 'pixelate'
    apply_sr=False,           # Enable super-resolution
    enhance=True,             # Apply frame enhancement
    skip_processing=False     # Skip processing, just copy
)
```

***

## 🔗 Integration

### With YOLOv5
After training your model in `YOLOv5_object_detection.ipynb`:
```python
yolo_weights = "./yolov5/runs/train/yolov5s_results/weights/best.pt"
detect_plates_with_yolo(frame, yolo_weights, conf_threshold=0.4)
```

### With GAN Super-Resolution
After training your model in `GAN_Super-Resolution.ipynb`:
```python
sr_model = load_sr_model("./models/generator.h5")
enhanced_frame = apply_super_resolution(frame, sr_model)
```

***

## 📋 Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- PyTorch (for YOLOv5)
- TensorFlow/Keras (for GAN)
- Matplotlib (for visualization)

***

## 📄 License

License plate detection and cropping using Ultralytics YOLOv5 object detection model and upscaling blurred images using Super-Resolution GAN (SR-GAN).

***

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

***

## 📧 Contact

For dataset requests or questions, please contact through GitHub issues.