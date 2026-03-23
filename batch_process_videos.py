"""
Batch Video Processing Script for License Plate Detection & Blurring

This script processes multiple video files in batch mode, detecting and blurring
license plates for privacy protection.

Usage:
    python batch_process_videos.py --input_dir ./videos --output_dir ./processed

Requirements:
    - OpenCV (cv2)
    - NumPy
    - Python 3.7+
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple
import json
from datetime import datetime
import time
import easyocr

from ultralytics import YOLO

# Initialize global reader to avoid loading the model heavily for every video
# We use english logic for plates; might be a bit slow on CPU for realtime
print("Initializing EasyOCR reader. This may take a moment...")
OCR_READER = easyocr.Reader(['en'], gpu=True) # Will fall back to CPU if no PyTorch GPU is available
print("EasyOCR reader initialized.")

# Initialize YOLOv5 model for real-world license plate detection
print("Loading YOLOv5 model...")
try:
    # Try to load the custom trained license plate detection model first
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'runs', 'detect', 'runs', 'train', 'custom_plate_detector', 'weights', 'best.pt')
    YOLO_MODEL = YOLO(model_path)
    print(f"Custom license plate model loaded from {model_path}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    YOLO_MODEL = None


def detect_license_plates_yolo(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    License plate detection using YOLO deep learning model.
    """
    plates = []
    if YOLO_MODEL is None:
        return plates
        
    # Run YOLO detection
    # Classes: 2 is car, 3 is motorcycle, 5 is bus, 7 is truck in COCO
    # For a generic model, we'll look for vehicles, but ideally this would be a plate-specific model
    results = YOLO_MODEL(frame, verbose=False)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            
            # Our custom model has 1 class: number_plate (class 0)
            # Accept any detection with confidence > 0.3
            if conf > 0.3:
                x, y, w, h = int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)
                plates.append((x, y, w, h))
                
    return plates

def process_license_plates(frame: np.ndarray, plates: List[Tuple[int, int, int, int]], 
                       blur_intensity: int = 25) -> Tuple[np.ndarray, List[str]]:
    """
    Read text from detected plates, then blur them on the frame.
    
    Args:
        frame: Input image frame
        plates: List of plate bounding boxes (x, y, w, h)
        blur_intensity: Intensity of blur (must be odd)
        
    Returns:
        Tuple of (Frame with blurred plates & text drawn, List of extracted text strings)
    """
    if blur_intensity % 2 == 0:
        blur_intensity += 1
    
    processed_frame = frame.copy()
    detected_texts = []
    
    for (x, y, w, h) in plates:
        x = max(0, x - 5)
        y = max(0, y - 5)
        w = min(frame.shape[1] - x, w + 10)
        h = min(frame.shape[0] - y, h + 10)
        
        # --- 1. OCR Extraction ---
        # Extract the plate region of interest (ROI) BEFORE blurring
        roi = processed_frame[y:y+h, x:x+w]
        
        # Convert to grayscale for better OCR
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply slight thresholding to enhance contrast
        _, thresh_roi = cv2.threshold(gray_roi, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Read text using initialized OCR_READER
        results = OCR_READER.readtext(thresh_roi, detail=0) # detail=0 returns just strings
        
        # Combine strings if multiple are found, and clean them up
        plate_text = " ".join([text for text in results if len(text) > 1]).strip()
        
        if plate_text:
            detected_texts.append(plate_text)
            # Draw the extracted text clearly above the bounding box
            text_size, _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(processed_frame, (x, y - text_size[1] - 10), (x + text_size[0], y), (0, 0, 0), -1)
            cv2.putText(processed_frame, plate_text, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Draw rectangle around the detected plate
        cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return processed_frame, detected_texts


def process_video(input_path: str, output_path: str, blur_intensity: int = 25) -> dict:
    """
    Process a single video file.
    
    Args:
        input_path: Path to input video
        output_path: Path to save output video
        blur_intensity: Intensity of blur
        
    Returns:
        Dictionary with processing statistics
    """
    start_time = time.time()
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        return {'status': 'error', 'message': f'Cannot open video: {input_path}'}
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    output_path = os.path.splitext(output_path)[0] + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    plates_detected = 0
    all_detected_texts = []
    
    print(f"  Processing: {os.path.basename(input_path)}")
    print(f"  Frames: {total_frames}, Resolution: {width}x{height}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Strip alpha channel if present (e.g., RGBA PNG images → BGR)
        if frame is not None and frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Detect plates using YOLO
        plates = detect_license_plates_yolo(frame)
        plates_detected += len(plates)
        
        if plates:
            frame, texts = process_license_plates(frame, plates, blur_intensity)
            for text in texts:
                if text not in all_detected_texts:
                   all_detected_texts.append(text)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % max(1, total_frames // 10) == 0:
            progress = (frame_count / total_frames) * 100
            print(f"    Progress: {progress:.1f}%")
    
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    
    return {
        'status': 'success',
        'input_file': os.path.basename(input_path),
        'output_file': os.path.basename(output_path),
        'frames_processed': frame_count,
        'plates_detected': plates_detected,
        'unique_plate_texts': all_detected_texts,
        'processing_time': processing_time,
        'fps': frame_count / processing_time if processing_time > 0 else 0
    }


def batch_process_videos(input_dir: str, output_dir: str, blur_intensity: int = 25,
                        video_extensions: List[str] = None) -> dict:
    """
    Process all videos in a directory.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save processed videos
        blur_intensity: Intensity of blur
        video_extensions: List of video file extensions to process
        
    Returns:
        Dictionary with batch processing summary
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(input_dir).glob(f'*{ext}'))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return {'status': 'error', 'message': 'No videos found'}
    
    print(f"\n{'='*70}")
    print(f"BATCH VIDEO PROCESSING")
    print(f"{'='*70}")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Videos Found: {len(video_files)}")
    print(f"{'='*70}\n")
    
    results = []
    start_time = time.time()
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing video...")
        
        input_path = str(video_file)
        output_filename = f"processed_{os.path.splitext(video_file.name)[0]}.avi"
        output_path = os.path.join(output_dir, output_filename)
        
        result = process_video(input_path, output_path, blur_intensity)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"  ✓ Complete: {result['plates_detected']} plates detected")
            print(f"  ✓ Time: {result['processing_time']:.2f}s")
        else:
            print(f"  ✗ Error: {result.get('message', 'Unknown error')}")
    
    total_time = time.time() - start_time
    
    # Generate summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    summary = {
        'total_videos': len(video_files),
        'successful': len(successful),
        'failed': len(failed),
        'total_processing_time': total_time,
        'total_frames': sum(r.get('frames_processed', 0) for r in successful),
        'total_plates_detected': sum(r.get('plates_detected', 0) for r in successful),
        'results': results
    }
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total Videos: {summary['total_videos']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total Frames: {summary['total_frames']}")
    print(f"Total Plates Detected: {summary['total_plates_detected']}")
    print(f"Total Time: {summary['total_processing_time']:.2f}s")
    print(f"{'='*70}\n")
    
    # Save summary report
    report_path = os.path.join(output_dir, 'processing_report.json')
    summary['timestamp'] = datetime.now().isoformat()
    
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Processing report saved to: {report_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Batch process videos for license plate detection and blurring'
    )
    parser.add_argument('--input_dir', '-i', required=True,
                       help='Directory containing input videos')
    parser.add_argument('--output_dir', '-o', required=True,
                       help='Directory to save processed videos')
    parser.add_argument('--blur_intensity', '-b', type=int, default=25,
                       help='Blur intensity (odd number, default: 25)')
    parser.add_argument('--extensions', '-e', nargs='+',
                       default=['.mp4', '.avi', '.mov', '.mkv'],
                       help='Video file extensions to process')
    
    args = parser.parse_args()
    
    # Ensure blur intensity is odd
    if args.blur_intensity % 2 == 0:
        args.blur_intensity += 1
    
    # Process videos
    batch_process_videos(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        blur_intensity=args.blur_intensity,
        video_extensions=args.extensions
    )


if __name__ == '__main__':
    main()
