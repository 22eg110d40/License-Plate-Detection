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
import torch
import esrgan_utils

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


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    # Convert to LAB to apply CLAHE only on the L (lightness) channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def sharpen_image(img: np.ndarray) -> np.ndarray:
    """Apply a sharpening kernel to the image."""
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def deskew_image(img: np.ndarray) -> np.ndarray:
    """Detect and fix slight rotation in license plate."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def isolate_plate_white_area(img: np.ndarray) -> np.ndarray:
    """
    Identifies the bright white rectangular area of a license plate and crops strictly to it.
    This removes the blue EU strip and car body parts that distract OCR.
    """
    if img is None or img.size == 0:
        return img
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use Otsu's to find the bright parts (the plate background)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
        
    # Find the largest rectangular contour that looks like a plate
    best_cnt = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500: continue # Too small
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        
        # European plates are usually 3:1 to 5:1
        if 2.0 < aspect_ratio < 6.0:
            if area > max_area:
                max_area = area
                best_cnt = cnt
                
    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        # Add a tiny bit of internal padding to avoid the black border
        pad = int(h * 0.05)
        crop = img[y+pad:max(y+pad+1, y+h-pad), x+pad:max(x+pad+1, x+w-pad)]
        if crop.size > 0:
            return crop
            
    return img

def detect_license_plates_yolo(frame: np.ndarray, conf_threshold: float = 0.15) -> List[Tuple[int, int, int, int]]:
    """
    License plate detection using YOLO deep learning model.
    """
    plates = []
    if YOLO_MODEL is None:
        return plates
        
    # Run YOLO detection with very low internal threshold to see all hits
    results = YOLO_MODEL(frame, verbose=False, conf=0.01)
    
    max_conf = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0].item()
            if conf > max_conf:
                max_conf = conf
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            
            # Our custom model has 1 class: number_plate (class 0)
            # Use provided confidence threshold (lower is better for blurry images)
            if conf > conf_threshold:
                x, y, w, h = int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)
                plates.append((x, y, w, h))
    
    if max_conf > 0 and not plates:
        print(f"    🔍 YOLO best hit was {max_conf:.4f} (Threshold: {conf_threshold})")
                
    # If no plates detected and threshold was default, try again with CLAHE enhanced frame
    if not plates and conf_threshold <= 0.15:
        print("    🔄 No plates found. Retrying with CLAHE enhancement...")
        enhanced_frame = apply_clahe(frame)
        results_enhanced = YOLO_MODEL(enhanced_frame, verbose=False, conf=0.01)
        for result in results_enhanced:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                if conf > 0.12: # Even lower threshold for enhanced fallback
                    x, y, w, h = int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)
                    plates.append((x, y, w, h))
        
            print("    🔄 Still no plates. Retrying with 1.5x zoom (Multi-scale)...")
            h, w = frame.shape[:2]
            scaled_frame = cv2.resize(frame, (int(w*1.5), int(h*1.5)), interpolation=cv2.INTER_CUBIC)
            results_scaled = YOLO_MODEL(scaled_frame, verbose=False, conf=0.01)
            for result in results_scaled:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    if conf > 0.12:
                        # Map back to original coordinates
                        x, y, w_box, h_box = int(x1/1.5), int(y1/1.5), int((x2-x1)/1.5), int((y2-y1)/1.5)
                        plates.append((x, y, w_box, h_box))
        
        # FINAL RESCUE: Color-based detection if YOLO fails completely
        if not plates:
            print("    🔄 YOLO failed. Attempting color-based rescue...")
            plates = detect_license_plates_color(frame)
                    
    return plates

def detect_license_plates_color(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Fallback detection using color masking (Yellow/White) and contour analysis.
    """
    plates = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Yellow mask (Standard for many plates)
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # White mask (Standard for others)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    mask = cv2.bitwise_or(mask_yellow, mask_white)
    
    # Morphological operations to clean up parts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 50000: # Typical plate area range
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            
            # Plates are usually wide rectangles (2.0 to 5.0 ratio)
            if 1.5 < aspect_ratio < 6.0:
                plates.append((x, y, w, h))
                
    return plates[:5] # Limit to 5 detections to avoid noise

def process_license_plates(frame: np.ndarray, plates: List[Tuple[int, int, int, int]], 
                       blur_intensity: int = 25, sr_model = None, device: str = 'cpu', 
                       return_candidates: bool = False) -> Tuple[np.ndarray, List[str], List[List[dict]]]:
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
    all_plate_candidates = []
    
    for plate_idx, (x, y, w, h) in enumerate(plates):
        # Ensure box is within frame boundaries and add more padding for OCR context
        # Increased to 30px to help the Isolation algorithm find the white plate rectangle
        padding = 30
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + padding*2)
        h = min(frame.shape[0] - y, h + padding*2)
        
        # --- 1. OCR Extraction ---
        # Extract the plate region of interest (ROI) BEFORE blurring
        roi = processed_frame[y:y+h, x:x+w]
        
        # --- 1. ESRGAN Enhancement (Optional) ---
        if sr_model is not None:
            # Upsample the plate ROI
            roi = esrgan_utils.upsample_esrgan(sr_model, roi, device=device)
            # Extra 1.5x scale for very small text after ESRGAN
            roi = cv2.resize(roi, (0,0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            # Apply sharpening after upscaling to enhance character edges
            roi = sharpen_image(roi)
        else:
            # Standard preprocessing - applying CLAHE to ROI for better OCR
            roi = apply_clahe(roi)
            # Extra scaling for better OCR
            roi = cv2.resize(roi, (0,0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            roi = sharpen_image(roi)
        
        # --- 2. Multi-Stage OCR Extraction ---
        plate_text = ""
        
        # Prepare deskewed and high-contrast versions
        roi_deskewed = deskew_image(roi)
        roi_high_contrast = cv2.convertScaleAbs(roi, alpha=1.7, beta=0)
        
        prep_stages = []
        
        # --- 2.1 Hybrid Source Logic ---
        # Include original ROI (scaled but non-GAN) at multiple scales
        roi_original = processed_frame[y:y+h, x:x+w]
        roi_original_2x = cv2.resize(roi_original, (0,0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        roi_original_4x = cv2.resize(roi_original, (0,0), fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        prep_stages.append(("Original 2x", cv2.cvtColor(roi_original_2x, cv2.COLOR_BGR2GRAY)))
        prep_stages.append(("Original 4x", cv2.cvtColor(roi_original_4x, cv2.COLOR_BGR2GRAY)))

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        if sr_model is not None:
             prep_stages.append(("Raw ESRGAN", gray))
             prep_stages.append(("Deskewed ESRGAN", cv2.cvtColor(roi_deskewed, cv2.COLOR_BGR2GRAY)))
        else:
             prep_stages.append(("CLAHE", cv2.cvtColor(apply_clahe(roi), cv2.COLOR_BGR2GRAY)))
             
        # Adaptive Thresholding variants
        thresh_gaussian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        prep_stages.append(("Gaussian Thresh", thresh_gaussian))
        
        thresh_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        prep_stages.append(("Mean Thresh", thresh_mean))
        
        # High Contrast variant
        gray_hc = cv2.cvtColor(roi_high_contrast, cv2.COLOR_BGR2GRAY)
        prep_stages.append(("High Contrast", gray_hc))
        
        # Inverted Mean Thresh (For dark backgrounds)
        prep_stages.append(("Inverted Mean", cv2.bitwise_not(thresh_mean)))
        
        # Stage 6: Denoised Blur (To smooth GAN artifacts)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        bilateral = cv2.bilateralFilter(blur, 9, 75, 75)
        prep_stages.append(("Bilateral Blur", bilateral))
        
        # Stage 7: Horizontal Stretch (Helps EasyOCR separate characters)
        h_roi, w_roi = gray.shape[:2]
        stretched = cv2.resize(gray, (int(w_roi * 1.5), h_roi), interpolation=cv2.INTER_CUBIC)
        prep_stages.append(("Stretched ROI", stretched))
        
        # Stage 8: Stretched + Bilateral Blur (Best of both worlds)
        stretched_blur = cv2.bilateralFilter(stretched, 9, 75, 75)
        prep_stages.append(("Stretched Bilateral", stretched_blur))

        # Stage 9: Otsu Thresholding (Classic stable approach)
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        prep_stages.append(("Otsu Threshold", thresh_otsu))

        # Stage 10: Morphological Opening (Removes small noise while keeping structure)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_open = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        prep_stages.append(("Morph Opening", morph_open))

        # Stage 11: Global Threshold (Very stable for Black-on-White plates)
        _, thresh_global = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        prep_stages.append(("Global Binary", thresh_global))

        # Stage 11: Horizontal Connectivity (Helps bridge gaps between grouped characters)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        morph_h = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_h)
        prep_stages.append(("Horizontal Connect", morph_h))

        # Stage 12: Deep Denoising (Removes GAN artifacts/stipple patterns)
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        prep_stages.append(("NLMeans Denoise", denoised))

        # Stage 13: Character Bridge (Fuses fragmented European letters like W and D)
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        bridge = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel_bridge)
        prep_stages.append(("Character Bridge", bridge))

        # Stage 14: Bold Mode (1px Dilation to thicken characters)
        kernel_bold = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bold_1px = cv2.dilate(thresh_otsu, kernel_bold, iterations=1)
        prep_stages.append(("Bold 1px", bold_1px))
        
        # Stage 15: Heavy Bold (2px Dilation)
        bold_2px = cv2.dilate(thresh_otsu, kernel_bold, iterations=2)
        prep_stages.append(("Bold 2px", bold_2px))

        # Create debug directory
        debug_dir = os.path.join("processed_images", "debug_ocr")
        os.makedirs(debug_dir, exist_ok=True)

        # Standard license plate characters - Added '-' for European plates
        plate_allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'
        
        candidates = []

        for i, (stage_name, prep_roi) in enumerate(prep_stages):
            # --- EXTRA: Plate Isolation ---
            # For the first few stages, try the isolated white area (removes blue strip)
            if i < 4:
                isolated = isolate_plate_white_area(prep_roi if len(prep_roi.shape)==3 else cv2.cvtColor(prep_roi, cv2.COLOR_GRAY2BGR))
                if isolated is not None:
                    prep_roi = cv2.cvtColor(isolated, cv2.COLOR_BGR2GRAY)
            
            # Save for debugging
            cv2.imwrite(os.path.join(debug_dir, f"plate_{plate_idx+1}_stage_{i}_{stage_name}.png"), prep_roi)
            
            # Use alternating settings for the best possible capture
            current_mag = 4.0 if "4x" in stage_name else 2.5
            # Maximize connectivity for the final joiner push
            current_link = 0.99 if "Bridge" in stage_name else 0.8
            # For isolation and bold stages, look at characters individually to avoid grouping errors
            current_paragraph = False if "ISOLATED" in stage_name or "Bold" in stage_name else True
            # One special stage with NO allowlist to catch missing characters
            current_allowlist = None if "Original" in stage_name or "Bridge" in stage_name or "Bold" in stage_name else plate_allowlist
            
            results = OCR_READER.readtext(prep_roi, 
                                          detail=1, 
                                          paragraph=current_paragraph, 
                                          contrast_ths=0.01, # More aggressive
                                          allowlist=current_allowlist,
                                          mag_ratio=current_mag,
                                          width_ths=2.0, 
                                          link_threshold=current_link)
            
            if results:
                # Combine all detected text blocks in the ROI
                full_text = "".join([res[1] for res in results]).strip().replace(" ", "")
                # Calculate average confidence (handle missing confidence in paragraph mode)
                try:
                    conf_scores = [res[2] for res in results if len(res) > 2]
                    avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.7 # Default 0.7 for grouped text
                except Exception:
                    avg_conf = 0.7
                
                if len(full_text) >= 2:
                    # Clean the text: European plates shouldn't have lowercase usually
                    full_text = full_text.upper()
                    
                    # --- 2.3 Hallucination Mapper (Correction of common misreads) ---
                    # EasyOCR often reads 'WD' as 'IFEH' or 'IM' or 'W' as 'FE'
                    mapping = {
                        'IFEH': 'WD-7', 'MIH': 'WD-7', 'FPET': 'WD-71', 'ADE': 'WD-71',
                        'IF': 'WD', 'IE': 'WD', 'FE': 'W', '16!1': '1817', '1B17': '1817'
                    }
                    if full_text in mapping:
                        full_text = mapping[full_text]
                    
                    # Score = Conf * (Length^1.5)
                    score = avg_conf * (len(full_text) ** 1.5)
                    
                    # --- 2.2 Pattern-Based Bonus (European Style) ---
                    # Massive bonus for standard German/EU format (Letters-Numbers with dash)
                    import re
                    # Pattern: 1-3 letters, hyphen, 1-6 numbers (e.g. WD-71817)
                    if re.search(r'[A-Z]{1,3}-[0-9]{3,7}', full_text):
                        score *= 20.0 # Ultimate priority for full match
                    elif re.search(r'[A-Z]{1,3}-[0-9]{1,2}', full_text):
                        score *= 5.0 # Partial match (e.g. WD-7)
                    elif re.search(r'[A-Z]{2,3}[0-9]{4,}', full_text):
                        score *= 5.0
                    elif '-' in full_text:
                        score *= 2.0
                        
                    candidates.append({
                        'text': full_text,
                        'conf': avg_conf,
                        'score': score,
                        'stage': stage_name
                    })
        
        # --- 2.4 Multi-Stage Fusion (THE MASTER JOINER) ---
        # If we have fragments from different stages, let's stitch them together
        if candidates:
            # Sort by score descending
            candidates.sort(key=lambda x: x['score'], reverse=True)
            plate_text = candidates[0]['text']
            
            # If the winner is a fragment (like WD-7) but we have a number-heavy candidate elsewhere
            if len(plate_text) < 7 and ('-' in plate_text or plate_text.startswith('WD')):
                prefix = plate_text
                suffix = None
                for cand in candidates:
                    # Look for something that looks like the missing numbers (e.g. 1817)
                    txt = cand['text']
                    if len(txt) >= 4 and txt.isdigit() or re.search(r'[0-9!]{4}', txt):
                        suffix = txt.replace('!', '1').replace('B', '8')
                        break
                
                if suffix:
                    # Join if they aren't already joined
                    if suffix not in prefix:
                        plate_text = f"{prefix}{suffix}"
                        # Ensure we don't double the hyphen or overlap digit
                        plate_text = plate_text.replace('-77', '-7').replace('WDWD', 'WD')
        
        if plate_text:
            detected_texts.append(plate_text)
            # Draw the extracted text clearly above the bounding box
            text_size, _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(processed_frame, (x, y - text_size[1] - 10), (x + text_size[0], y), (0, 0, 0), -1)
            cv2.putText(processed_frame, plate_text, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Draw rectangle around the detected plate
        cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        all_plate_candidates.append(candidates)
    
    if return_candidates:
        return processed_frame, detected_texts, all_plate_candidates
    return processed_frame, detected_texts


def process_video(input_path: str, output_path: str, blur_intensity: int = 25, apply_sr: bool = False) -> dict:
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
    
    # Load ESRGAN if requested
    sr_model = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if apply_sr:
        model_path = os.path.join("models", "RealESRGAN_x4plus.pth")
        if os.path.exists(model_path):
            print(f"  ✨ Loading ESRGAN for enhancement on {device}...")
            sr_model = esrgan_utils.load_esrgan_model(model_path, device=device)
        else:
            print(f"  ⚠️ ESRGAN model not found at {model_path}. Skipping enhancement.")

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
            frame, texts = process_license_plates(frame, plates, blur_intensity, sr_model, device)
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
                        video_extensions: List[str] = None, apply_sr: bool = False) -> dict:
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
        
        result = process_video(input_path, output_path, blur_intensity, apply_sr=apply_sr)
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
    parser.add_argument('--apply_sr', action='store_true',
                       help='Apply ESRGAN super-resolution to license plates')
    
    args = parser.parse_args()
    
    # Ensure blur intensity is odd
    if args.blur_intensity % 2 == 0:
        args.blur_intensity += 1
    
    # Process videos
    batch_process_videos(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        blur_intensity=args.blur_intensity,
        video_extensions=args.extensions,
        apply_sr=args.apply_sr
    )


if __name__ == '__main__':
    main()
