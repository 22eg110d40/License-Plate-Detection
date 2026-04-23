import os
import sys
import cv2

# Import our detection logic from the main batch script
import batch_process_videos
import esrgan_utils
import torch

def predict_single_image(image_path):
    print(f"\n{'='*50}")
    print(f"🚗 LICENSE PLATE RECOGNITION (Single Image)")
    print(f"{'='*50}")

    if not os.path.exists(image_path):
        print(f"❌ Error: File not found at {image_path}")
        return

    print(f"📸 Loading image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Failed to load image. Ensure it's a valid image file.")
        return
        
    # Handle PNGs with alpha channels
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    print("🔍 Detecting license plates using YOLO model...")
    plates = batch_process_videos.detect_license_plates_yolo(frame)
    print(f"✅ Found {len(plates)} plate(s).")
    
    if plates:
        print("📝 Running ESRGAN Super-Resolution and OCR extraction...")
        
        # Load ESRGAN model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join("models", "RealESRGAN_x4plus.pth")
        sr_model = None
        if os.path.exists(model_path):
            sr_model = esrgan_utils.load_esrgan_model(model_path, device=device)
        
        # We pass sr_model and device to process_license_plates for the same robust OCR pipeline
        processed_frame, texts = batch_process_videos.process_license_plates(frame.copy(), plates, sr_model=sr_model, device=device)
        
        # Create output directories
        out_dir = "processed_images"
        roi_dir = os.path.join(out_dir, "enhanced_plates")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(roi_dir, exist_ok=True)

        if sr_model:
            for i, (x, y, w, h) in enumerate(plates):
                # Padding
                px, py, pw, ph = max(0, x-5), max(0, y-5), min(frame.shape[1]-x, w+10), min(frame.shape[0]-y, h+10)
                roi = frame[py:py+ph, px:px+pw]
                
                print(f"✨ Enhancing Plate {i+1} with ESRGAN...")
                enhanced_roi = esrgan_utils.upsample_esrgan(sr_model, roi, device=device)
                # Apply sharpening to the saved thumbnail as well
                enhanced_roi = batch_process_videos.sharpen_image(enhanced_roi)
                
                roi_path = os.path.join(roi_dir, f"plate_{i+1}_{os.path.basename(image_path)}")
                cv2.imwrite(roi_path, enhanced_roi)
                print(f"✅ Enhanced plate saved to: {roi_path}")
        
        print(f"\n{'-'*50}")
        print("🎯 FINAL RESULTS:")
        if not texts:
            print("   Plates were visually found, but no text could be recognized.")
        else:
            for i, text in enumerate(texts, 1):
                print(f"   Plate {i}:  {text}")
        print(f"{'-'*50}")
            
        # Save output
        out_path = os.path.join(out_dir, f"result_{os.path.basename(image_path)}")
        cv2.imwrite(out_path, processed_frame)
        print(f"\n💾 Saved annotated image to: {out_path}")
        
        # Automatically pop open the image on Windows!
        try:
            os.startfile(os.path.abspath(out_path))
            print("🖼️ Opening image viewer...")
        except Exception:
            pass
    else:
        print("\n❌ No license plates detected in this image.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <path_to_image>")
        print("Example: python predict_image.py sample_videos/user_images/YOLO_dataset/images/val/Cars102.png")
    else:
        predict_single_image(sys.argv[1])
