import cv2
import os
import batch_process_videos
import predict_image

def test_robustness():
    sample_path = "sample_videos/user_images/YOLO_dataset/images/val/Cars102.png"
    if not os.path.exists(sample_path):
        print("Sample image not found.")
        return

    # 1. Load and Blur the image
    img = cv2.imread(sample_path)
    # Apply Gaussian Blur to simulate low quality
    blurry_img = cv2.GaussianBlur(img, (9, 9), 0)
    
    blur_path = "processed_images/test_blurry_input.png"
    os.makedirs("processed_images", exist_ok=True)
    cv2.imwrite(blur_path, blurry_img)
    print(f"Created blurry test image: {blur_path}")

    # 2. Run Robust Detection
    print("\n🔍 Testing Robust Detection on Blurry Image...")
    # The new default threshold is 0.15 with CLAHE fallback
    plates = batch_process_videos.detect_license_plates_yolo(blurry_img)
    
    print(f"✅ Found {len(plates)} plate(s) in the blurry image!")
    
    if plates:
        # 3. Run Enhanced OCR
        print("📝 Testing Enhanced OCR...")
        # Load ESRGAN model for best results
        import esrgan_utils
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join("models", "RealESRGAN_x4plus.pth")
        sr_model = esrgan_utils.load_esrgan_model(model_path, device=device)
        
        processed_frame, texts = batch_process_videos.process_license_plates(
            blurry_img.copy(), plates, sr_model=sr_model, device=device
        )
        
        print(f"🎯 OCR RESULTS: {texts}")
        
        out_path = "processed_images/test_robust_result.png"
        cv2.imwrite(out_path, processed_frame)
        print(f"💾 Result saved to: {out_path}")
    else:
        print("❌ Failed to detect plates in blurry image. Thresholds might need further adjustment.")

if __name__ == "__main__":
    test_robustness()
