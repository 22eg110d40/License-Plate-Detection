import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def evaluate_sr(model_path, image_path):
    out_dir = "processed_output"
    print(f"\n{'='*50}")
    print(f"SUPER-RESOLUTION EVALUATION (PSNR/SSIM)")
    print(f"{'='*50}")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load model
    print("Loading EDSR model...")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", 4)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    # Ensure image size is divisible by 4 (to avoid artifacts in comparison)
    h, w = img.shape[:2]
    h, w = (h // 4) * 4, (w // 4) * 4
    img = img[:h, :w]

    print(f"Original Image Shape: {w}x{h}")

    # 1. Create simulated low-res image (Ground Truth -> LR)
    lr_img = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)

    # 2. Reconstruct with model (LR -> SR)
    print("Upscaling with EDSR x4 (This may take a moment)...")
    sr_img = sr.upsample(lr_img)

    # 3. Baseline: Simple bicubic upscaling (LR -> Bicubic)
    bicubic_img = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_CUBIC)

    # --- Metrics ---
    # PSNR
    psnr_edsr = psnr(img, sr_img)
    psnr_bicubic = psnr(img, bicubic_img)

    # SSIM (needs multichannel=True for color)
    ssim_edsr = ssim(img, sr_img, channel_axis=2)
    ssim_bicubic = ssim(img, bicubic_img, channel_axis=2)

    print(f"\n{'-'*50}")
    print("EVALUATION METRICS:")
    print(f"{'Method':<15} | {'PSNR (dB)':<12} | {'SSIM':<10}")
    print(f"{'-'*15} + {'-'*12} + {'-'*10}")
    print(f"{'Bicubic':<15} | {psnr_bicubic:<12.4f} | {ssim_bicubic:<10.4f}")
    print(f"{'EDSR x4':<15} | {psnr_edsr:<12.4f} | {ssim_edsr:<10.4f}")
    print(f"{'-'*50}")

    improvement_psnr = psnr_edsr - psnr_bicubic
    improvement_ssim = (ssim_edsr - ssim_bicubic) / ssim_bicubic * 100

    results_text = f"""
==================================================
      SUPER-RESOLUTION EVALUATION (PSNR/SSIM)
==================================================
Original Image Shape: {w}x{h}
--------------------------------------------------
Method          | PSNR (dB)    | SSIM      
--------------------------------------------------
Bicubic        | {psnr_bicubic:.4f}      | {ssim_bicubic:.4f}    
EDSR x4        | {psnr_edsr:.4f}      | {ssim_edsr:.4f}    
--------------------------------------------------
EDSR improves PSNR by {improvement_psnr:.2f} dB
EDSR improves SSIM by {improvement_ssim:.2f}%
==================================================
"""
    print(results_text)
    
    # Save text results
    with open(os.path.join(out_dir, "scores.txt"), "w") as f:
        f.write(results_text)

    # Save results
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "eval_original.png"), img)
    cv2.imwrite(os.path.join(out_dir, "eval_lowres.png"), lr_img)
    cv2.imwrite(os.path.join(out_dir, "eval_upscaled.png"), sr_img)
    print(f"\nSaved comparison images to: {out_dir}/")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    MODEL = os.path.join("models", "EDSR_x4.pb")
    # Use a sample from the dataset if available
    SAMPLE = os.path.join("sample_videos", "user_images", "YOLO_dataset", "images", "val", "Cars102.png")
    
    if not os.path.exists(SAMPLE):
        # Find any png in val
        val_dir = os.path.join("sample_videos", "user_images", "YOLO_dataset", "images", "val")
        if os.path.exists(val_dir):
            files = [f for f in os.listdir(val_dir) if f.endswith('.png') or f.endswith('.jpg')]
            if files:
                SAMPLE = os.path.join(val_dir, files[0])
    
    evaluate_sr(MODEL, SAMPLE)
