import os
import cv2
import argparse
from pathlib import Path
import batch_process_videos

def batch_process_images(input_dir, output_dir, extensions=['.png', '.jpg', '.jpeg']):
    """
    Process all images in a directory and save results as images.
    """
    print(f"\n{'='*70}")
    print(f"📸 BATCH IMAGE PROCESSING")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} image(s) to process.")
    
    for i, img_path in enumerate(image_files, 1):
        input_file = str(img_path)
        output_file = os.path.join(output_dir, f"result_{img_path.name}")
        
        print(f"\n[{i}/{len(image_files)}] Processing: {img_path.name}")
        
        # Load image
        frame = cv2.imread(input_file)
        if frame is None:
            print(f"  ✗ Error: Could not load {img_path.name}")
            continue

        # Convert RGBA to BGR if necessary
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 1. Detect plates
        plates = batch_process_videos.detect_license_plates_yolo(frame)
        
        # 2. Process (OCR + Annotated Frame)
        if plates:
            processed_frame, texts = batch_process_videos.process_license_plates(frame, plates)
            print(f"  ✓ Detected: {', '.join(texts) if texts else 'Plate found, no text recognized'}")
        else:
            processed_frame = frame
            print("  ✗ No plates detected.")
            
        # 3. Save as Image
        cv2.imwrite(output_file, processed_frame)
        print(f"  ✓ Saved to: {os.path.basename(output_file)}")

    print(f"\n{'='*70}")
    print(f"🎉 BATCH PROCESSING COMPLETE")
    print(f"Results are in: {os.path.abspath(output_dir)}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process images for license plate detection and OCR')
    parser.add_argument('--input_dir', '-i', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', '-o', required=True, help='Directory to save processed images')
    parser.add_argument('--extensions', '-e', nargs='+', default=['.png', '.jpg', '.jpeg'], help='Image extensions to process')
    
    args = parser.parse_args()
    batch_process_images(args.input_dir, args.output_dir, args.extensions)
