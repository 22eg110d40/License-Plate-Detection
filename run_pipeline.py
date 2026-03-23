import os
import cv2
import sys
from pathlib import Path

# Add the current directory to sys.path to import the pipeline function if it were in a separate file
# But here we will just include the necessary logic or import from batch_process_videos

import batch_process_videos

def run_pipeline(video_path):
    print(f"--- Pipeline Runner ---")
    print(f"Target Video: {video_path}")
    
    # 1. Check if file exists
    if not os.path.exists(video_path):
        print(f"ERROR: File not found at {os.path.abspath(video_path)}")
        return
    
    # 2. Try to open with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open the video file.")
        print(f"Possible reasons:")
        print(f" - Missing codecs (ffmpeg/gstreamer backend issues)")
        print(f" - File path contains non-ASCII characters that your OpenCV version can't handle")
        print(f" - The file is not a valid video file")
        return

    # 3. Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video detected! {width}x{height}, {fps} FPS, {frames} Total Frames.")
    cap.release()
    
    # 4. Run the batch process
    input_dir = os.path.dirname(os.path.abspath(video_path))
    output_dir = os.path.join(os.getcwd(), "processed_videos")
    
    print(f"Processing...")
    # Use the logic from batch_process_videos
    # We'll just run one file
    output_path = os.path.join(output_dir, f"processed_{os.path.splitext(os.path.basename(video_path))[0]}.avi")
    os.makedirs(output_dir, exist_ok=True)
    
    result = batch_process_videos.process_video(video_path, output_path, blur_intensity=25)
    
    if result['status'] == 'success':
        print(f"--- SUCCESS ---")
        print(f"Processed video saved to: {os.path.abspath(output_path)}")
        print(f"Plates detected: {result['plates_detected']}")
    else:
        print(f"--- FAILURE ---")
        print(f"Message: {result.get('message', 'Unknown error')}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_pipeline(sys.argv[1])
    else:
        # Default to the sample video
        sample = os.path.abspath("sample_videos/hit_and_run_sample.mp4")
        if os.path.exists(sample):
            run_pipeline(sample)
        else:
            print("No video provided and sample video not found.")
            print("Usage: python run_pipeline.py path/to/video.mp4")
