import json
import os

def display_results(report_path):
    if not os.path.exists(report_path):
        print(f"Error: {report_path} not found")
        return

    with open(report_path, 'r') as f:
        data = json.load(f)

    print("\n" + "="*50)
    print("      LICENSE PLATE DETECTION & OCR RESULTS")
    print("="*50)
    print(f"Total Images: {data.get('total_videos', 0)}")
    
    # Collate all unique recognized texts
    all_texts = []
    results_found = False
    
    print(f"{'Image Filename':<25} | {'Detected Text'}")
    print("-" * 50)
    
    for entry in data.get('results', []):
        filename = entry.get('input_file', 'unknown')
        texts = entry.get('unique_plate_texts', [])
        if texts:
            results_found = True
            print(f"{filename:<25} | {', '.join(texts)}")
            all_texts.extend(texts)
            
    if not results_found:
        print("No license plates were detected in this batch.")
    
    print("="*50)
    print(f"Grand Total Unique Plate Readings: {len(set(all_texts))}")
    print("="*50 + "\n")

if __name__ == "__main__":
    report_file = os.path.join('processed_videos', 'demo_result', 'processing_report.json')
    display_results(report_file)
