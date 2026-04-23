import streamlit as st
import cv2
import numpy as np
import os
import torch
import time
from PIL import Image
import batch_process_videos as bpv
import esrgan_utils
from ultralytics import YOLO

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PlateEnhance AI | License Plate Recovery",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CUSTOM THEME & CSS ---
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Outfit:wght@400;700&display=swap');

    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* Result Section */
    .result-text {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        color: #10b981;
        text-align: center;
        background: rgba(16, 185, 129, 0.1);
        border: 2px dashed #10b981;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        text-transform: uppercase;
    }

    /* File Uploader Customization */
    .stFileUploader section {
        background: rgba(56, 189, 248, 0.02) !important;
        border: 2px dashed rgba(56, 189, 248, 0.2) !important;
        border-radius: 15px !important;
    }

    /* Status Icons */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-blue { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_app_resources():
    # The models are already initialized as globals in batch_process_videos
    # but we want to ensure they are ready and ESRGAN is loaded for the app
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check if ESRGAN is already loaded in bpv or needs loading
    # bpv doesn't have a global SR_MODEL, it loads it inside process_video
    # Let's load it here and cache it
    sr_model_path = os.path.join("models", "RealESRGAN_x4plus.pth")
    sr_model = None
    if os.path.exists(sr_model_path):
        sr_model = esrgan_utils.load_esrgan_model(sr_model_path, device=device)
        
    return sr_model, device

# --- PROCESSING PIPELINE ---
def process_recovery(img_array, sr_model, device):
    # Detection
    with st.status("🔍 Analyzing Image Pipeline...", expanded=True) as status:
        st.write("Detecting license plates using YOLOv5...")
        # bpv.YOLO_MODEL is a global in that module
        plates = bpv.detect_license_plates_yolo(img_array)
        
        if not plates:
            status.update(label="❌ No Plates Detected", state="error", expanded=False)
            return None, None, None
        
        st.write(f"✅ Found {len(plates)} plate candidates.")
        st.write("Enhancing resolution with ESRGAN x4+...")
        
        # Process plates (Enhancement + OCR)
        processed_frame, detected_texts = bpv.process_license_plates(
            img_array.copy(), 
            plates, 
            sr_model=sr_model, 
            device=device
        )
        
        # Crop the first detected plate for display
        x, y, w, h = plates[0]
        # ESRGAN specific crop (higher quality)
        plate_roi = img_array[y:y+h, x:x+w]
        enhanced_plate = esrgan_utils.upsample_esrgan(sr_model, plate_roi, device=device)
        enhanced_plate = bpv.sharpen_image(enhanced_plate)
        
        status.update(label="✨ Optimization Complete!", state="complete", expanded=False)
        
    return processed_frame, enhanced_plate, detected_texts

# --- MAIN UI ---
def main():
    # Hero Header
    st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">PlateEnhance AI</h1>
            <p style="font-size: 1.2rem; color: #94a3b8; max-width: 700px; margin: 0 auto;">
                Deep Learning pipeline for License Plate Detection and Super-Resolution Recovery. 
                Upload a blurred or low-res image to extract plate numbers with extreme precision.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar / Settings
    with st.sidebar:
        st.image("https://img.icons8.com/isometric/100/car-badge.png", width=80)
        st.title("Settings")
        conf_thresh = st.slider("Detection Confidence", 0.05, 0.95, 0.20)
        debug_mode = st.toggle("Debug Mode (Show AI Confidence)", value=False)
        st.divider()
        if st.button("♻️ Reset App Cache"):
            st.cache_resource.clear()
            st.rerun()
        st.divider()
        st.markdown("""
            ### Technical Stack
            - **YOLOv5**: Custom Detection
            - **ESRGAN**: 4x Super Resolution
            - **EasyOCR**: Text Extraction
        """)

    # Main Content
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose a vehicle image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image.convert("RGB"))
            # Convert RGB to BGR for OpenCV
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            st.image(image, caption="Original Input", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if uploaded_file is not None:
            sr_model, device = load_app_resources()
            
            # Use our recovery function
            # Updated to handle debug candidates
            with st.status("🔍 Analyzing Image Pipeline...", expanded=True) as status:
                st.write("Detecting license plates using YOLOv5...")
                plates = bpv.detect_license_plates_yolo(img_array)
                
                if not plates:
                    status.update(label="❌ No Plates Detected", state="error", expanded=False)
                    processed_img, enhanced_plate, texts, debug_data = None, None, None, None
                else:
                    st.write(f"✅ Found {len(plates)} plate candidates.")
                    st.write("Enhancing resolution and extracting text...")
                    
                    # Call process_license_plates with our new debug flag
                    processed_img, texts, debug_data = bpv.process_license_plates(
                        img_array.copy(), 
                        plates, 
                        sr_model=sr_model, 
                        device=device,
                        return_candidates=True
                    )
                    
                    # Crop the first detected plate for display
                    x, y, w, h = plates[0]
                    plate_roi = img_array[y:y+h, x:x+w]
                    enhanced_plate = esrgan_utils.upsample_esrgan(sr_model, plate_roi, device=device)
                    enhanced_plate = bpv.sharpen_image(enhanced_plate)
                    
                    status.update(label="✨ Optimization Complete!", state="complete", expanded=False)
            
            if processed_img is not None:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("🎯 Recovery Results")
                
                # Show Enhanced Plate ROI
                st.write("Enhanced License Plate View:")
                enhanced_rgb = cv2.cvtColor(enhanced_plate, cv2.COLOR_BGR2RGB)
                st.image(enhanced_rgb, use_container_width=True)
                
                # Show Detected Text
                if texts:
                    st.markdown(f'<div class="result-text">{texts[0]}</div>', unsafe_allow_html=True)
                    st.success(f"Recognized {len(texts)} plates in total.")
                else:
                    st.warning("Plate detected, but characters were illegible.")

                # Debug Mode: Show Candidates
                if debug_mode and debug_data:
                    st.divider()
                    st.subheader("🤖 AI Decision Analysis")
                    for p_idx, candidates in enumerate(debug_data):
                        st.write(f"**Plate {p_idx+1} Candidates:**")
                        # Show as a dataframe for neatness
                        import pandas as pd
                        df = pd.DataFrame(candidates)
                        if not df.empty:
                            df = df.sort_values(by='score', ascending=False)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.write("No candidates found in any stage.")
                
                # Download button for the full annotated image
                processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                res_img = Image.fromarray(processed_rgb)
                
                st.divider()
                st.markdown("Download Full Annotation:")
                st.download_button(
                    label="💾 Download Result",
                    data=cv2.imencode(".png", processed_img)[1].tobytes(),
                    file_name="plate_recovery_result.png",
                    mime="image/png"
                )
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Please upload an image to start the recovery pipeline.")

if __name__ == "__main__":
    main()
