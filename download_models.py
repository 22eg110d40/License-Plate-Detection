import os
import requests

def download_file(url, target_path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Downloaded to {target_path}")
    else:
        print(f"❌ Failed to download. Status code: {response.status_code}")

def setup_models():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # EDSR Model (x4 upscaling)
    edsr_url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
    edsr_path = os.path.join(model_dir, "EDSR_x4.pb")
    
    if not os.path.exists(edsr_path):
        download_file(edsr_url, edsr_path)
    else:
        print("✅ EDSR model already exists.")

    # Real-ESRGAN Model (x4 upscaling)
    esrgan_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    esrgan_path = os.path.join(model_dir, "RealESRGAN_x4plus.pth")
    
    if not os.path.exists(esrgan_path):
        download_file(esrgan_url, esrgan_path)
    else:
        print("✅ ESRGAN model already exists.")

if __name__ == "__main__":
    setup_models()
