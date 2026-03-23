import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = RRDB

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RRDB_block_f(nf=nf, gc=gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # upsampling
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        res = self.body(fea)
        res = self.conv_body(res)
        fea = fea + res

        fea = self.lrelu(self.conv_up1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.conv_up2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))

        return out

def load_esrgan_model(model_path, device='cpu'):
    """
    Loads ESRGAN model weights into RRDBNet architecture.
    Default parameters nb=23, nf=64 are for Real-ESRGAN x4 plus.
    """
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=torch.device(device))
    
    # Real-ESRGAN weights often have a 'params' or 'params_ema' key
    if 'params' in state_dict:
        state_dict = state_dict['params']
    elif 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
        
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)
    return model

def upsample_esrgan(model, img, device='cpu'):
    """
    Upsample an image using ESRGAN model.
    Input: OpenCV BGR image
    Output: OpenCV BGR image (x4 resolution)
    """
    # Preprocessing
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_lr = img.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_lr).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    # Postprocessing
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    
    return output

if __name__ == "__main__":
    # Quick test if script is run directly
    model_path = os.path.join("models", "RealESRGAN_x4plus.pth")
    if os.path.exists(model_path):
        print("Loading ESRGAN model...")
        model = load_esrgan_model(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Model not found at {model_path}. Run download_models.py first.")
