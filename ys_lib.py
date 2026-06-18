
import torch
import torch.fft as fft
import torch.nn.functional as F
import ADMM_Torch_color
from utils import *
from torch.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt
import numpy as np
import math
import glob
import cv2 as cv
from scipy.ndimage import rotate
from skimage.metrics import structural_similarity as ssim
from scipy import signal
import time


def conv_psf_fft(x, y):
    # Calculate the necessary padding sizes
    pad_left = pad_right = y.size(3) // 2
    pad_top = pad_bottom = y.size(2) // 2
    
    # Pad the single channel tensor with zeros (constant padding)
    x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    y_padded = F.pad(y, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    # Perform the convolution
    output = conv_fft2(x_padded, y_padded)

    # Adjust output_channel size if necessary
    # Assuming output_channel needs to be cropped to match the original size
    output_cropped = center_crop(output, size=(pad_right*2, pad_bottom*2))
    output_cropped = output_cropped / torch.max(output_cropped)
    output = output / torch.max(output)
    return output_cropped.squeeze(0), output.squeeze(0)

def pad_half(x):
    pad_left = pad_right = x.size(3) // 2
    pad_top = pad_bottom = x.size(2) // 2
    
    # Pad the single channel tensor with zeros (constant padding)
    x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return x_padded


def conv_fft2(x,y):
    # return torch.abs((ifft2(fft2(x)*fft2(y))))
    return torch.abs(ifftshift(ifft2(fftshift(fft2(x),dim=(-2,-1))*fftshift(fft2(y),dim=(-2,-1))),dim=(-2,-1)))


def center_crop(img, center=None, size=(0, 0), mode="crop", **kwargs):
    """
    Crop the input image based on the center and size.
    
    Parameters:
    - img: Input image (numpy array or torch tensor)
    - center: Center coordinates for cropping
    - size: Size of the cropped region
    - mode: Cropping mode ("crop", "same", "center")
    
    Returns:
    - output: Cropped image
    """
    if torch.is_tensor(img):
        return center_crop_t(img, center=center, size=size, mode=mode, **kwargs)
    else:
        return center_crop_n(img, center=center, size=size, mode=mode, **kwargs)

    
def center_crop_n(img, center=None, size=(0, 0), mode="crop"):
    """
    Crop numpy array based on the center and size.
    
    Parameters:
    - img: Input numpy array
    - center: Center coordinates for cropping
    - size: Size of the cropped region
    - mode: Cropping mode ("crop", "same", "center")
    
    Returns:
    - output: Cropped numpy array
    """
    img_h, img_w = np.shape(img)[:2]
    crop_h, crop_w = size
    crop_h_half, crop_h_mod = divmod(crop_h, 2)
    crop_w_half, crop_w_mod = divmod(crop_w, 2)

    if center is None:
        crop_center_h = img_h // 2
        crop_center_w = img_w // 2
    else:
        crop_center_h, crop_center_w = center

    if mode == "crop":
        output = img[crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
                     crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod, ...]
    elif mode == "same":
        output = np.zeros_like(img)
        output[crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
               crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod, ...] = img[
            crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
            crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod, ...
        ]
    elif mode == "center":
        output = np.zeros_like(img)
        output[img_h - crop_h_half: img_h + crop_h_half + crop_h_mod,
               img_w - crop_w_half: img_w + crop_w_half + crop_w_mod, ...] = img[
            crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
            crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod, ...
        ]
    return output

def center_crop_t(img, center=None, size=(0, 0), mode="crop"):
    """
    Crop torch tensor based on the center and size.
    
    Parameters:
    - img: Input torch tensor
    - center: Center coordinates for cropping
    - size: Size of the cropped region
    - mode: Cropping mode ("crop", "same", "center")
    
    Returns:
    - output: Cropped torch tensor
    """
    img_h, img_w = img.size()[-2:]
    crop_h, crop_w = size
    crop_h_half, crop_h_mod = divmod(crop_h, 2)
    crop_w_half, crop_w_mod = divmod(crop_w, 2)

    if center is None:
        crop_center_h = img_h // 2
        crop_center_w = img_w // 2
    else:
        crop_center_h, crop_center_w = center

    if mode == "crop":
        output = img[..., crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
                    crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod]
    elif mode == "same":
        output = torch.zeros_like(img)
        output[..., crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
               crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod] = img[
            ..., crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
            crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod
        ]
    elif mode == "center":
        output = torch.zeros_like(img)
        output[..., img_h - crop_h_half: img_h + crop_h_half + crop_h_mod,
               img_w - crop_w_half: img_w + crop_w_half + crop_w_mod] = img[
            ..., crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
            crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod
        ]
    return output
    
    
    
def alt_axis(X,mode):
    if mode ==1:
        return torch.einsum('ijk->jki', X)   
    elif mode ==2:
        return torch.einsum('ijk->kij', X)   
    
def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d



def conv_psf_fft_down(x, y, down_size):
    # Calculate the necessary padding sizes
    pad_left = pad_right = y.size(3) // 2
    pad_top = pad_bottom = y.size(2) // 2
    
    # Pad the single channel tensor with zeros (constant padding)
    # x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    # y_padded = F.pad(y, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    # Perform the convolution
    raw_fft = (fftshift(fft2(x,dim=(-2,-1)),dim=(-2,-1))*fftshift(fft2(y,dim=(-2,-1)),dim=(-2,-1))) # origin 
    #raw_fft = fftshift(fft2(x, dim=(-2, -1)), dim=(-2, -1)) # no kernel
    raw_fft = center_crop(raw_fft, size=(down_size, down_size))
    output = torch.abs(ifftshift(ifft2(raw_fft,dim=(-2,-1)),dim=(-2,-1)))
    output = output / torch.max(output)
    return output.squeeze(0)