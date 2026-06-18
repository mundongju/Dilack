import numpy as np
import matplotlib.pyplot as plt
import torch.fft as fft
import torch
import cv2
import glob
import time
import os
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
    
class ImageDataset(Dataset):
    """Custom image file dataset loader."""
    
    def __init__(self, data_path, label_path, transforms_image, transforms_label):
        # Initialize dataset paths and length
        self.img_paths = []
        self.label_paths = []
        self.transforms_image = transforms_image
        self.transforms_label = transforms_label
        for (img_path, label_path) in zip(sorted(glob.glob(data_path + '/*.tiff')), sorted(glob.glob(label_path + '/*'))):
            self.img_paths.append(img_path)
            self.label_paths.append(label_path)
        self.len = len(self.img_paths)

    def __getitem__(self, index):
        # Return image and label for a given index
        img = Image.open(self.img_paths[index])
        label = Image.open(self.label_paths[index])
        if img.format == 'TIFF':
            img = np.array(img)/255
        
        if self.transforms_image is not None:
            img = self.transforms_image(img)
        if self.transforms_label is not None:
            label = self.transforms_label(label)
                            
        return (img, label)

    def __len__(self):
        # Return dataset length
        return self.len

class CustomCenterCrop(object):
    def __init__(self, crop_size, center):
        self.crop_size = crop_size
        self.center = center

    def __call__(self, img):
        h, w = img.shape[-2:]  # Use img.shape for a PyTorch tensor
        new_h, new_w = self.crop_size
        center_x, center_y = self.center

        # Calculate the top-left corner of the crop
        top = int(center_x - new_w / 2)
        left = int(center_y - new_h / 2)

        # Perform the center crop
        img = transforms.functional.crop(img, top, left, new_h, new_w)

        return img

def hyperparameter_table(camera):
    """
    Define hyperparameters for a given camera.
    
    Parameters:
    - camera: Camera information
    
    Returns:
    - table: Dictionary of hyperparameters
    """
    if camera == 'v':
        table = {'mu1':[3.8009656577742135e-07],
                 'mu2':[7.55148676034878e-07],
                 'mu3':[6.681235049654788e-07],
                 'tau':[5.023222797717608e-07]}
        
    elif camera =='t':
        table = {'mu1':[1.6140469938363822e-07],
                 'mu2':[2.7243615363659046e-07],
                 'mu3':[1.8733291540229402e-07],
                 'tau':[5.492345280799782e-07]}
        
    elif camera =='SNU':
        table = {'mu1':[3.8009656577742135e-07],
                 'mu2':[7.55148676034878e-04],
                 'mu3':[6.681235049654788e-07],
                 'tau':[5.023222797717608e-07]}
    
    
    else:
        table = {'mu1':[3.8009656577742135e-07],
                 'mu2':[7.55148676034878e-06],
                 'mu3':[6.681235049654788e-07],
                 'tau':[5.023222797717608e-07]}
        
       
    return table

def to_tensor_or_numpy(data):
    """
    Convert input data to either a PyTorch Tensor or a Numpy array.

    Parameters:
    - data: Input data, either a PyTorch Tensor or a Numpy array

    Returns:
    - Converted data as a PyTorch Tensor or a Numpy array
    """
    # Check if input is a Tensor
    if torch.is_tensor(data):
        if data.is_cuda:
            data = data.cpu().detach()

        # Permute color channels if needed
        if len(data.shape) == 3:
            data = data.permute(1, 2, 0)  # Change color channel order (C, H, W) -> (H, W, C)
        elif len(data.shape) == 4:
            data = data.permute(0, 2, 3, 1)  # Change color channel order (B, C, H, W) -> (B, H, W, C)

        return data.numpy()

    # Check if input is a Numpy array
    elif isinstance(data, np.ndarray):
        # Permute color channels if needed
        data = torch.from_numpy(data)
        if len(data.shape) == 3:
            data = data.permute(2, 0, 1)  # Change color channel order (H, W, C) -> (C, H, W)
        elif len(data.shape) == 4:
            data = data.permute(0, 3, 1, 2)  # Change color channel order (B, H, W, C) -> (B, C, H, W)
        return data

    else:
        raise ValueError("Input must be a PyTorch Tensor or a Numpy array")


def psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Parameters:
    - img1: First input image
    - img2: Second input image
    
    Returns:
    - PSNR value
    """
    if torch.is_tensor(img1):
        mse = torch.mean((img1 - img2) ** 2).detach().cpu().numpy()
    else:
        mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-9:
        return "Same Image"
    return 10 * np.log10(1. / mse)

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

def clamp(img, percentile_lower=0, percentile_upper=99.9):
    """
    Clamp image values within the specified percentile range.
    
    Parameters:
    - img: Input torch tensor
    - percentile_lower: Lower percentile for clamping
    - percentile_upper: Upper percentile for clamping
    
    Returns:
    - clipped_img: Clamped torch tensor
    """
    lower_percentile = percentile_lower / 100.0
    upper_percentile = percentile_upper / 100.0

    img_shape = img.size()
    if len(img_shape)==2:
        channel = 1
        height = img_shape[0]
        width = img_shape[1]
    else:
        channel = img_shape[-3]
        height = img_shape[-2]
        width = img_shape[-1]
    
    reshaped_img = img.reshape(channel * height * width)
    sorted_values, _ = torch.sort(reshaped_img, dim=0)

    lower_index = int(lower_percentile * (channel * height * width))
    upper_index = int(upper_percentile * (channel * height * width))

    lower_values = sorted_values[lower_index]
    upper_values = sorted_values[upper_index]

    lower_values = lower_values.reshape(1, 1, 1)
    upper_values = upper_values.reshape(1, 1, 1)

    clipped_img = torch.clamp(img, lower_values, upper_values)

    return clipped_img

def call(img_path, crop=False, center=(1824-120, 2736-140), size=(2600, 2600), figure=False, is_color=False):
    """
    Read image, optionally crop, and return the image.
    
    Parameters:
    - img_path: Path to the input image
    - crop: Whether to perform cropping
    - crop_center: Center coordinates for cropping
    - crop_size: Size of the cropped region
    - figure: Whether to display the image
    - is_color: Whether to use as 3 channel color image
    
    Returns:
    - img: Processed image
    """
    if is_color:
        img = cv2.imread(img_path, -1)
    else:
        img = cv2.imread(img_path, 0)
        img = np.expand_dims(img, axis=-1)
    img = np.array(img, dtype="float32")
    img_size = np.shape(img)
    if figure:
        plt.imshow(img, cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.show()
    if crop:
        img = center_crop(img, center=(center[0], center[1]), size=(size[0], size[1]), mode='crop')
    return img

def get_dict(img_path, camera=None):
    """
    Create a dictionary mapping image names to file paths.
    
    Parameters:
    - camera: Camera information
    - img_path: Path to the image directory
    
    Returns:
    - img_dict: Dictionary mapping image names to file paths
    """
    img_dict = {}
    desired_extensions = ['tiff', 'png']
    img_path = os.path.join(img_path, '')

    for ext in desired_extensions:
        for path in glob.glob(os.path.join(img_path, f'**/*.{ext}'), recursive=True):
            relative_path = os.path.relpath(path, img_path)
            _, img_name = os.path.split(relative_path)
            key = os.path.splitext(img_name)[0]
            if path.find('archieve') != -1 or path.find('result') != -1:
                continue
            if key.find('psf') != -1:
                img_dict['psf'] = path
            else:
                img_dict[key] = path

    return img_dict

def linalg_norm(img):
    if torch.is_tensor(img):
        return minmax_norm_t(img)
    else:
        return minmax_norm_n(img)
    
def minmax_norm(img):
    if torch.is_tensor(img):
        return minmax_norm_t(img)
    else:
        return minmax_norm_n(img)
    
def max_norm(img):
    if torch.is_tensor(img):
        return max_norm_t(img)
    else:
        return max_norm_n(img)

def linalg_norm_t(img):
    """
    Normalize a PyTorch tensor using the L2 norm.

    Parameters:
    - img: Input PyTorch tensor

    Returns:
    - Normalized PyTorch tensor
    """
    return img / torch.linalg.norm(img.contiguous().view(-1))

def minmax_norm_t(img):
    """
    Min-max normalize the tensor to the float range.
    
    Parameters:
    - img: Input image
    
    Returns:
    - Normalized image
    """
    return (img - img.min()) / (img.max() - img.min())

def max_norm_t(img):
    """
    Normalize the tensor to the float range based on the maximum value.
    
    Parameters:
    - img: Input image
    
    Returns:
    - Normalized image
    """
    return img / img.max()

def linalg_norm_n(img):
    """
    Normalize a Numpy array using the L2 norm.

    Parameters:
    - img: Input Numpy array

    Returns:
    - Normalized Numpy array
    """
    return (img / np.linalg.norm(img.reshape(-1)) * 255).astype(np.uint8)

def minmax_norm_n(img):
    """
    Min-max normalize the image to the 8-bit range.
    
    Parameters:
    - img: Input image
    
    Returns:
    - Normalized image
    """
    return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)

def max_norm_n(img):
    """
    Normalize the image to the 8-bit range based on the maximum value.
    
    Parameters:
    - img: Input image
    
    Returns:
    - Normalized image
    """
    return ((img / np.max(img)) * 255).astype(np.uint8)

def deconvolve_rgb(x,psf,alpha=1e5):
    '''
    wiener filtering for monochome raw image
    x     : raw image
    psf   : psf
    alpha : reconstruction hyperparameter (high alpha means more noise)
    '''
    psf = to_tensor_or_numpy(psf)
    x = to_tensor_or_numpy(x)
    psf_ft = torch.fft.fftn(psf, dim=(-2,-1))
    psf_ft = torch.conj(psf_ft) / (abs(psf_ft)**2 + alpha)    
    x_ft = torch.fft.fftn(x, dim=(-2, -1))
    # Wiener deconvolution Operation
    recon = torch.fft.fftshift(torch.fft.ifftn(x_ft * psf_ft, dim=(-2, -1)), dim=(-2, -1))
    return torch.real(recon)


def plot_ADMM(psf, raw, result, times, hyperparameters, clamp_=0.1, total_time=0, size=1200, crop=None, norm=None, iteration=None, is_color=False):
    """
    Display PSF, raw image, and result side by side with relevant information.
    
    Parameters:
    - psf: Point Spread Function
    - raw: Raw input image
    - result: Output result image
    - times: Time taken for the operation
    - hyperparameters: Dictionary of hyperparameters
    - clamp_: Clamping parameter
    - total_time: Total time taken
    - size: Crop size
    - crop: Crop factor
    - norm: Normalization method
    - iteration: Iteration number (optional)
    - is_color: Wheter image is RGB
    """
    display_time = time.time()
    
    # Convert torch tensors to NumPy arrays for plotting
    if torch.is_tensor(psf):
        psf = to_tensor_or_numpy(psf)
    if torch.is_tensor(raw):
        raw = to_tensor_or_numpy(raw)
    if torch.is_tensor(result):
        # Clamp result tensor within the specified percentile range
        result = clamp(result, percentile_upper=99. + (1. - clamp_))
        result = to_tensor_or_numpy(result)
    
    # Apply cropping if specified
    if crop:
        result = center_crop(result, center=None, size=(size, size), mode='crop')

    # Normalize images based on the specified normalization method
    if norm == 'max':
        psf = max_norm(psf)
        raw = max_norm(raw)
        result = max_norm(result)
        psf[psf<0] = 0
        raw[raw<0] = 0
        result[result<0] = 0
    elif norm == 'linalg':
        psf = linalg_norm(psf)
        raw = linalg_norm(raw)
        result = linalg_norm(result)        
    else:
        psf = minmax_norm(psf)
        raw = minmax_norm(raw)
        result = minmax_norm(result)
    
    # Plot PSF, raw, and result images side by side
    fig, ax = plt.subplots(1, 3)
    fig.set_figheight(8)
    fig.set_figwidth(30)
    
    if is_color:
        ax[0].imshow(cv2.cvtColor(psf, cv2.COLOR_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        ax[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    else:
        ax[0].imshow(psf, cmap='gray')
        ax[1].imshow(raw, cmap='gray')
        ax[2].imshow(result, cmap='gray')
        

    ax[0].set_title("PSF", fontsize=20)
    ax[1].set_title("RAW", fontsize=20)
    ax[2].set_title("OUTPUT", fontsize=20)

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    
    # Display relevant information as title
    if total_time and iteration is not None:
        fig.suptitle('{} Iteration Elapsed: {:.2} ms (total {:3.3} ms), \n $\\mu_1$:{:.2e}, $\\mu_2$:{:.2e}, $\\mu_3$:{:.2e}, $\\tau$:{:.2e}'.format(
            iteration,
            times * 1e3,
            total_time * 1e3,
            hyperparameters['mu1'][0],
            hyperparameters['mu2'][0],
            hyperparameters['mu3'][0],
            hyperparameters['tau'][0]), fontsize=25)
    else:
        fig.suptitle(
            'Total Elapsed (+ displaying): {:3.3} s, \n $\\mu_1$:{:.2e}, $\\mu_2$:{:.2e}, $\\mu_3$:{:.2e}, $\\tau$:{:.2e}'.format(
                times + time.time() - display_time,
                hyperparameters['mu1'][0],
                hyperparameters['mu2'][0],
                hyperparameters['mu3'][0],
                hyperparameters['tau'][0]), fontsize=25)
    
    fig.tight_layout()

    plt.show()



def plot_training_results(outputs_l, labels_l, mu1s, mu2s, mu3s, taus):
    """
    Plot training results including loss, PSNR, and hyperparameters.

    Parameters:
    - outputs_l: Model outputs after cropping.
    - labels_l: Ground truth labels after cropping.
    - mu1s, mu2s, mu3s, taus: Lists containing hyperparameter values over epochs.
    """
    
    # If the shape of the outputs is 4D, squeeze it to 2D
    if len(outputs_l.shape) == 4:
        outputs_l = outputs_l[0, 0, :, :]
        labels_l = labels_l[0, 0, :, :]
        
    # Plot ground truth and model outputs side by side
    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    plt.imshow(labels_l.cpu().numpy(), cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(outputs_l.to('cpu').detach().numpy(), cmap='gray')
    plt.show()

    # Plot hyperparameter values over epochs
    plt.figure(figsize=(20, 20))
    plt.subplot(4, 4, 1)
    plt.plot(mu1s, color='red', label='mu1')
    plt.legend()
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Parameters", fontsize=15)
    plt.subplot(4, 4, 2)
    plt.plot(mu2s, color='blue', label='mu2')
    plt.legend()
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Parameters", fontsize=15)
    plt.subplot(4, 4, 3)
    plt.plot(mu3s, color='green', label='mu3')
    plt.legend()
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Parameters", fontsize=15)
    plt.subplot(4, 4, 4)
    plt.plot(taus, color='black', label='tau')
    plt.legend()
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Parameters", fontsize=15)
    plt.show()

