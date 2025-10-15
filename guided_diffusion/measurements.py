'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m

import cv2
import numpy as np
from torch.fft import fft2, ifft2, fftshift, ifftshift
from guided_diffusion.custom_util import *

import argparse

import ys_lib
import torchvision as tv


def do_erase(data,ratio):
    
    origin_shape = data.shape
    
    flattened = data.flatten()

    num_zeroes = int(len(flattened) *ratio)

    indices = np.random.choice(len(flattened), num_zeroes, replace=False)

    flattened[indices] = 0
    
    data = flattened.reshape(origin_shape)
    
    return data


def sharpen_image(image, strength=1.0):
    """
    Apply sharpening to an image tensor using unsharp masking.
    :param image: Image tensor of shape [batch_size, channels, height, width].
    :param strength: Strength of the sharpening effect.
    """
    if image.dim() != 4:
        raise ValueError("Image tensor must be 4-dimensional.")

    # Create a blur kernel for unsharp masking
    blur_kernel = torch.tensor([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=torch.float32).to(image.device) / 16.0
    blur_kernel = blur_kernel.reshape(1, 1, 3, 3).repeat(image.size(1), 1, 1, 1)

    # Apply padding to maintain the image size
    padded_image = F.pad(image, (1, 1, 1, 1), mode='reflect')

    # Blur the image
    blurred_image = F.conv2d(padded_image, blur_kernel, groups=image.size(1))

    # Create the unsharp mask
    unsharp_mask = image - blurred_image

    # Create the sharpened image by adding the unsharp mask to the original image
    sharpened_image = image + strength * unsharp_mask

    return sharpened_image   

# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self,args, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        #return self.down_sample(data)

        resize_raw = tv.transforms.Resize(256, interpolation=3)
        return resize_raw(self.down_sample(data))

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)
    

def cubic(x):
    # See Keys, "Cubic Convolution Interpolation for Digital Image
    # Processing, " IEEE Transactions on Acoustics, Speech, and Signal Processing, Vol.ASSP - 29, No. 6, December 1981, p.1155.
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    f = (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2))
    return f
    
def prepare_cubic_filter(scale):
    # uses the kernel part of matlab's imresize function (before subsampling)
    # note: scale<1 for downsampling

    kernel_width = 4
    kernel_width = kernel_width / scale

    u = 0

    # What is the left-most pixel that can be involved in the computation?
    left = np.floor(u - kernel_width/2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = np.ceil(kernel_width) + 1

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left + np.arange(0,P,1) # = left + [0:1:P-1]

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    weights = scale * cubic(scale *(u-indices))
    weights = np.reshape(weights, [1,weights.size])
    return np.matmul(weights.T,weights)

    

@register_operator(name='super_resolution_fft')
class SuperResolutionOperatorFFT(LinearOperator):
    def __init__(self,args, in_shape, scale_factor, device, eta_reg=1e-4):

        factor = int(args.deg_scale)
        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0
            
         # Device setting
        device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_str)  


        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        #kernel = torch.from_numpy(k).float().to(device)
        k = prepare_cubic_filter(1/factor)

        # k (17,17)
        k = k[:-1,:-1]
       

        self.device = device
        kernel = torch.from_numpy(k).float().to(self.device)        
        self.scale_factor = scale_factor
        self.kernel = kernel.to(device)
        self.in_shape = in_shape
        self.eta_reg = eta_reg

        # Prepare the kernel for FFT operations
        self.kernel_size = kernel.shape
        self.img_dim = in_shape[-1]
        self.small_dim = self.img_dim // scale_factor
        self._prepare_fft_kernel()

    def _prepare_fft_kernel(self):
        # Zero-pad kernel and center it for FFT
        mk, nk = self.kernel_size
        big_kernel = torch.zeros((self.img_dim, self.img_dim)).to(self.device)
        big_kernel[:mk, :nk] = self.kernel
        big_kernel = torch.roll(big_kernel, (-int((mk - 1) / 2), -int((nk - 1) / 2)), dims=(0, 1))
        self.fft_kernel = torch.fft.fft2(big_kernel)
        self.inv_fft_kernel = 1 / (torch.abs(self.fft_kernel)**2 + self.eta_reg)

        self.big_kernel = big_kernel

    def forward(self, data, **kwargs):
        # Apply down-sampling after convolution
        reshaped_data = data.reshape(-1, *self.in_shape)[0]
        convolved = self._apply_fft_conv(reshaped_data, flag_invertB=False)
        return self._downsample(convolved).reshape(data.shape[0], -1)

    def transpose(self, data, **kwargs):
        # Apply up-sampling followed by transposed convolution
        reshaped_data = data.reshape(-1, *self.in_shape[:2], self.small_dim, self.small_dim)
        upsampled = self._upsample(reshaped_data)
        return self._apply_fft_conv(upsampled, flag_invertB=True).reshape(data.shape[0], -1)

    def project(self, data, measurement, **kwargs):
        # Project onto the solution space
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

    def _apply_fft_conv(self, data, flag_invertB):
        # FFT-based convolution
        batch_size, channels, height, width = data.shape
        output = torch.zeros_like(data)
        kernel = self.fft_kernel if not flag_invertB else torch.conj(self.fft_kernel)
        for ch in range(channels):
            fft_data = torch.fft.fft2(data[:, ch, :, :])
            fft_result = fft_data * kernel
            output[:, ch, :, :] = torch.real(torch.fft.ifft2(fft_result))
        return output

    def _downsample(self, data):
        # Down-sample data by scale_factor
        return data[:, :, ::self.scale_factor, ::self.scale_factor]

    def _upsample(self, data):
        # Up-sample data to original resolution
        batch_size, channels, small_h, small_w = data.shape
        output = torch.zeros(batch_size, channels, self.img_dim, self.img_dim).to(data.device)
        for b in range(batch_size):
            for c in range(channels):
                output[b, c, ::self.scale_factor, ::self.scale_factor] = data[b, c, :, :]
        return output    


@register_operator(name='sr_fft_ys')
class SRFFT(LinearOperator):
    def __init__(self,args, in_shape, scale_factor, device, eta_reg=1e-4):

        self.deg_scale = args.deg_scale
        self.gt_size= 256
        std= 1
        #self.down_size = 64
        self.down_size = int(self.gt_size/args.deg_scale)
        psf = torch.tensor(ys_lib.gkern(self.gt_size,std),dtype=torch.float32).to(device)
        self.psf = psf.unsqueeze(0).unsqueeze(0)
        #psf_crop = ys_lib.center_crop(psf, size=(down_size, down_size))
    
    def forward(self, data, **kwargs):
        # Apply down-sampling after convolution
        #data = data/torch.max(data)
        print ("data",data.shape,",psf", self.psf.shape)
        raw_simulated=ys_lib.conv_psf_fft_down(data,self.psf, self.down_size) 
        print ('raw_sim:',raw_simulated.shape)
        resize_raw = tv.transforms.Resize(self.gt_size, interpolation=3)

        raw_simulated_resize = resize_raw(raw_simulated)

        #raw_simulated_resize = raw_simulated_resize *2 -1 #(0~1) =>(-1~1)

        return raw_simulated_resize.unsqueeze(0) #[3,256,256] 
        #return raw_simulated

    def transpose(self, data, **kwargs):
        return data

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)
    

@register_operator(name='denoising')
class SRFFT(LinearOperator):
    def __init__(self,args, in_shape, scale_factor, device, eta_reg=1e-4):

        data = 0

    def forward(self, data, **kwargs):
    
        return data

    def transpose(self, data, **kwargs):
        return data

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)



    
@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self,args, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?
        
        """

        while True:
            self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)        
            if np.sum(self.kernel.kernelMatrix) >150000:
                self.kernel.kernelMatrix /=  np.sum(self.kernel.kernelMatrix)
                break
        """

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        self.kernel.kernelMatrix /=  np.sum(self.kernel.kernelMatrix)

        #breakpoint()
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        
        if kernel_size < 256:
            pad_size = int((256-kernel_size)/2)
            kernel = F.pad(kernel,(pad_size,pad_size,pad_size,pad_size))
    
        #self.conv.update_weights(kernel)

        self.new_kernel = kernel.unsqueeze(0).unsqueeze(0).to(device)
        #self.new_kernel = sharpen_image(self.new_kernel, 25.0)
        #self.new_kernel[self.new_kernel<0.0001] = 0
        
        print ("motion blur range :",self.new_kernel.max(),self.new_kernel.min())

    def forward(self, data, **kwargs):
        # A^T * A
        #return self.conv(data)
        pad_size = 128
        
        data_pad = F.pad(data,[pad_size,pad_size,pad_size,pad_size], mode='circular')#replicate,reflect
        kernel_pad = F.pad(self.new_kernel,[pad_size,pad_size,pad_size,pad_size])

        #return crop_and_noise_2(conv_psf_fft2(data_pad,kernel_pad),250,0)
        return conv_psf_fft2(data_pad,kernel_pad)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = torch.from_numpy(self.kernel.kernelMatrix).type(torch.float32)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    
    def __init__(self,args, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
        self.new_kernel = self.conv.get_kernel().unsqueeze(0).unsqueeze(0).to(device).type(torch.float32)

    def forward(self, data, **kwargs):
        pad_size = 128
        data_pad = F.pad(data,[pad_size,pad_size,pad_size,pad_size],mode='replicate')#replicate,reflect
        kernel_pad = F.pad(self.new_kernel,[pad_size,pad_size,pad_size,pad_size])

        #return crop_and_noise_2(conv_psf_fft2(data_pad,kernel_pad),250,0)
        return conv_psf_fft2(data_pad,kernel_pad)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma
    

@register_noise(name='sr_gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma
    

@register_noise(name='dps_gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma
    
    
@register_noise(name='gaussian_mean')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        """
        Initializes the GaussianNoise layer.
        
        Parameters:
        - sigma (float): Standard deviation of the Gaussian noise.
        - mu (float): Mean of the Gaussian noise. Default is 0.0.
        """
        self.sigma = sigma
    
    def forward(self, data):
        """
        Applies Gaussian noise to the input data.

        Parameters:
        - data (torch.Tensor): Input tensor to which noise will be added.

        Returns:
        - torch.Tensor: Noisy data.
        """
        noise = torch.randn_like(data, device=data.device) * self.sigma + torch.mean(data)
        return data + noise


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)


    

@register_operator(name='lensless_custom')        
class PsfOperator(LinearOperator):
    def __init__(self,args, device, **kwargs) -> None:
        
                
        self.args = args
        GPU_NUM =self.args.gpu
        self.device = device
    

        print("GPU: ", GPU_NUM)

        #self.psf = load_psf("/data3/lensless_data/diffuser_v1/psf.tiff",256)[:,0:1,:,:].cuda()
        if args.voronoi_new:
            self.psf_512 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v4/psf_voronoi.png", resize_dim= 512,arg_voronoi=0).unsqueeze(0).cuda(GPU_NUM)
            self.psf_256 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v4/psf_voronoi.png", resize_dim= 256,arg_voronoi=0).unsqueeze(0).cuda(GPU_NUM)
        else:
            self.psf_512 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v1/psf/psf_camera1_original.tiff", resize_dim= 512).unsqueeze(0).cuda(GPU_NUM)
            self.psf_256 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v1/psf/psf_camera1_original.tiff", resize_dim= 256).unsqueeze(0).cuda(GPU_NUM)

        
        if args.resize_psf:
            resized_psf = F.interpolate(self.psf_512,(args.resize_psf,args.resize_psf))
            pad_size = int((512-args.resize_psf)/2)
            self.psf_512 = F.pad(resized_psf,(pad_size,pad_size,pad_size,pad_size))
 
        
        
        self.new_kernel = self.psf_512 
    
    def forward(self, data, **kwargs):
        pad_size = 128
        data_pad = F.pad(data,(pad_size,pad_size,pad_size,pad_size), mode='constant', value=-1) # # [1,3,256,256] =>[1,3,512,512]

        Ax = self.apply_kernel(data_pad,kernel_size=512)
        if self.args.crop:
            crop_Ax = crop_and_noise_2(Ax,self.args.crop,0)
        else:
            crop_Ax = Ax
        return crop_Ax
        
    def forward_nopad(self, data, **kwargs):
        return self.apply_kernel(data,kwargs.get('kernel_size'))

    def transpose(self, data, **kwargs):
        return data
    
    def apply_kernel(self, data, kernel_size):
        #TODO: faster way to apply conv?:W

        if kernel_size == 256:
            kernel = self.psf_256
        else:
            kernel = self.psf_512


        output = conv_psf_fft(data,kernel)

        return output

@register_operator(name='lensless')
class PsfOperator(LinearOperator):
    def __init__(self,args, device, **kwargs) -> None:
        
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_config', type=str)
        parser.add_argument('--diffusion_config', type=str)
        parser.add_argument('--task_config', type=str)
        parser.add_argument('--gpu', type=int, default=4)
        parser.add_argument('--save_dir', type=str, default='./results')
        parser.add_argument('--erase_ratio', type=float, default=0.9999)
        args = parser.parse_args()
        """
        
        self.args = args
        GPU_NUM =self.args.gpu
        self.device = device

        print("GPU: ", GPU_NUM)
        
        #breakpoint()
        
        if args.voronoi_new:
            self.psf_512 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v4/psf_voronoi.png", resize_dim= 512,arg_voronoi=0).unsqueeze(0).cuda(GPU_NUM)
            self.psf_256 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v4/psf_voronoi.png", resize_dim= 256,arg_voronoi=0).unsqueeze(0).cuda(GPU_NUM)
        else:
            self.psf_512 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v1/psf/psf_camera1_original.tiff", resize_dim= 512).unsqueeze(0).cuda(GPU_NUM)
            self.psf_256 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v1/psf/psf_camera1_original.tiff", resize_dim= 256).unsqueeze(0).cuda(GPU_NUM)


        if args.resize_psf:
            resized_psf = F.interpolate(self.psf_512,(args.resize_psf,args.resize_psf))
            pad_size = int((512-args.resize_psf)/2)
            self.psf_512 = F.pad(resized_psf,(pad_size,pad_size,pad_size,pad_size))

        
                
        # erase
        if args.erase_ratio:
            psf = do_erase(self.psf_512[0].cpu(),args.erase_ratio)
            self.psf_512 = psf.unsqueeze(0).to(device)

        
        self.new_kernel = self.psf_512 
    
    def forward(self, data, **kwargs):
        pad_size = 128
        data_pad = F.pad(data,(pad_size,pad_size,pad_size,pad_size), mode='constant', value=-1) # # [1,3,256,256] =>[1,3,512,512]

        Ax = self.apply_kernel(data_pad,kernel_size=512)
        crop_Ax = crop_and_noise_2(Ax,self.args.crop,0)
        return crop_Ax
        
    def forward_nopad(self, data, **kwargs):
        return self.apply_kernel(data,kwargs.get('kernel_size'))

    def transpose(self, data, **kwargs):
        return data
    
    def apply_kernel(self, data, kernel_size):
        #TODO: faster way to apply conv?:W

        if kernel_size == 256:
            kernel = self.psf_256
        else:
            kernel = self.psf_512


        output = conv_psf_fft(data,kernel)

        return output

@register_operator(name='lensless_turing')
class PsfOperator(LinearOperator):
    def __init__(self,args, device, **kwargs) -> None:
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_config', type=str)
        parser.add_argument('--diffusion_config', type=str)
        parser.add_argument('--task_config', type=str)
        parser.add_argument('--gpu', type=int, default=4)
        parser.add_argument('--save_dir', type=str, default='./results')
        args = parser.parse_args()
        """
        self.args = args
        GPU_NUM =self.args.gpu
        self.device = device

        print("GPU: ", GPU_NUM)

        self.psf_512 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v2_dj_turing/psf/psf_turing.png", resize_dim= 512,arg_voronoi=0).unsqueeze(0).cuda(GPU_NUM)
        self.psf_256 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v2_dj_turing/psf/psf_turing.png", resize_dim= 256,arg_voronoi=0).unsqueeze(0).cuda(GPU_NUM)


        if args.resize_psf:
            resized_psf = F.interpolate(self.psf_512,(args.resize_psf,args.resize_psf))
            pad_size = int((512-args.resize_psf)/2)
            self.psf_512 = F.pad(resized_psf,(pad_size,pad_size,pad_size,pad_size))

        
        self.new_kernel = self.psf_512 
    
    def forward(self, data, **kwargs):
        pad_size = 128
        data_pad = F.pad(data,(pad_size,pad_size,pad_size,pad_size), mode='constant', value=-1) # # [1,3,256,256] =>[1,3,512,512]

        Ax = self.apply_kernel(data_pad,kernel_size=512)
        crop_Ax = crop_and_noise_2(Ax,self.args.crop,0)
        return crop_Ax
        
    def forward_nopad(self, data, **kwargs):
        return self.apply_kernel(data,kwargs.get('kernel_size'))

    def transpose(self, data, **kwargs):
        return data
    
    def apply_kernel(self, data, kernel_size):
        #TODO: faster way to apply conv?:W

        if kernel_size == 256:
            kernel = self.psf_256
        else:
            kernel = self.psf_512


        output = conv_psf_fft(data,kernel)

        return output


### Real : YS voronoi Dataset ###
@register_operator(name='lensless_real_voronoi')
class PsfOperator(LinearOperator):
    def __init__(self,args, device, **kwargs) -> None:
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_config', type=str)
        parser.add_argument('--diffusion_config', type=str)
        parser.add_argument('--task_config', type=str)
        parser.add_argument('--gpu', type=int, default=4)
        parser.add_argument('--save_dir', type=str, default='./results')
        args = parser.parse_args()
        """
        self.args = args
        GPU_NUM =self.args.gpu
        self.device = device

        print("GPU: ", GPU_NUM)

        self.psf_512 = load_psf_real(args = self.args, psf_file = "./samples/dps_val/ys_flickr/psf/psf_camera1_original.tiff", resize_dim= 512).unsqueeze(0).cuda(GPU_NUM)
        self.psf_256 = load_psf(args = self.args, psf_file = "./samples/dps_val/ys_flickr/psf/psf_camera1_original.tiff", resize_dim= 256).unsqueeze(0).cuda(GPU_NUM)

        
        self.new_kernel = self.psf_512 
    
    def forward(self, data, **kwargs):
        pad_size = 128
        data_pad = F.pad(data,(pad_size,pad_size,pad_size,pad_size), mode='constant', value=-1) # # [1,3,256,256] =>[1,3,512,512]

        Ax = self.apply_kernel(data_pad,kernel_size=512)
        # crop 4%
        crop_Ax = crop_and_noise_2(Ax,self.args.crop,0)
        # vignetting
        crop_Ax = vignetting(crop_Ax,175)
        return crop_Ax
        
    def forward_nopad(self, data, **kwargs):
        return self.apply_kernel(data,kwargs.get('kernel_size'))

    def transpose(self, data, **kwargs):
        return data
    
    def apply_kernel(self, data, kernel_size):
        #TODO: faster way to apply conv?:W

        if kernel_size == 256:
            kernel = self.psf_256
        else:
            kernel = self.psf_512


        output = conv_psf_fft(data,kernel)

        return output

#forward : vignette filter
@register_operator(name='lensless_vignett')
class PsfOperator(LinearOperator):
    def __init__(self,args, device, **kwargs) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_config', type=str)
        parser.add_argument('--diffusion_config', type=str)
        parser.add_argument('--task_config', type=str)
        parser.add_argument('--gpu', type=int, default=4)
        parser.add_argument('--save_dir', type=str, default='./results')
        args = parser.parse_args()
        self.args = args
        GPU_NUM =self.args.gpu
        self.device = device

        print("GPU: ", GPU_NUM)

        self.psf_512 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v1/psf/psf_camera1_original.tiff", resize_dim= 512).unsqueeze(0).cuda(GPU_NUM)
        self.psf_256 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v1/psf/psf_camera1_original.tiff", resize_dim= 256).unsqueeze(0).cuda(GPU_NUM)
    
    def forward(self, data, **kwargs):
        return self.apply_kernel(data,kwargs.get('kernel_size'))

    def transpose(self, data, **kwargs):
        return data

    # vignett
    def apply_kernel(self, data, kernel_size):
        #TODO: faster way to apply conv?:W

        if kernel_size == 256:
            kernel = self.psf_256
        else:
            kernel = self.psf_512

        output = conv_psf_fft(data,kernel)
        output = vignetting(output,175)

        return output

@register_operator(name='lensless_psf_resize')
class PsfOperator_resize(LinearOperator):
    def __init__(self,args, device, **kwargs) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_config', type=str)
        parser.add_argument('--diffusion_config', type=str)
        parser.add_argument('--task_config', type=str)
        parser.add_argument('--gpu', type=int, default=4)
        parser.add_argument('--save_dir', type=str, default='./results')
        args = parser.parse_args()
        self.args = args
        GPU_NUM =self.args.gpu
        self.device = device

        print("GPU: ", GPU_NUM)

        self.psf_16 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v1/psf/psf_camera1_original.tiff", resize_dim=16).unsqueeze(0).cuda(GPU_NUM)
        self.psf_32 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v1/psf/psf_camera1_original.tiff", resize_dim=32).unsqueeze(0).cuda(GPU_NUM)
        self.psf_64 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v1/psf/psf_camera1_original.tiff", resize_dim=64).unsqueeze(0).cuda(GPU_NUM)
        self.psf_128 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v1/psf/psf_camera1_original.tiff", resize_dim=128).unsqueeze(0).cuda(GPU_NUM)
        self.psf_256 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v1/psf/psf_camera1_original.tiff", resize_dim=256).unsqueeze(0).cuda(GPU_NUM)
        self.psf_512 = load_psf(args = self.args, psf_file = "./samples/lensless_data/ys_v1/psf/psf_camera1_original.tiff", resize_dim=512).unsqueeze(0).cuda(GPU_NUM)
    
    def forward(self, data, **kwargs):
        return self.apply_kernel(data,kwargs.get('kernel_size'))

    def transpose(self, data, **kwargs):
        return data
    
    def apply_kernel(self, data, kernel_size):
        #TODO: faster way to apply conv?:W

        if kernel_size == 16:
            kernel = self.psf_16
        elif kernel_size == 32:
            kernel = self.psf_32
        elif kernel_size == 64:
            kernel = self.psf_64
        elif kernel_size == 128:
            kernel = self.psf_128
        elif kernel_size == 256:
            kernel = self.psf_256
        elif kernel_size == 512:
            kernel = self.psf_512


        output = conv_psf_fft(data,kernel)

        return output
    

### Real : Waller Dataset ###

@register_operator(name='lensless_flicker')
class PsfOperator(LinearOperator):
    def __init__(self,args, device, **kwargs) -> None:
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_config', type=str)
        parser.add_argument('--diffusion_config', type=str)
        parser.add_argument('--task_config', type=str)
        parser.add_argument('--gpu', type=int, default=4)
        parser.add_argument('--save_dir', type=str, default='./results')
        args = parser.parse_args()
        """
        self.args = args
        GPU_NUM =self.args.gpu
        self.device = device

        print("GPU: ", GPU_NUM)

        self.psf_512 = load_psf_waller(args = self.args, psf_file = "./samples/lensless_data/diffuser_v1/psf.tiff", resize_dim= 512).unsqueeze(0).cuda(GPU_NUM)
        self.psf_256 = load_psf_waller(args = self.args, psf_file = "./samples/lensless_data/diffuser_v1/psf.tiff", resize_dim= 256).unsqueeze(0).cuda(GPU_NUM)

        
        self.new_kernel = self.psf_256 
    
    def forward(self, data, **kwargs):
        #pad_size = 128
        #data_pad = F.pad(data,(pad_size,pad_size,pad_size,pad_size), mode='constant', value=-1) # # [1,3,256,256] =>[1,3,512,512]

        Ax = self.apply_kernel(data,kernel_size=256)
        # crop 4%
        # crop_Ax = crop_and_noise_2(Ax,self.args.crop,0)
        # vignetting
        crop_Ax = vignetting(Ax,175)

        # crop_Ax = Ax
        return crop_Ax
        
    def forward_nopad(self, data, **kwargs):
        return self.apply_kernel(data,kwargs.get('kernel_size'))

    def transpose(self, data, **kwargs):
        return data
    
    def apply_kernel(self, data, kernel_size):
        #TODO: faster way to apply conv?:W

        if kernel_size == 256:
            kernel = self.psf_256
        else:
            kernel = self.psf_512
        
        output = conv_psf_fft(data,kernel)

        

        return output



