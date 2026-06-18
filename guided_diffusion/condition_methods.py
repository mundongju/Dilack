from abc import ABC, abstractmethod
import torch
from guided_diffusion.custom_util import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from util.img_utils import clear_color

from util.logger import get_logger

from guided_diffusion.custom_util import *
import argparse

from scipy.stats import entropy

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
        self.scale = kwargs.get('scale')

        self.num_call = 0
        self.data_min_list = []
        self.data_max_list = []
        self.data_mean_list = []
        self.mask_patch_list = []
        self.threshold = 1.5
        self.threshold_init = 0
        self.threshold_mid_1 = 0
        self.decay = 0.998

        self.num_call_ind = 0

        self.DPSSAG_mask_save = 0

        self.mean_kl_matrix_dec_list = []

        
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    # original Gradient
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':

            a = measurement[0] # [1,3,512,512]
            pad_size = 128
            x_0_hat_pad = F.pad(x_0_hat,(pad_size,pad_size,pad_size,pad_size), mode='constant', value=-1) # # [1,3,256,256] =>[1,3,512,512]
            b = self.operator.forward(x_0_hat_pad,kernel_size=512, **kwargs)[0]
            b = crop_and_noise_2(b,450,0) 
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0] # [1,3,256,256]

            self.num_call += 1
        
        elif self.noiser.__name__ == 'sr_gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]


        else:
            raise NotImplementedError
        
        return norm_grad, norm
    
    # DPS gradient
    def grad_and_value_conv(self, x_prev, x_0_hat, measurement,deconv_guide,diff_scale,deconv_scale, **kwargs):
        if self.noiser.__name__ == 'gaussian':

            a = measurement[0] # [1,3,512,512]
            b = self.operator.forward(x_0_hat, **kwargs)[0] # [1,3,512,512]
            c = deconv_guide[0] # [1,3,256,256]
            d = x_0_hat[0] # [1,3,256,256]
            
            a = a[:,128:-128,128:-128] 
            b = b[:,128:-128,128:-128] 
            difference = normalize(a)-normalize(b) #normalize(a)-normalize(b)
            difference_deconv =  c-d #normalize(c)-normalize(d)
           
            norm_conv = torch.linalg.norm(difference)
            norm_dec = torch.linalg.norm(difference_deconv)
            norm_total = norm_conv
            
            norm_grad = torch.autograd.grad(outputs=norm_total, inputs=x_prev)[0]

            self.num_call += 1
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError

        return norm_grad, norm_total
    


    # Dilack Gradient
    def grad_and_value_dilack(self, x_prev, x_0_hat, measurement,deconv_guide,deconv_scale,patch_size,threshold_init,skip_point,YS, **kwargs):
        
        if self.noiser.__name__ == 'dps_gaussian':
            a = measurement[0] # [1,3,512,512]
            b = self.operator.forward(x_0_hat, **kwargs)[0] # [1,3,512,512]

            difference =  normalize(a) - normalize(b)
            norm = torch.linalg.norm(difference) 
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
            conv_mask = d = self.mean_kl_matrix_dec_list = arg_trans_S = 0
            arg_norm_true =1



        elif self.noiser.__name__ == 'gaussian':
            #########for size 128 dilack##################
            data_pad_crop_size = self.patch_size * 4
            data_total_size = data_pad_crop_size * 2
            ###############################################
            
            #YS data
            if YS :
                x_0_hat = vignetting(x_0_hat,150) # vignette!!
            
            arg_trans_S = False
            
            a = measurement[0] # [1,3,512,512]
            b = self.operator.forward(x_0_hat, **kwargs)[0] # [1,3,512,512]
            c = deconv_guide[0] # [1,3,256,256]
            d = x_0_hat[0] # [1,3,256,256]

            c = normalize(c)*2-1 #-1~1
            # print ('c')
            # print (c.max(),c.min()) #0~0
            # print ('d')
            # print (d.max(),d.min()) #1~-1
            
            difference_conv = normalize(a)-normalize(b)
            difference_deconv = abs(c-d) #normalize(c)-normalize(d)

            difference_conv_nonorm = abs(a-b)
            difference_deconv_nonorm = abs(c-d)

            difference_conv_nonorm = torch.sum(difference_conv_nonorm,dim=0)
            difference_deconv_nonorm = torch.sum(difference_deconv_nonorm,dim=0)


                        
            
            #mask making (original)
            unfolded_conv = difference_conv_nonorm.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            unfolded = difference_deconv_nonorm.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)

            patch_sums_conv = unfolded_conv.sum(dim=-1).sum(dim=-1)
            patch_sums = unfolded.sum(dim=-1).sum(dim=-1)

            quantile_th_conv = torch.quantile(patch_sums_conv, threshold_init)
            quantile_th = torch.quantile(patch_sums, threshold_init)
            patch_mask_conv = (patch_sums_conv > quantile_th_conv).int()
            patch_mask = (patch_sums > quantile_th).int()
            
            conv_mask = patch_mask_conv.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
            deconv_mask = patch_mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
            conv_mask = conv_mask.unsqueeze(0).repeat(3, 1, 1)
            deconv_mask = deconv_mask.unsqueeze(0).repeat(3, 1, 1)

            #mask making (dilack)
            
            shift_size = self.num_call_ind % 16
            difference_deconv_nonorm_cyc = cyclic_shift_torch(difference_deconv_nonorm,shift_size,shift_size)
            unfolded_cyc = difference_deconv_nonorm_cyc.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            patch_sums_cyc = unfolded_cyc.sum(dim=-1).sum(dim=-1)
            quantile_th_cyc = torch.quantile(patch_sums_cyc, threshold_init)
            patch_mask_cyc = (patch_sums_cyc > quantile_th_cyc).int()
            deconv_mask_cyc = patch_mask_cyc.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
            deconv_mask_cyc = deconv_mask_cyc.unsqueeze(0).repeat(3, 1, 1)



            norm = torch.linalg.norm(difference_conv)
            norm_dec = torch.linalg.norm(difference_deconv)   
            deconv_norm_only = norm_dec

            self.num_call_ind = self.num_call % 1000
            self.num_call += 1

            norm_deconv = torch.sum(deconv_norm_only)
            arg_norm_true = True
            # print("Dilack Guidance !!!!!")

            if self.num_call_ind <= skip_point:
                if self.num_call_ind % 2 ==0:
                    # print("Guiding on!!!")
                    norm_grad = deconv_scale * deconv_mask_cyc * torch.autograd.grad(outputs=norm_deconv, inputs=x_prev)[0]
                else:
                    # print("Guiding off!!!")
                    norm_grad = 0
            else:
                # print("Guiding on!!!")
                norm_grad = deconv_scale * deconv_mask_cyc * torch.autograd.grad(outputs=norm_deconv, inputs=x_prev)[0]


        
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError

        return norm_grad, norm, arg_norm_true,arg_trans_S  
    


#####################
#DPS sampling    
@register_conditioning_method(name='ps_conv')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale')
        self.diff_scale = kwargs.get('diff_scale')
        self.deconv_scale = kwargs.get('deconv_scale')

    def conditioning(self, x_prev, x_t, x_0_hat, measurement,deconv_guide,s_theta, **kwargs):
        norm_grad, norm_total = self.grad_and_value_conv(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement,deconv_guide=deconv_guide,diff_scale=self.diff_scale,deconv_scale=self.deconv_scale, **kwargs)
        s_theta = s_theta
        x_t -= norm_grad * self.scale
        return x_t, norm_total ,s_theta, norm_grad


#Dilack sampling
@register_conditioning_method(name='ps_dilack')
class PosteriorSamplingROI(ConditioningMethod):
    def __init__(self,args, operator, noiser, device='cuda', **kwargs):
        self.args = args
        GPU_NUM =self.args.gpu
        
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale')
        self.deconv_scale = kwargs.get('deconv_scale')
        
        self.device = device
        self.kernel_size = kwargs.get('kernel_size')
        self.exp_task = kwargs.get('exp_task')
        self.patch_size = kwargs.get('patch_size')
        self.threshold_init = kwargs.get('threshold_init')

        self.skip_point = kwargs.get('skip_point')

        self.YS = kwargs.get('YS')


    def conditioning(self, x_prev, x_t, x_0_hat, measurement,deconv_guide,s_theta, **kwargs):
        s_theta = s_theta
        norm_grad, norm_total,arg_norm_true, arg_trans_S = self.grad_and_value_dilack(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement,deconv_guide=deconv_guide,deconv_scale=self.deconv_scale,patch_size=self.patch_size, threshold_init=self.threshold_init,skip_point = self.skip_point,YS=self.YS,**kwargs)

        if arg_norm_true:
            x_t -= norm_grad * self.scale
            # print("conditional diffusion!")
        else:
            if self.num_call_ind <= 200:
                x_t -= norm_grad * self.scale
                # print("guiding mode!")
            else:
                if arg_trans_S:
                    x_t = x_t
                    # print("generation mode!")
                else:
                    x_t -= norm_grad * self.scale
                    # print("guiding mode!")
        
        return x_t, norm_total ,s_theta, norm_grad


    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        # self.scale = kwargs.get('scale', 1.0)
        self.scale = kwargs.get('scale')   
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale')

    def conditioning(self, x_prev, x_t, x_0_hat, measurement,s_theta, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        s_theta = s_theta
        x_t -= norm_grad * self.scale
        return x_t, norm ,s_theta, norm_grad
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        # self.scale = kwargs.get('scale', 1.0)
        self.scale = kwargs.get('scale')   

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm

