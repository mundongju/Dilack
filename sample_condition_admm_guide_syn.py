from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color,clear_gray, mask_generator
from util.logger import get_logger
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms.functional as f

from guided_diffusion.custom_util import *
from util.img_utils import dynamic_thresholding

import math
import pytorch_fid_wrapper as pfw

import ADMM_Torch_color


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--deconv_type', type=str, default='admm')
    parser.add_argument('--is_real', type=int, default=0)
    parser.add_argument('--crop', type=int, default=450)
    parser.add_argument('--noise', type=int, default=1)
    parser.add_argument('--wiener_alpha', type=float, default=0)
    parser.add_argument('--type', type=float, default=0.0)
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--sdedit', type=int, default=0)
    parser.add_argument('--admm_only', type=int, default=0)
    parser.add_argument('--skip_processed', type=int, default=0)
    parser.add_argument('--erase_ratio', type=float, default=0.0)
    parser.add_argument('--target_test', type=int, default=0)
    parser.add_argument('--fixed_seed', type=int, default=0)
    parser.add_argument('--resize_psf', type=int, default=0)
    parser.add_argument('--voronoi_new', type=int, default=1)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--sampling_seed', type=int, default=-1)

    args = parser.parse_args()
   
    # logger
    logger = get_logger(log_file='experiment_eval.log', log_dir=args.save_dir)
    data_num = 0
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(args=args,device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser,args=args, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 

    print("diffisuon_config")
    print (diffusion_config)
    if args.sdedit:
        sample_fn = partial(sampler.p_sample_loop_sdedit, model=model, measurement_cond_fn=measurement_cond_fn)
    else:
        sample_fn = partial(sampler.p_sample_loop_deconv, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'],task_config['data']['name'], task_config['conditioning']['method'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label','admm','wiener','kernel']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
        
        
    recon_path = os.path.join(out_path, 'recon')
    
    if args.skip_processed:
        processed_num_files = count_png_files(recon_path)
    else:
        processed_num_files = 0

    # Prepare dataloader
    data_config = task_config['data']

    transform_origin = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([transforms.Lambda(to_tensor_no_scaling)])
    
    
    if measure_config['operator']['name'] == 'lensless' or measure_config['operator']['name'] == 'lensless_custom' or measure_config['operator']['name'] == 'lensless_turing':
        dataset = get_dataset(**data_config, transforms=None, normalize=False)
    else:
        dataset = get_dataset(**data_config, transforms=None, normalize=True) 
    
    
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
    


    for i, ref_img in enumerate(loader):
        if args.start_idx>i:
            print ('skip',i)
            continue
        if measure_config['operator']['name'] == 'lensless' or  measure_config['operator']['name'] == 'lensless_custom':
            if args.voronoi_new:
                psf = load_psf(args, "./samples/lensless_data/ys_v4/psf_voronoi.png",512,arg_voronoi=0)
            else:
                psf = load_psf(args,"./samples/lensless_data/ys_v1/psf/psf_camera1_original.tiff",512)
        if measure_config['operator']['name'] == 'lensless_turing':
            psf = load_psf(args,"./samples/lensless_data/ys_v2_dj_turing/psf/psf_turing.png",512,0)

        
        #breakpoint()
                
        if args.target_test:
            if i<args.target_test:
                continue
            elif i>args.target_test:
                print ("exit program")
                exit(-1)
                
             
        if measure_config['operator']['name'] == 'motion_blur':
            
            if args.fixed_seed:
                seed_everything(args.fixed_seed)
                operator.__init__(args, kernel_size=measure_config['operator']['kernel_size'], intensity=measure_config['operator']['intensity'], device=device)
            else:
                seed_everything(2024+i) # setting kernel
                operator.__init__(args, kernel_size=measure_config['operator']['kernel_size'], intensity=measure_config['operator']['intensity'], device=device)
        else:
            if args.fixed_seed:
                seed_everything(args.fixed_seed)
            else:
                seed_everything(2024+i) # setting random
        
        if i < processed_num_files: 
            print ("processed_num_files: ",processed_num_files)
            print ("skip",i)
            continue

        if args.sampling_seed!=-1:
            seed_everything(args.sampling_seed) # for sampling
            
        
        if i == args.test_num:
            print ("end of test")
            break
        
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        mfname = str(i).zfill(5) + '.txt'
        
        ref_img = ref_img.to(device)

 
        
        
        if measure_config['operator']['name'] == 'lensless' or measure_config['operator']['name'] == 'lensless_turing' or measure_config['operator']['name'] == 'lensless_custom':
            ref_img = transform_gt(ref_img)

        
        print ('1.ref_img shape')
        print (ref_img.shape) # torch.Size([1, 3, 512, 512])

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        
        elif measure_config['operator'] ['name'] == 'lensless' or measure_config['operator']['name'] == 'lensless_turing' or measure_config['operator']['name'] == 'lensless_custom':


            if args.is_real:
                # real
                lq = transform_padding_centercrop(lq[0],512)
                y_n = lq.unsqueeze(0)
            else:
                # synsthetic
                print("synthetic!!")
                y_n = operator.forward_nopad(ref_img,kernel_size=512)
                
                if args.crop:
                    y_n,psf = crop_and_noise(y_n,psf,args.crop,0)
                # add noise
                if args.noise:
                    y_n = noiser(y_n)

            print ('y_n range',y_n.max(),y_n.min())
        else: 
            y = operator.forward(ref_img)
            if args.noise:
                y_n = noiser(y) # noise add
            else:
                y_n = y

        if measure_config['operator']['name'] == 'lensless' or measure_config['operator']['name'] == 'lensless_turing'  or measure_config['operator']['name'] == 'lensless_custom':
            ref_img = ref_img[...,128:-128,128:-128]
        
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        
        if measure_config['operator']['name'] == 'lensless' or measure_config['operator']['name'] == 'lensless_turing'  or measure_config['operator']['name'] == 'lensless_custom':
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        else:
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n[...,128:-128,128:-128]))
        
        if args.deconv_type == 'noise':
            x_start = torch.randn((1,3,256,256), device=device).requires_grad_()
            sample = sample_fn(x_start=x_start, measurement=y_n,deconv_guide=None, record=True, save_root=out_path)

        elif args.deconv_type == 'admm' or  args.deconv_type == 'deconv':
            
            kernel = operator.new_kernel
            pad_size = 128
            
            plt.imsave(os.path.join(out_path, 'kernel', fname), clear_gray(kernel))
            
            if measure_config['operator']['name'] == 'lensless'  or measure_config['operator']['name'] == 'lensless_turing'  or measure_config['operator']['name'] == 'lensless_custom':
                wiener_kernel = psf 
            else:
                wiener_kernel = F.pad(kernel,[pad_size,pad_size,pad_size,pad_size])

            deconv_lq = deconv_psf(y_n,wiener_kernel,alpha=args.wiener_alpha) 
            
              
            if (measure_config['operator']['name'] == 'lensless'  or measure_config['operator']['name'] == 'lensless_turing'  or measure_config['operator']['name'] == 'lensless_custom') and args.crop:
                pad_size = int((512-args.crop)/2)
                deconv_lq = F.pad(deconv_lq, (pad_size, pad_size, pad_size, pad_size))
 
            crop_deconv_lq = deconv_lq[:,:,128:-128,128:-128]  
            plt.imsave(os.path.join(out_path, 'wiener', fname), clear_color(crop_deconv_lq))
            psnr,ssim,fid,lpips = show_metric(crop_deconv_lq, ref_img, device,"wiener")
            write_results(os.path.join(out_path, 'wiener', mfname),psnr,ssim,fid,lpips)
            
            CLAMP = 1e-3
            
            if measure_config['operator']['name'] == 'lensless'  or measure_config['operator']['name'] == 'lensless_turing'  or measure_config['operator']['name'] == 'lensless_custom':
                admm_kernel = psf
            else:
                admm_kernel = F.pad(kernel, [pad_size, pad_size, pad_size, pad_size])
                
            A = ADMM_Torch_color.ADMM_LGE(device=device,
                                            iterations=1000,
                                            stacks=1,
                                            psf=admm_kernel,
                                            infer=True,
                                            display=False,
                                            clamp_=CLAMP,
                                            )
            
            print ("y_n.shape, kernel.shape, ref_img , start ADMM",y_n.shape, admm_kernel.shape)
            admm_img = A.forward(y_n)
            
            if (measure_config['operator']['name'] == 'lensless'  or measure_config['operator']['name'] == 'lensless_turing'  or measure_config['operator']['name'] == 'lensless_custom') and args.crop:
                pad_size = int((512-args.crop)/2)
                admm_img = F.pad(admm_img, (pad_size, pad_size, pad_size, pad_size))
                y_n = F.pad(y_n, (pad_size, pad_size, pad_size, pad_size))

            
            admm_img = admm_img[:,:,128:-128,128:-128]

            plt.imsave(os.path.join(out_path, 'admm', fname), clear_color(normalize(admm_img) ))
            plt.imsave(os.path.join(out_path, 'wiener', fname), clear_color(normalize(crop_deconv_lq) ))
            

            if args.deconv_type == 'admm':
                deconv_guide = normalize(admm_img) 
                psnr,ssim,fid,lpips = show_metric(deconv_guide, ref_img, device,"admm")
            else:
                deconv_guide = normalize(crop_deconv_lq) 
                psnr,ssim,fid,lpips = show_metric(deconv_guide, ref_img, device,"wiener")
        
            write_results(os.path.join(out_path, 'admm', mfname),psnr,ssim,fid,lpips)
            
            if args.admm_only:
                continue
            
            
            if args.sdedit:
                deconv_guide = deconv_guide *2 -1 #(0~1)=>(-1~1)
                x_start = sampler.q_sample(deconv_guide,args.sdedit).requires_grad_() 
                sample = sample_fn(x_start=x_start, measurement=y_n,deconv_guide=deconv_guide, record=True, save_root=out_path)#[256,256]
            else:
                deconv_guide = deconv_guide *2 -1 #(0~1)=>(-1~1)
                x_start = torch.randn((1,3,256,256), device=device).requires_grad_()
                sample = sample_fn(x_start=x_start, measurement=y_n,deconv_guide=deconv_guide, record=True, save_root=out_path)#[256,256]    
            
        else:
            # ADMM NO # test with ref_img
            x_start = torch.randn((1,3,256,256), device=device).requires_grad_()
            sample = sample_fn(x_start=x_start, measurement=y_n,deconv_guide=ref_img, record=True, save_root=out_path)

        print ("sample range")
        print (sample.max(),sample.min())
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

        
        psnr,ssim,fid,lpips = show_metric(sample, ref_img, device,"recon")
        write_results(os.path.join(out_path, 'recon', mfname),psnr,ssim,fid,lpips)
    

if __name__ == '__main__':
    seed_everything(2024)
    main()
