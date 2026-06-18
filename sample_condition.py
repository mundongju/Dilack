"""Unified zero-shot sampling script for Dilack.

This single entry point reproduces every experiment reported in the paper
"Fidelity-preserving zero-shot diffusion models for highly ill-posed inverse
problems in lensless imaging" (Image and Vision Computing, 2025).

The task is selected through the `measurement.operator.name` field of the task
config, and the script dispatches to the appropriate measurement / anchor
pipeline:

    Synthetic lensless        : lensless_voronoi, lensless_turing
    Synthetic motion deblur   : motion_blur
    Real (SNU/Yonsei Voronoi) : lensless_real_ys
    Real (Waller DiffuserCam) : lensless_real_waller

For every image we (i) form the lensless / blurry measurement y, (ii) compute a
classical Wiener and ADMM-TV pseudo-inverse anchor, and (iii) run the diffusion
posterior sampler guided by the masked PiAC fidelity (Dilack) or by the vanilla
DPS fidelity (ps_conv baseline).
"""

from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, clear_gray, mask_generator
from util.logger import get_logger
from guided_diffusion.custom_util import *

import ADMM_Torch_color
import ADMM_Torch_color_real
import ADMM_Torch_color_waller


# --- Task groups ------------------------------------------------------------
# Synthetic lensless tasks: the 256x256 image is convolved by a 512x512 PSF and
# the central measurement is cropped, so the measurement / anchor live at 512.
LENSLESS_SYN = ('lensless_voronoi', 'lensless_turing')
# Real tasks: the data loader yields (measurement, ground-truth) pairs.
PAIRED_OPS = ('lensless_real_ys', 'lensless_real_waller')
# Tasks whose ground-truth has to be re-formatted to the 512x512 capture grid.
GT_TO_GRID = ('lensless_voronoi', 'lensless_turing',
              'lensless_real_ys', 'lensless_real_waller')


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_admm_backend(op_name: str):
    """Return the ADMM-TV solver module matching the task."""
    if op_name == 'lensless_real_ys':
        return ADMM_Torch_color_real
    elif op_name == 'lensless_real_waller':
        return ADMM_Torch_color_waller
    return ADMM_Torch_color


def build_psf(args, op_name: str):
    """Load the PSF tensor for the lensless operators (None for motion blur)."""
    if op_name == 'lensless_voronoi':
        if args.voronoi_new:
            return load_psf(args, "./samples/lensless_data/ys_v4/psf_voronoi.png",
                            512, arg_voronoi=0)
        return load_psf(args, "./samples/lensless_data/ys_v1/psf/psf_camera1_original.tiff", 512)
    if op_name == 'lensless_turing':
        return load_psf(args, "./samples/lensless_data/ys_v2_dj_turing/psf/psf_turing.png",
                        512, 0)
    if op_name == 'lensless_real_ys':
        return load_psf_real(args, "./samples/dps_val/ys_flickr_100/psf/psf_camera1_original.tiff", 512)
    if op_name == 'lensless_real_waller':
        psf = load_psf_waller(args, "./samples/dps_val/diffusercam_100/psf.tiff", 256)
        return psf / psf.sum()
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--diffusion_config', type=str, required=True)
    parser.add_argument('--task_config', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--deconv_type', type=str, default='admm',
                        help="anchor used to guide sampling: 'admm', 'deconv' (Wiener) or 'noise'")
    parser.add_argument('--crop', type=int, default=450,
                        help='central measurement crop for synthetic lensless tasks (use 0 for Waller)')
    parser.add_argument('--noise', type=int, default=1,
                        help='add sensor Gaussian noise to the synthetic measurement')
    parser.add_argument('--wiener_alpha', type=float, default=0.0,
                        help='regularization of the Wiener pseudo-inverse anchor')
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--sdedit', type=int, default=0)
    parser.add_argument('--admm_only', type=int, default=0,
                        help='only run the classical anchor, skip diffusion sampling')
    parser.add_argument('--skip_processed', type=int, default=0,
                        help='resume by skipping images already written to recon/')
    parser.add_argument('--target_test', type=int, default=0)
    parser.add_argument('--fixed_seed', type=int, default=0)
    parser.add_argument('--resize_psf', type=int, default=0)
    parser.add_argument('--voronoi_new', type=int, default=1)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--sampling_seed', type=int, default=-1)
    # kept for backward-compatibility; the task is driven by the operator name.
    parser.add_argument('--is_real', type=int, default=0)
    parser.add_argument('--erase_ratio', type=float, default=0.0)
    parser.add_argument('--type', type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()

    logger = get_logger(log_file='experiment_eval.log', log_dir=args.save_dir)

    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    measure_config = task_config['measurement']
    op_name = measure_config['operator']['name']
    is_paired = op_name in PAIRED_OPS
    is_native_256 = op_name == 'lensless_real_waller'      # solves at 256, no central 128 crop
    admm_backend = get_admm_backend(op_name)

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare operator and noise
    operator = get_operator(args=args, device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {op_name} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser,
                                          args=args, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {cond_config['method']}")

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    if args.sdedit:
        sample_fn = partial(sampler.p_sample_loop_sdedit, model=model,
                            measurement_cond_fn=measurement_cond_fn)
    else:
        sample_fn = partial(sampler.p_sample_loop_deconv, model=model,
                            measurement_cond_fn=measurement_cond_fn)

    # Working directory
    out_path = os.path.join(args.save_dir, op_name,
                            task_config['data']['name'], cond_config['method'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label', 'admm', 'wiener', 'kernel']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    recon_path = os.path.join(out_path, 'recon')
    processed_num_files = count_png_files(recon_path) if args.skip_processed else 0

    # Prepare dataloader
    data_config = task_config['data']
    if op_name in LENSLESS_SYN:
        dataset = get_dataset(**data_config, transforms=None, normalize=False)
    elif op_name == 'lensless_real_ys':
        transform = transforms.Compose([transforms.Lambda(to_tensor_no_scaling)])
        dataset = get_dataset(**data_config, transforms=transform)
    elif op_name == 'lensless_real_waller':
        # the Waller dataset class defines its own resize/normalize transform.
        dataset = get_dataset(**data_config, transforms=None)
    else:
        dataset = get_dataset(**data_config, transforms=None, normalize=True)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    if op_name == 'inpainting':
        mask_gen = mask_generator(**measure_config['mask_opt'])

    # The real SNU/Yonsei PSF is fixed across the whole set.
    psf_fixed = build_psf(args, op_name) if op_name == 'lensless_real_ys' else None

    for i, batch in enumerate(loader):
        if is_paired:
            lq, ref_img = batch
            lq = lq.to(device)
        else:
            ref_img = batch
        ref_img = ref_img.to(device)

        if i < args.start_idx:
            print('skip', i)
            continue
        if args.target_test:
            if i < args.target_test:
                continue
            elif i > args.target_test:
                print("exit program")
                break

        # PSF for this image
        psf = psf_fixed if op_name == 'lensless_real_ys' else build_psf(args, op_name)

        # Per-image seeding. Synthetic tasks fix the kernel/noise per index; the
        # real SNU/Yonsei task keeps a single global seed.
        if op_name == 'motion_blur':
            seed_everything(args.fixed_seed if args.fixed_seed else 2024 + i)
            operator.__init__(args,
                              kernel_size=measure_config['operator']['kernel_size'],
                              intensity=measure_config['operator']['intensity'],
                              device=device)
        elif op_name != 'lensless_real_ys':
            seed_everything(args.fixed_seed if args.fixed_seed else 2024 + i)

        if i < processed_num_files:
            print("skip processed", i)
            continue
        if args.sampling_seed != -1:
            seed_everything(args.sampling_seed)
        if i == args.test_num:
            print("end of test")
            break

        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        mfname = str(i).zfill(5) + '.txt'

        if op_name in GT_TO_GRID:
            ref_img = transform_gt(ref_img)

        # --- Forward measurement y = A x (+ n) ------------------------------
        if op_name == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
        elif op_name in LENSLESS_SYN:
            y_n = operator.forward_nopad(ref_img, kernel_size=512)
            if args.crop:
                y_n, psf = crop_and_noise(y_n, psf, args.crop, 0)
            if args.noise:
                y_n = noiser(y_n)
        elif op_name == 'lensless_real_ys':
            lq = transform_padding_centercrop_real(lq[0], 512)
            y_n = lq.unsqueeze(0)
        elif op_name == 'lensless_real_waller':
            y_n = (lq + 1) / 2
        else:  # motion_blur
            y = operator.forward(ref_img)
            y_n = noiser(y) if args.noise else y

        # crop the ground-truth to the central 256x256 region for evaluation
        if op_name in GT_TO_GRID:
            ref_img = ref_img[..., 128:-128, 128:-128]

        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        if op_name in GT_TO_GRID:
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        else:
            plt.imsave(os.path.join(out_path, 'input', fname),
                       clear_color(y_n[..., 128:-128, 128:-128]))

        # --- Classical anchor + guided diffusion ----------------------------
        if args.deconv_type == 'noise':
            x_start = torch.randn((1, 3, 256, 256), device=device).requires_grad_()
            sample = sample_fn(x_start=x_start, measurement=y_n,
                               deconv_guide=None, record=True, save_root=out_path)

        elif args.deconv_type in ('admm', 'deconv'):
            kernel = operator.new_kernel
            plt.imsave(os.path.join(out_path, 'kernel', fname), clear_gray(kernel))

            # Wiener / ADMM kernel: the measured PSF for lensless, the padded
            # blur kernel for motion deblur.
            if psf is not None:
                anchor_kernel = psf
            else:
                anchor_kernel = F.pad(kernel, [128, 128, 128, 128])

            # Wiener pseudo-inverse anchor
            deconv_lq = deconv_psf(y_n, anchor_kernel, alpha=args.wiener_alpha)
            if op_name in LENSLESS_SYN and args.crop:
                pad = int((512 - args.crop) / 2)
                deconv_lq = F.pad(deconv_lq, (pad, pad, pad, pad))
            crop_deconv_lq = deconv_lq if is_native_256 else deconv_lq[:, :, 128:-128, 128:-128]

            plt.imsave(os.path.join(out_path, 'wiener', fname), clear_color(crop_deconv_lq))
            psnr, ssim, fid, lpips = show_metric(crop_deconv_lq, ref_img, device, "wiener")
            write_results(os.path.join(out_path, 'wiener', mfname), psnr, ssim, fid, lpips)

            # ADMM-TV pseudo-inverse anchor
            A = admm_backend.ADMM_LGE(device=device, iterations=1000, stacks=1,
                                      psf=anchor_kernel, infer=True, display=False,
                                      clamp_=1e-3)
            admm_img = A.forward(y_n)
            if op_name in LENSLESS_SYN and args.crop:
                pad = int((512 - args.crop) / 2)
                admm_img = F.pad(admm_img, (pad, pad, pad, pad))
                y_n = F.pad(y_n, (pad, pad, pad, pad))
            if not is_native_256:
                admm_img = admm_img[:, :, 128:-128, 128:-128]

            plt.imsave(os.path.join(out_path, 'admm', fname), clear_color(normalize(admm_img)))
            plt.imsave(os.path.join(out_path, 'wiener', fname), clear_color(normalize(crop_deconv_lq)))

            if args.deconv_type == 'admm':
                deconv_guide = normalize(admm_img)
                psnr, ssim, fid, lpips = show_metric(deconv_guide, ref_img, device, "admm")
            else:
                deconv_guide = normalize(crop_deconv_lq)
                psnr, ssim, fid, lpips = show_metric(deconv_guide, ref_img, device, "wiener")
            write_results(os.path.join(out_path, 'admm', mfname), psnr, ssim, fid, lpips)

            if args.admm_only:
                continue

            deconv_guide = deconv_guide * 2 - 1   # (0~1) -> (-1~1)
            if args.sdedit:
                x_start = sampler.q_sample(deconv_guide, args.sdedit).requires_grad_()
            else:
                x_start = torch.randn((1, 3, 256, 256), device=device).requires_grad_()
            sample = sample_fn(x_start=x_start, measurement=y_n,
                               deconv_guide=deconv_guide, record=True, save_root=out_path)

        else:
            x_start = torch.randn((1, 3, 256, 256), device=device).requires_grad_()
            sample = sample_fn(x_start=x_start, measurement=y_n,
                               deconv_guide=ref_img, record=True, save_root=out_path)

        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))
        psnr, ssim, fid, lpips = show_metric(sample, ref_img, device, "recon")
        write_results(os.path.join(out_path, 'recon', mfname), psnr, ssim, fid, lpips)


if __name__ == '__main__':
    seed_everything(2024)
    main()
