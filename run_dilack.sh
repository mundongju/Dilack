#!/usr/bin/env bash
# =============================================================================
# Dilack — one reference command per task.
#
# Fidelity-preserving zero-shot diffusion models for highly ill-posed inverse
# problems in lensless imaging (Image and Vision Computing, 2025).
#
# All experiments share a single entry point (sample_condition.py); the task is
# selected by the chosen task config. Swap the ImageNet model/config for the
# FFHQ ones (configs/model/ffhq_model_config.yaml + the *_ffhq.yaml task config)
# to reproduce the face experiments. Use the configs under configs/baselines/
# to run the vanilla DPS baseline.
# =============================================================================

set -e

GPU=0
SAVE_DIR=./results
DIFFUSION=configs/diffusion_config.yaml
IMAGENET=configs/model/imagenet256_model_config.yaml
FFHQ=configs/model/ffhq_model_config.yaml

# ----------------------------------------------------------------------------
# 1) Synthetic lensless imaging — Voronoi-pattern PSF (ImageNet)
# ----------------------------------------------------------------------------
python sample_condition.py \
    --diffusion_config $DIFFUSION \
    --model_config $IMAGENET \
    --task_config configs/synthetic/lensless_voronoi_imagenet.yaml \
    --deconv_type admm --crop 450 --noise 1 --wiener_alpha 0 \
    --gpu $GPU --save_dir $SAVE_DIR

# ----------------------------------------------------------------------------
# 2) Synthetic lensless imaging — Turing-pattern PSF (ImageNet)
# ----------------------------------------------------------------------------
python sample_condition.py \
    --diffusion_config $DIFFUSION \
    --model_config $IMAGENET \
    --task_config configs/synthetic/lensless_turing_imagenet.yaml \
    --deconv_type admm --crop 450 --noise 1 --wiener_alpha 0 \
    --gpu $GPU --save_dir $SAVE_DIR

# ----------------------------------------------------------------------------
# 3) Severe synthetic motion deblurring (ImageNet)
# ----------------------------------------------------------------------------
python sample_condition.py \
    --diffusion_config $DIFFUSION \
    --model_config $IMAGENET \
    --task_config configs/synthetic/motion_deblur_imagenet.yaml \
    --deconv_type admm --noise 1 --wiener_alpha 0 \
    --gpu $GPU --save_dir $SAVE_DIR

# ----------------------------------------------------------------------------
# 4) Real lensless — SNU/Yonsei Voronoi camera (MirFlickr captures)
# ----------------------------------------------------------------------------
python sample_condition.py \
    --diffusion_config $DIFFUSION \
    --model_config $IMAGENET \
    --task_config configs/real/ys_flickr.yaml \
    --deconv_type admm --noise 0 --wiener_alpha 0 \
    --gpu $GPU --save_dir $SAVE_DIR

# ----------------------------------------------------------------------------
# 5) Real lensless — Waller-lab DiffuserCam (native 256, no central crop)
# ----------------------------------------------------------------------------
python sample_condition.py \
    --diffusion_config $DIFFUSION \
    --model_config $IMAGENET \
    --task_config configs/real/diffusercam.yaml \
    --deconv_type admm --crop 0 --noise 0 --wiener_alpha 0.001 \
    --gpu $GPU --save_dir $SAVE_DIR
