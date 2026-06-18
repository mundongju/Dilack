# Dilack: Fidelity-preserving Zero-shot Diffusion Models for Highly Ill-posed Inverse Problems in Lensless Imaging

Official implementation of the paper
**"Fidelity-preserving zero-shot diffusion models for highly ill-posed inverse problems in lensless imaging"**
(Haechang Lee\*, Dong Ju Mun\*, Hyunwoo Lee\*, Kyung Chul Lee, Seongmin Hong, Gwanghyun Kim, Seung Ah Lee†, Se Young Chun†),
*Image and Vision Computing (Elsevier), 2025.* &nbsp; [`doi:10.1016/j.imavis.2025.105786`](https://doi.org/10.1016/j.imavis.2025.105786)

<sub>\* equal contribution &nbsp;&nbsp; † corresponding authors &nbsp;&nbsp; Seoul National University</sub>

---

## Overview

**Dilack** (*Diffusion model for large and complex kernels*) is a **training-free, zero-shot**
diffusion solver for restoring raw images captured by **mask-based lensless cameras**, whose
forward operators are large, complex, and severely ill-conditioned (condition number κ(A) ≈ 7–15).
Plugging such operators into vanilla diffusion posterior sampling (DPS) leads to erratic behaviour
and strong artifacts. Dilack fixes this with two ingredients:

1. **PiAC fidelity (Pseudo-inverse Anchor for Constraining).** Instead of the least-squares term
   `‖y − Ax̂₀‖²`, Dilack guides the diffusion with a **pseudo-inverse anchor** `x* ≈ A†y`,
   computed by a stable Tikhonov/Wiener pseudo-inverse and refined by an **ADMM-TV** optimization.
   This is far more robust under high condition numbers.
2. **Masked fidelity (ROI mask).** A dynamic, patch-wise, *stochastic* mask restricts the PiAC
   guidance to the regions where the anchor and the current estimate disagree most. This lets the
   globally-acting diffusion prior interact with **spatially / step-wise local** fidelity,
   suppressing artifacts on severely ill-posed real data.

> **Abstract.** Diffusion models have been extensively explored for solving ill-posed inverse
> problems, achieving remarkable performance. However, their applicability to real-world scenarios,
> such as lensless imaging, has not been well investigated. Modern lensless imaging has compact form
> factor, low-cost hardware requirement, and intrinsic compressive imaging capabilities, but these
> advantages pose highly ill-posed inverse problems. In this work, we introduce a training-free
> zero-shot diffusion model, termed Dilack, for restoring raw images captured by lensless cameras
> that are degraded by large and complex kernels. Our approach incorporates novel data fidelity
> terms, referred to as the pseudo-inverse anchor for constraining (PiAC) fidelity loss, to enhance
> reconstruction quality by addressing the ill-posed nature of challenging inverse problems.
> Additionally, inspired by locally acting classical regularizers, we propose integrating masked
> fidelity within the PiAC loss. This scheme enables interaction with globally acting diffusion
> models while adaptively enforcing spatially and stepwise local fidelity through masks. Our proposed
> framework effectively mitigates erratic behavior and inherent artifacts in diffusion models when
> used for highly ill-posed inverse problems, significantly improving the quality of lensless camera
> raw image restoration, including perceptual aspects. Experimental results on both synthetic and
> real-world datasets for modern lensless imaging demonstrate that our approach outperforms prior
> arts including classical and existing diffusion based methods.

---

## Tasks

All experiments are launched from a **single entry point** (`sample_condition.py`); the task is
selected entirely by the task config. The benchmark is organized into three groups:

| Group | Task | Operator (`measurement.operator.name`) | Data | Config |
|-------|------|----------------------------------------|------|--------|
| **Synthetic** | Lensless — **Voronoi** PSF | `lensless_voronoi` | ImageNet / FFHQ | `configs/synthetic/lensless_voronoi_{imagenet,ffhq}.yaml` |
| **Synthetic** | Lensless — **Turing** PSF | `lensless_turing` | ImageNet / FFHQ | `configs/synthetic/lensless_turing_{imagenet,ffhq}.yaml` |
| **Synthetic** | **Severe motion deblurring** | `motion_blur` | ImageNet / FFHQ | `configs/synthetic/motion_deblur_{imagenet,ffhq}.yaml` |
| **Real** | **SNU/Yonsei** Voronoi camera | `lensless_real_ys` | MirFlickr captures (100) | `configs/real/ys_flickr.yaml` |
| **Real** | **Waller** DiffuserCam | `lensless_real_waller` | DiffuserCam DLMD (100) | `configs/real/diffusercam.yaml` |

The vanilla **DPS** baselines (`ps_conv`) used in the paper are provided under `configs/baselines/`.

---

## Repository structure

```
Dilack/
├── sample_condition.py          # Unified entry point for every task
├── run_dilack.sh                # One reference command per task
├── check_result_integrated.py   # Aggregate PSNR / SSIM / FID / LPIPS over a result folder
├── ADMM_Torch_color.py          # ADMM-TV pseudo-inverse anchor (synthetic + Waller)
├── ADMM_Torch_color_real.py     # ADMM-TV anchor for the SNU/Yonsei real data
├── configs/
│   ├── diffusion_config.yaml
│   ├── model/                   # imagenet256 / ffhq UNet configs
│   ├── synthetic/               # Voronoi / Turing / motion-deblur tasks
│   ├── real/                    # ys_flickr / diffusercam tasks
│   └── baselines/               # vanilla DPS (ps_conv) configs
├── guided_diffusion/            # UNet, DDPM sampler, measurements, conditioning (PiAC)
├── data/dataloader.py           # FFHQ / ImageNet / real lensless datasets
├── motionblur/                  # motion-blur kernel generator (LeviBorodenko)
├── util/                        # image utils, metrics, logger
├── models/                      # pretrained diffusion checkpoints (downloaded)
└── samples/                     # PSFs and datasets (downloaded)
```

---

## Prerequisites

- Python ≥ 3.8, CUDA-capable GPU
- PyTorch ≥ 1.12, torchvision ≥ 0.13

```bash
git clone https://github.com/mundongju/Dilack
cd Dilack

conda create -n dilack python=3.8
conda activate dilack
pip install -r requirements.txt
# install a CUDA build of torch/torchvision matching your system, e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Pretrained checkpoints

Dilack is **zero-shot**: it reuses off-the-shelf unconditional diffusion priors, identical to the
ones used by [DPS](https://github.com/DPS2022/diffusion-posterior-sampling).

```bash
mkdir -p models
```

| Model | File | Source |
|-------|------|--------|
| ImageNet 256×256 (unconditional) | `models/imagenet256.pt` | [guided-diffusion](https://github.com/openai/guided-diffusion) `256x256_diffusion_uncond.pt` |
| FFHQ 256×256 | `models/ffhq_10m.pt` | [DPS Google Drive](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing) |

```bash
# FFHQ (from the DPS Google Drive folder)
mv {DOWNLOAD_DIR}/ffhq_10m.pt ./models/

# ImageNet 256x256 unconditional (from OpenAI guided-diffusion)
wget -O models/imagenet256.pt \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```

---

## Datasets

### Synthetic tasks — ImageNet / FFHQ validation images

Identical to the DPS protocol: ~1,000 validation images per dataset at 256×256.

```
samples/dps_val/imagenet/   *.JPEG     # ImageNet-1k validation images
samples/dps_val/ffhq/       *.png      # FFHQ 256x256 validation images
```

The **synthetic PSFs** (custom Voronoi / Turing phase masks) are already shipped with the repo:

```
samples/lensless_data/ys_v4/psf_voronoi.png
samples/lensless_data/ys_v2_dj_turing/psf/psf_turing.png
```

### Real task 1 — SNU/Yonsei Voronoi lensless data (MirFlickr captures)

100 images from the [MirFlickr](https://press.liacs.nl/mirflickr/) dataset, displayed on a monitor
and **captured through our custom mask-based lensless optical system** (built-in Voronoi-pattern PSF,
working distance ≈ 20 cm). Raw measurements / ground-truth are originally 2464×3280, center-cropped
to 2400×2400 and resized to 512×512.

Download the packaged data and unzip it into `samples/dps_val/ys_flickr_100/`:

> 📦 **[Download `ys_flicker_100.zip` (Google Drive)](https://drive.google.com/file/d/REPLACE_WITH_YOUR_FILE_ID/view?usp=sharing)**
> &nbsp; *(replace with the shared link to your uploaded archive)*

```
samples/dps_val/ys_flickr_100/
├── raw/      im*.tiff   # lensless measurements (100)
├── label/    im*.jpg    # ground-truth (100)
└── psf/      psf_camera1_original.tiff
```

### Real task 2 — Waller-lab DiffuserCam data

100 samples (indices **10000–10099**) of the public **DiffuserCam Lensless MirFlickr Dataset (DLMD)**
released by the Waller lab.

- Dataset & PSF: <https://waller-lab.github.io/LenslessLearning/dataset.html>
- Reference: N. Antipa *et al.*, *"DiffuserCam: lensless single-exposure 3D imaging,"* Optica, 2018.

Place the measurement / ground-truth png pairs and the calibration PSF as follows:

```
samples/dps_val/diffusercam_100/
├── LQ/       *.png      # diffuser measurements (100, indices 10000-10099)
├── GT/       *.png      # ground-truth lensed images (100)
└── psf.tiff             # DiffuserCam calibration PSF
```

---

## Running

A reference command for every task is collected in [`run_dilack.sh`](run_dilack.sh):

```bash
bash run_dilack.sh
```

The general invocation mirrors DPS:

```bash
python sample_condition.py \
    --model_config     configs/model/imagenet256_model_config.yaml \
    --diffusion_config configs/diffusion_config.yaml \
    --task_config      configs/synthetic/lensless_voronoi_imagenet.yaml \
    --deconv_type admm --crop 450 --noise 1 \
    --gpu 0 --save_dir ./results
```

### Per-task examples

**Synthetic — Voronoi / Turing lensless** (use the matching `_ffhq.yaml` + `ffhq_model_config.yaml`
for faces):

```bash
python sample_condition.py --diffusion_config configs/diffusion_config.yaml \
    --model_config configs/model/imagenet256_model_config.yaml \
    --task_config  configs/synthetic/lensless_voronoi_imagenet.yaml \
    --deconv_type admm --crop 450 --noise 1 --wiener_alpha 0 --gpu 0 --save_dir ./results

python sample_condition.py --diffusion_config configs/diffusion_config.yaml \
    --model_config configs/model/imagenet256_model_config.yaml \
    --task_config  configs/synthetic/lensless_turing_imagenet.yaml \
    --deconv_type admm --crop 450 --noise 1 --wiener_alpha 0 --gpu 0 --save_dir ./results
```

**Synthetic — severe motion deblurring:**

```bash
python sample_condition.py --diffusion_config configs/diffusion_config.yaml \
    --model_config configs/model/imagenet256_model_config.yaml \
    --task_config  configs/synthetic/motion_deblur_imagenet.yaml \
    --deconv_type admm --noise 1 --wiener_alpha 0 --gpu 0 --save_dir ./results
```

**Real — SNU/Yonsei Voronoi camera:**

```bash
python sample_condition.py --diffusion_config configs/diffusion_config.yaml \
    --model_config configs/model/imagenet256_model_config.yaml \
    --task_config  configs/real/ys_flickr.yaml \
    --deconv_type admm --noise 0 --wiener_alpha 0 --gpu 0 --save_dir ./results
```

**Real — Waller DiffuserCam** (native 256, so `--crop 0`):

```bash
python sample_condition.py --diffusion_config configs/diffusion_config.yaml \
    --model_config configs/model/imagenet256_model_config.yaml \
    --task_config  configs/real/diffusercam.yaml \
    --deconv_type admm --crop 0 --noise 0 --wiener_alpha 0.001 --gpu 0 --save_dir ./results
```

### Key command-line arguments

| Argument | Meaning |
|----------|---------|
| `--deconv_type` | anchor used to guide sampling: `admm` (Dilack default), `deconv` (Wiener only), `noise` (no anchor) |
| `--crop` | central measurement crop for synthetic lensless (450); use `0` for Waller |
| `--noise` | add sensor Gaussian noise to the synthetic measurement (`1` synthetic, `0` real) |
| `--wiener_alpha` | regularization of the Wiener pseudo-inverse anchor |
| `--admm_only` | only compute the classical anchor and skip diffusion sampling |
| `--skip_processed` | resume by skipping images already written to `recon/` |
| `--gpu`, `--save_dir` | device id / output directory |

### Outputs

For each task a folder `results/<operator>/<data>/<method>/` is created with sub-folders
`label/`, `input/`, `wiener/`, `admm/`, `recon/`, `progress/`, and per-image
`PSNR,SSIM,FID,LPIPS` text files.

---

## Evaluation

Aggregate the metrics over a folder of reconstructions against the ground-truth:

```bash
python check_result_integrated.py results/<operator>/<data>/<method>/label \
                                  results/<operator>/<data>/<method>/recon
```

---

## Results (highlights)

PSNR↑ / SSIM↑ / FID↓ / LPIPS↓. Full tables are in the paper.

**Synthetic lensless (ImageNet, Voronoi PSF):**

| Method | PSNR | SSIM | FID | LPIPS |
|--------|------|------|-----|-------|
| Wiener (A†y) | 13.17 | 0.274 | 241.33 | 0.606 |
| ADMM-TV | 19.74 | 0.574 | 36.45 | 0.299 |
| DPS | 8.13 | 0.268 | 130.77 | 0.666 |
| **PiAC (w/o ROI)** | **23.02** | **0.791** | 40.01 | **0.232** |
| **Dilack** | 22.88 | 0.773 | 41.54 | 0.250 |

**Real data (full Dilack with ROI mask):**

| Dataset | PSNR | SSIM | FID | LPIPS |
|---------|------|------|-----|-------|
| DiffuserCam (Waller) | **14.74** | **0.512** | **214.76** | **0.376** |
| MirFlickr-Lensless (SNU/Yonsei) | **13.47** | **0.326** | 290.54 | **0.584** |

On highly ill-posed lensless data the vanilla DPS collapses (PSNR ≈ 7–10), while Dilack restores
fidelity (+14–18 dB over DPS on synthetic lensless), and the ROI mask gives a further boost on real
data.

---

## Citation

```bibtex
@article{lee2025dilack,
  title   = {Fidelity-preserving zero-shot diffusion models for highly ill-posed inverse problems in lensless imaging},
  author  = {Lee, Haechang and Mun, Dong Ju and Lee, Hyunwoo and Lee, Kyung Chul and Hong, Seongmin and Kim, Gwanghyun and Lee, Seung Ah and Chun, Se Young},
  journal = {Image and Vision Computing},
  publisher = {Elsevier},
  year    = {2025},
  doi     = {10.1016/j.imavis.2025.105786},
  note    = {Article 105786}
}
```

---

## Acknowledgements

This codebase builds upon [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) and
[guided-diffusion](https://github.com/openai/guided-diffusion) for the diffusion priors and sampler,
[motionblur](https://github.com/LeviBorodenko/motionblur) for the motion kernels, and the
[Waller-lab DiffuserCam](https://waller-lab.github.io/LenslessLearning/) dataset for the real
lensless benchmark. We thank the authors for releasing their code and data.
