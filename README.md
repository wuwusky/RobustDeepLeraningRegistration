# Robust Deep Learning Registration

> **A deep-learning toolkit for robust image registration, panoramic-view construction, and registration-data annotation.**

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-required-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

This repository provides code for robust deep-learning-based image registration. It includes a learnable registration module, panoramic-view utilities, scripts for generating registration annotations, training and evaluation tools, and example image/video capture workflows.

## What is included

- **Registration model** — estimate the geometric relationship between an image pair for alignment.
- **Panoramic-view module** — compose aligned views into a wider field-of-view result.
- **Annotation generation** — automated and tool-assisted scripts for creating registration supervision.
- **Training and evaluation** — entry points for model training, cross-modal evaluation, and metric calculation.
- **Demos** — sample data plus image/video capture examples for testing the workflow.
- **Pretrained weights** — three checkpoints trained with distinct supervision and feature settings.

## Repository layout

```text
RobustDeepLeraningRegistration/
├── demo_data/                         # Example input data
├── demo_video_data_capture/           # Example video-capture data
├── Registration_Module.py              # Core registration module
├── Panoramic_View_Module.py            # Panoramic-view construction module
├── models.py                           # Neural-network definitions
├── train_RM.py                         # Registration-model training entry point
├── eval_different_modals.py            # Cross-modal evaluation
├── evaluation_metric.py                # Registration metrics
├── annotation_generation_auto.py       # Automatic annotation generation
├── annotation_generation_tool.py       # Tool-assisted annotation generation
├── demo_imaging_capture.py             # Image-capture demo
├── utils.py / utils_eff.py             # Shared utilities
├── border_mask.png                     # Border-mask asset
└── LICENSE                             # GPL-3.0 licence
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/wuwusky/RobustDeepLeraningRegistration.git
cd RobustDeepLeraningRegistration
```

### 2. Create an environment

The project is Python-based and uses a deep-learning runtime. A CUDA-capable PyTorch installation is recommended for model training and efficient inference.

```bash
conda create -n robust-registration python=3.10 -y
conda activate robust-registration

# Install the PyTorch build matching your CUDA version:
# https://pytorch.org/get-started/locally/
pip install torch torchvision

# Common image-processing and numerical dependencies
pip install numpy opencv-python pillow scipy scikit-image matplotlib tqdm
```

> This repository does not currently publish a pinned `requirements.txt`. After a successful setup, save the resolved environment (`pip freeze > requirements-lock.txt` or `conda env export`) to make experiments repeatable.

## Pretrained checkpoints

Choose a checkpoint according to the registration signal and training regime needed for your experiment.

| Checkpoint | Training / feature setting | Baidu Pan | Google Drive |
| --- | --- | --- | --- |
| `model_l` | Fast batch RANSAC; local feature information for registration | [Download](https://pan.baidu.com/s/1pvTNefMs8VDoExY4HzeWbw) (code: `m5zx`) | [Download](https://drive.google.com/file/d/1cuwVa0b4iVkZdTVKnKvxUQarSuSAzJEa/view?usp=sharing) |
| `model_w` | Fully connected linear layers; global / whole-image feature information | [Download](https://pan.baidu.com/s/1V2ZxVPgEwNuxAoZFIyIeWA) (code: `gtnt`) | [Download](https://drive.google.com/file/d/1fcjxpX_aKruuHRI-8s58DTZ8Ty-4_ClH/view?usp=sharing) |
| `model_ss` | Self-supervised learning without automatically annotated data | [Download](https://pan.baidu.com/s/1fxM6ZecY5yIXkL80DLo8OQ) (code: `fd8u`) | [Download](https://drive.google.com/file/d/1fkq3K4kVb8WZsCt9cNiPCa4T5dUe5N4f/view?usp=sharing) |

Place downloaded files outside version control, for example:

```text
checkpoints/
├── model_l.pth
├── model_w.pth
└── model_ss.pth
```

Set the selected checkpoint path in the relevant demo, evaluation, or model-loading section before execution. Add `checkpoints/` to `.gitignore` if you maintain a fork.

## Quick start

### 1. Inspect the demo data

Start with the provided `demo_data/` and `demo_video_data_capture/` directories. Confirm that paired inputs are ordered correctly: source image/video frame first, target image/video frame second.

### 2. Configure the run

Open the script you intend to use and set the following paths and options:

- source and target image/video locations;
- selected pretrained checkpoint;
- output directory for aligned images, transformation results, and panoramas;
- device (`cpu` or CUDA GPU) and input resolution;
- modality-specific preprocessing, where relevant.

### 3. Run the relevant workflow

```bash
# Capture or process example imaging data
python demo_imaging_capture.py

# Evaluate registration across modalities
python eval_different_modals.py

# Train a registration model after configuring data and hyperparameters
python train_RM.py
```

The scripts in this repository are configuration-driven. Review their top-level path, checkpoint, and hyperparameter settings before running; command-line arguments are not assumed where they are not explicitly defined.

## Training a registration model

Use [`train_RM.py`](train_RM.py) as the main training entry point. A typical training workflow is:

1. Assemble image pairs that represent the registration problem of interest.
2. Generate correspondence or transformation supervision with [`annotation_generation_auto.py`](annotation_generation_auto.py), or use [`annotation_generation_tool.py`](annotation_generation_tool.py) for assisted annotation.
3. Configure paths, batch size, learning rate, epochs, image size, and device in `train_RM.py`.
4. Train the model and save checkpoints to an external `checkpoints/` directory.
5. Evaluate only on image pairs withheld from training, using [`eval_different_modals.py`](eval_different_modals.py) and [`evaluation_metric.py`](evaluation_metric.py).

For reproducible experiments, record the image-pair split, annotation method, checkpoint, random seed, PyTorch/CUDA versions, and all geometric preprocessing steps.

## Working with panoramas

[`Panoramic_View_Module.py`](Panoramic_View_Module.py) contains the panoramic-view component. A practical pipeline is:

```text
Image/video frames
        ↓
Registration_Module.py estimates alignment
        ↓
Panoramic_View_Module.py composes the aligned views
        ↓
Panorama and intermediate registration outputs
```

Inspect output images after each stage. Registration errors compound during panorama construction, so verify overlap quality, border handling, and transformation direction on a small sequence before processing a complete video.

## Evaluation guidance

Use the same preprocessing and coordinate convention during evaluation as during training. In particular:

- retain the original source/target ordering for each pair;
- apply identical resize, crop, and normalization transforms;
- assess results qualitatively as well as with the metrics in `evaluation_metric.py`;
- evaluate each modality separately before drawing conclusions about cross-modal robustness;
- report failed registrations rather than silently excluding them.

## Licence

This project is released under the [GNU General Public License v3.0](LICENSE). Any redistribution or derivative work must comply with the terms of that licence.

## Acknowledgements and contact

Please use the [GitHub Issues page](https://github.com/wuwusky/RobustDeepLeraningRegistration/issues) for implementation questions, bug reports, and reproducibility discussions.

If you use this project in academic work, please consult the repository owner for the preferred citation, as a citation record is not currently provided in the repository.
