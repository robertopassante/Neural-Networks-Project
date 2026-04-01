# Satellite Image Segmentation with Pretrained Models

This project contains the foundational code to implement and experiment with various models and approaches for satellite image segmentation, as described in the assignment.

## Reference Files Included

Each file covers a specific aspect of the project or additional tasks:

1. **`satmae_backbone.py`**: Implementation of the pre-trained Multi-Spectral Vision Transformer (SatMAE++). Shows how to adapt the input to support images with more than 3 spectral bands (e.g., 10 bands for Sentinel-2).
2. **`sam_inference.py`**: Script to load and execute zero-shot inference with the **Segment Anything Model (SAM)** to generate pseudo-labels from optical satellite images.
3. **`swin_unet.py`**: Base structure of an Encoder-Decoder segmentation network (Swin-UNet) using Transformers as bottleneck layers, useful for final fine-tuning.
4. **`multimodal_fusion.py`**: Model for optical and radar (SAR - Sentinel-1) data fusion. Addresses one of the additional tasks to leverage radar's cloud-penetrating capability.
5. **`clip_segmentation.py`**: Integration of the **CLIP** textual encoder to allow text-based queries (e.g., `"find roads"`, `"find water"`), combining visual features and textual conditioning.
6. **`visualize_results.py`**: Tool to show segmentation output and compute parameters like IoU and Dice score against Ground Truths.
7. **`run_all.py`**: Training loop pipeline script built with `tqdm` and datasets for processing data over deep learning models directly out-of-the-box.

## Prerequisites

To run the base scripts you will need Python 3 and the following libraries:
```bash
pip install torch torchvision numpy opencv-python transformers minigpt4 tqdm
pip install git+https://github.com/facebookresearch/segment-anything.git # Optional for SAM
```

You can run individual Python files to see the tensor shape output and verify that the architecture syntax is correct.
