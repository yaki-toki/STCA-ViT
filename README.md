# STCA-ViT: Spatio-Temporal Cross-Attention Vision Transformer for Video Action Recognition

Official PyTorch implementation of **STCA-ViT**, a hybrid video action recognition framework combining a pretrained VideoMAE ViT-Base backbone with lightweight auxiliary modules for spatio-temporal understanding.

## Architecture Overview

```
x (B, 3, T, H, W)
├── VideoMAE ViT-Base (pretrained, partial fine-tuning) → b_back (B, 768)
├── STCA Module (bidirectional spatio-temporal cross-attention) → f_stca (B, D)
├── MGU Module (Sobel-based motion gating) → f_mgu (B, D) + temporal_info
└── Person Attention Module (YOLO-guided person-centric attention) → f_person (B, D)
         ↓
    4-way Residual Contribution Fusion → f_fused (B, 768)
         ↓
    Multi-Head Ensemble Classifier → logits (B, num_classes)
    [optional] WTAL Head → t_cam, action_scores, mil_logits
```

**Key modules:**
- **STCA Module**: Bidirectional cross-attention between spatial and temporal tokens
- **MGU (Motion Gating Unit)**: Sobel-based edge detection for motion-sensitive feature gating
- **Person Attention Module**: YOLO-based person detection to focus on action-relevant regions
- **WTAL Head**: Weakly-supervised temporal action localization (optional)

## Results

| Dataset | Top-1 Acc | Top-5 Acc |
|---------|-----------|-----------|
| UCF-101 | 98.29% | 99.96% |
| HMDB-51 | 90.46% | 99.03% |
| Kinetics-400 | 79.44% | 94.1% |
| Something-Something-v2 | 65.01% | 88.7% |

## Project Structure

```
stca_vit/
├── model.py              # STCA-ViT architecture
├── model_ablation.py     # Ablation study model variants (A0–A7)
├── train_ucf101.py       # Training script for UCF-101
├── train_hmdb51.py       # Training script for HMDB-51
├── train_kinetics400.py  # Training script for Kinetics-400
├── train_ssv2.py         # Training script for Something-Something-v2
├── test_ucf101.py        # Evaluation script for UCF-101
├── test_hmdb51.py        # Evaluation script for HMDB-51
├── test_kinetics400.py   # Evaluation script for Kinetics-400
├── test_ssv2.py          # Evaluation script for Something-Something-v2
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

YOLO-based person attention requires:
```bash
pip install ultralytics
```

## Usage

### Training

```bash
# UCF-101
python train_ucf101.py --data_root /path/to/UCF-101 --num_classes 101

# HMDB-51
python train_hmdb51.py --data_root /path/to/HMDB-51 --num_classes 51

# Kinetics-400
python train_kinetics400.py --data_root /path/to/Kinetics-400 --num_classes 400

# Something-Something-v2
python train_ssv2.py --data_root /path/to/SSv2 --num_classes 174
```

### Evaluation

```bash
# UCF-101
python test_ucf101.py --checkpoint STCA_ViT_UCF101_Best.pth

# HMDB-51
python test_hmdb51.py --checkpoint STCA_ViT_HMDB51_Best.pth

# Kinetics-400
python test_kinetics400.py --checkpoint STCA_ViT_Kinetics400_Best.pth

# Something-Something-v2
python test_ssv2.py --checkpoint STCA_ViT_SSv2_Best.pth
```

### Ablation Study

```python
from model_ablation import stca_vit_ablation

# A0: backbone only
model = stca_vit_ablation(config='A0', num_classes=51)

# A7: full model (all modules)
model = stca_vit_ablation(config='A7', num_classes=51)
```

Available configs: `A0` (backbone only), `A1` (+STCA), `A2` (+MGU), `A3` (+PersonAttn), `A4` (+WTAL), `A5` (+OrthoLoss), `A6` (+MultiHead), `A7` (full model).

## Dependencies

- Python 3.9+
- PyTorch 2.0+
- HuggingFace Transformers (VideoMAE)
- Ultralytics (YOLOv8, for person attention)
- Accelerate (distributed training)
- WandB (experiment tracking)

## Pretrained Backbone

The model uses [VideoMAE ViT-Base](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) pretrained on Kinetics-400, loaded automatically via HuggingFace Hub:

```python
videomae_name = 'MCG-NJU/videomae-base-finetuned-kinetics'
```
