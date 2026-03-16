"""
STCA-ViT Training for HMDB-51 (Full Dataset)
====================================================

Key improvements over the previous version:
-------------------
1. Backbone: R2Plus1D-18 (512-d) -> VideoMAE ViT-Base (768-d, Kinetics pretrained)
2. embed_dim: 192 -> 768
3. Resolution: 112x112 (VideoMAE backbone auto-resizes to 224 internally; other modules use 112)
4. Training settings: lr=5e-5, weight_decay=0.05, backbone_lr_ratio=0.01
5. decord-based fast video loading
6. RandAugment + CutMix + Random Erasing + Temporal Aug

Data:
- training_*.avi: Training + Validation (8:2 split)
- testing_*.avi: Test

Usage (Single GPU):
---------
python train_hmdb51.py --try_number 1

Usage (with WTAL enabled):
---------
python train_hmdb51.py --try_number 1 --use_wtal

Usage (Accelerate - Multi-GPU):
---------
accelerate launch train_hmdb51.py --try_number 1 --use_wtal
"""

import os
import sys
import random
import cv2
import numpy as np
import torch
import gc
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
import torch.backends.cudnn as cudnn
import platform
import argparse
import multiprocessing
import psutil

import torch.nn.functional as F
import decord
decord.bridge.set_bridge('torch')

import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast  # PyTorch 1.x compatible
from copy import deepcopy

# Accelerate for distributed training
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

# STCA-ViT model import
sys.path.insert(0, '/home/user/data/codes')
from model import stca_vit_v2_5_parallel

# Suppress OpenCV/FFmpeg error messages
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'


# ===========================
# Constants
# ===========================
AUX_LOSS_WEIGHT = 0.3
ORTHO_LOSS_WEIGHT = 0.1
GRL_LAMBDA_INIT = 0.0
GRL_LAMBDA_MAX = 0.1
WTAL_LOSS_WEIGHT = 0.3
WTAL_WARMUP_EPOCHS = 5  # Start WTAL after classification has stabilized


# ===========================
# Utility
# ===========================
def calculate_model_size(model):
    param_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4
    return param_size / (1024 * 1024)


# ============================================================================
# RandAugment for Video (Spatial)
# ============================================================================
class VideoRandAugment:
    """RandAugment for video - applies the same augmentation to each frame."""
    def __init__(self, n=2, m=9):
        self.n = n
        self.m = m
        self.ops = [
            'autocontrast', 'equalize', 'rotate', 'color',
            'contrast', 'brightness', 'sharpness', 'shearx', 'sheary',
            'translatex', 'translatey'
        ]

    def __call__(self, frames):
        ops = random.choices(self.ops, k=self.n)
        magnitude = self.m / 10.0
        sign = random.choice([-1, 1])

        augmented = []
        for frame in frames:
            aug_frame = frame.copy()
            if aug_frame.dtype != np.uint8:
                aug_frame = np.clip(aug_frame, 0, 255).astype(np.uint8)
            for op in ops:
                aug_frame = self._apply_op(aug_frame, op, magnitude, sign)
                if aug_frame.dtype != np.uint8:
                    aug_frame = np.clip(aug_frame, 0, 255).astype(np.uint8)
            augmented.append(aug_frame)
        return augmented

    def _apply_op(self, img, op, magnitude, sign):
        if op == 'autocontrast':
            return self._autocontrast(img)
        elif op == 'equalize':
            return self._equalize(img)
        elif op == 'rotate':
            angle = magnitude * 30 * sign
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        elif op == 'color':
            return self._adjust_color(img, max(1 + magnitude * 0.9 * sign, 0.1))
        elif op == 'contrast':
            return self._adjust_contrast(img, max(1 + magnitude * 0.9 * sign, 0.1))
        elif op == 'brightness':
            return self._adjust_brightness(img, max(1 + magnitude * 0.9 * sign, 0.1))
        elif op == 'sharpness':
            return self._adjust_sharpness(img, max(1 + magnitude * 0.9 * sign, 0.1))
        elif op == 'shearx':
            return self._shear(img, magnitude * 0.3 * sign, axis='x')
        elif op == 'sheary':
            return self._shear(img, magnitude * 0.3 * sign, axis='y')
        elif op == 'translatex':
            return self._translate(img, magnitude * 0.3 * sign, axis='x')
        elif op == 'translatey':
            return self._translate(img, magnitude * 0.3 * sign, axis='y')
        return img

    def _autocontrast(self, img):
        result = np.zeros_like(img)
        for c in range(3):
            ch = img[:,:,c]
            lo, hi = ch.min(), ch.max()
            if hi > lo:
                result[:,:,c] = np.clip((ch.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
            else:
                result[:,:,c] = ch
        return result

    def _equalize(self, img):
        result = np.zeros_like(img)
        for c in range(3):
            result[:,:,c] = cv2.equalizeHist(img[:,:,c])
        return result

    def _adjust_color(self, img, factor):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return np.clip(img.astype(np.float32) * factor + gray_3ch.astype(np.float32) * (1 - factor),
                       0, 255).astype(np.uint8)

    def _adjust_contrast(self, img, factor):
        mean = img.astype(np.float32).mean()
        return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    def _adjust_brightness(self, img, factor):
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def _adjust_sharpness(self, img, factor):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(img, -1, kernel)
        return np.clip(img.astype(np.float32) * (1 - factor) + sharpened.astype(np.float32) * factor,
                       0, 255).astype(np.uint8)

    def _shear(self, img, magnitude, axis='x'):
        h, w = img.shape[:2]
        if axis == 'x':
            M = np.float32([[1, magnitude, 0], [0, 1, 0]])
        else:
            M = np.float32([[1, 0, 0], [magnitude, 1, 0]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _translate(self, img, magnitude, axis='x'):
        h, w = img.shape[:2]
        pixels = int(magnitude * (w if axis == 'x' else h))
        if axis == 'x':
            M = np.float32([[1, 0, pixels], [0, 1, 0]])
        else:
            M = np.float32([[1, 0, 0], [0, 1, pixels]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


# ============================================================================
# VideoRandomErasing (tensor-level)
# ============================================================================
class VideoRandomErasing:
    """Random Erasing for video tensors - maintains temporal consistency."""
    def __init__(self, probability=0.25, sl=0.02, sh=0.33, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, video_tensor):
        if random.random() > self.probability:
            return video_tensor
        C, T, H, W = video_tensor.shape
        area = H * W
        for _ in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            rh = int(round(np.sqrt(target_area * aspect_ratio)))
            rw = int(round(np.sqrt(target_area / aspect_ratio)))
            if rw < W and rh < H:
                x1 = random.randint(0, H - rh)
                y1 = random.randint(0, W - rw)
                video_tensor[:, :, x1:x1+rh, y1:y1+rw] = torch.randn(C, T, rh, rw) * 0.1
                return video_tensor
        return video_tensor


# ===========================
# HMDB-51 Dataset (Full Dataset, decord + augmentation)
# ===========================
class HMDB51FullDataset(Dataset):
    """
    HMDB-51 Full Dataset
    - training_*.avi -> Training + Validation (8:2 stratified split)
    - testing_*.avi -> Test
    - decord-based fast video loading
    - Supports RandAugment, Random Erasing, and Temporal Augmentation
    - 112x112 resolution (speed optimization; VideoMAE backbone auto-resizes to 224 internally)
    """
    KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
    KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

    def __init__(self, data_root, split='train', clip_length=16, frame_stride=2,
                 w_size=112, h_size=112, val_ratio=0.2, seed=42, random_flip=True,
                 use_randaugment=False, randaug_n=2, randaug_m=9,
                 use_random_erasing=False, erasing_prob=0.25,
                 use_temporal_aug=False, speed_range=(0.8, 1.2)):
        self.data_root = data_root
        self.split = split
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.w_size = w_size
        self.h_size = h_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.random_flip = random_flip
        self.use_temporal_aug = use_temporal_aug
        self.speed_range = speed_range

        # Augmentation
        self.rand_augment = VideoRandAugment(n=randaug_n, m=randaug_m) if (use_randaugment and split == 'train') else None
        self.random_erasing = VideoRandomErasing(probability=erasing_prob) if (use_random_erasing and split == 'train') else None

        # Class list (folder name = class name)
        self.classes = sorted([d for d in os.listdir(data_root)
                               if os.path.isdir(os.path.join(data_root, d))
                               and any(f.endswith('.avi') for f in os.listdir(os.path.join(data_root, d)))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        print(f"HMDB-51 number of classes: {len(self.classes)}")

        # Load samples
        self.samples = self._load_samples()
        split_name = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}[split]
        print(f"HMDB-51 {split_name}: {len(self.samples)} videos")
        print(f"   Frame sampling: {clip_length} frames x stride {frame_stride} = {clip_length * frame_stride} frame span")

    def _load_samples(self):
        """Load samples separated by training_*.avi and testing_*.avi prefixes."""
        train_samples = []
        test_samples = []

        for class_name in self.classes:
            class_dir = os.path.join(self.data_root, class_name)
            if not os.path.isdir(class_dir):
                continue

            for video_name in os.listdir(class_dir):
                if not video_name.endswith('.avi'):
                    continue

                video_path = os.path.join(class_dir, video_name)
                label = self.class_to_idx[class_name]

                if video_name.startswith('training_'):
                    train_samples.append((video_path, label))
                elif video_name.startswith('testing_'):
                    test_samples.append((video_path, label))

        if self.split == 'test':
            return test_samples

        # Split training_*.avi into train/val (per-class stratified split)
        class_samples = defaultdict(list)
        for video_path, label in train_samples:
            class_samples[label].append((video_path, label))

        random.seed(self.seed)
        final_train, final_val = [], []

        for label in sorted(class_samples.keys()):
            samples = class_samples[label]
            random.shuffle(samples)
            n_val = max(1, int(len(samples) * self.val_ratio))
            final_val.extend(samples[:n_val])
            final_train.extend(samples[n_val:])

        if self.split == 'train':
            return final_train
        elif self.split == 'val':
            return final_val
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_video(video_path)
        return frames, label

    def _get_frame_indices(self, total_frames):
        """Determine frame indices - Train: random + temporal aug, Val/Test: center."""
        stride = self.frame_stride

        # Temporal augmentation: speed jittering during training
        if self.split == 'train' and self.use_temporal_aug:
            speed = random.uniform(self.speed_range[0], self.speed_range[1])
            stride = max(1, int(self.frame_stride * speed))

        span = self.clip_length * stride

        if total_frames >= span:
            max_start = total_frames - span
            if self.split == 'train':
                start = random.randint(0, max_start)
            else:
                start = max_start // 2  # center crop for val/test
        else:
            start = 0

        indices = [min(start + i * stride, max(total_frames - 1, 0))
                   for i in range(self.clip_length)]
        return indices

    def _load_video(self, video_path):
        """Load video using decord."""
        try:
            vr = decord.VideoReader(video_path, num_threads=1)
            total_frames = len(vr)

            if total_frames == 0:
                return torch.zeros((3, self.clip_length, self.h_size, self.w_size))

            frame_indices = self._get_frame_indices(total_frames)

            # Batch frame decoding with decord
            frame_indices_clamped = [min(idx, total_frames - 1) for idx in frame_indices]
            frames_tensor = vr.get_batch(frame_indices_clamped)  # (T, H, W, C) torch tensor

            # Decide random flip only for training
            do_flip = self.split == 'train' and self.random_flip and random.random() < 0.5

            # Convert to numpy for augmentation
            frames_np = frames_tensor.numpy().astype(np.uint8)  # (T, H, W, C) RGB

            # RandAugment (per frame on uint8)
            if self.rand_augment is not None:
                frame_list = [frames_np[t] for t in range(frames_np.shape[0])]
                frame_list = self.rand_augment(frame_list)
                frames_np = np.stack(frame_list, axis=0)

            # Resize + Flip + Normalize
            processed = []
            for t in range(frames_np.shape[0]):
                frame = frames_np[t]  # (H, W, C) RGB uint8

                if do_flip:
                    frame = cv2.flip(frame, 1)

                # Resize to target
                frame = cv2.resize(frame, (self.w_size, self.h_size))

                # Normalize: [0,255] -> [0,1] -> Kinetics norm
                frame = frame.astype(np.float32) / 255.0
                frame = (frame - self.KINETICS_MEAN) / self.KINETICS_STD
                processed.append(frame)

            # (T, H, W, C) -> (C, T, H, W)
            clip = np.stack(processed, axis=0)
            clip = clip.transpose(3, 0, 1, 2)
            clip_tensor = torch.from_numpy(clip).float()

            # Random Erasing (tensor-level)
            if self.random_erasing is not None:
                clip_tensor = self.random_erasing(clip_tensor)

            return clip_tensor

        except Exception as e:
            # Return zero tensor on failure
            return torch.zeros((3, self.clip_length, self.h_size, self.w_size))


# ===========================
# Label Smoothing Loss
# ===========================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        num_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(true_dist * log_probs).sum(dim=1)

        if self.weight is not None:
            sample_weights = self.weight[target]
            loss = loss * sample_weights

        return loss.mean()


# ===========================
# EMA
# ===========================
class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        self.device = device if device is not None else torch.device('cpu')
        self.module = deepcopy(model).to(self.device)
        self.module.eval()
        self.updates = 0

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            decay = min(self.decay, (1 + self.updates) / (10 + self.updates))

            state_dict = model.state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    src = state_dict[k].detach()
                    if src.device != v.device:
                        src = src.to(v.device)
                    v *= decay
                    v += (1.0 - decay) * src


# ===========================
# MixUp / CutMix Augmentation
# ===========================
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Random box
    _, _, T, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cy = random.randint(0, H)
    cx = random.randint(0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    x[:, :, :, y1:y2, x1:x2] = x[index, :, :, y1:y2, x1:x2]
    lam = 1 - ((y2 - y1) * (x2 - x1) / (H * W))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ===========================
# Training Functions
# ===========================
def train_epoch(model, train_loader, criterion, optimizer, accelerator, device,
                model_ema=None, accumulation_steps=1, max_grad_norm=1.0, use_aux_loss=True,
                use_mixup=False, mixup_alpha=0.2, use_cutmix=False, cutmix_alpha=1.0,
                use_ortho_loss=True, ortho_weight=ORTHO_LOSS_WEIGHT,
                use_wtal=False, wtal_weight=WTAL_LOSS_WEIGHT,
                current_epoch=0, wtal_warmup=WTAL_WARMUP_EPOCHS):
    model.train()
    running_loss = 0.0
    running_ortho_loss = 0.0
    running_wtal_loss = 0.0
    correct = 0
    total = 0
    nan_count = 0

    # WTAL activates only after warmup + gradual ramp-up
    wtal_active = use_wtal and (current_epoch >= wtal_warmup)
    if wtal_active:
        # Ramp from 0 to 1 over 5 epochs after warmup
        ramp_progress = min(1.0, (current_epoch - wtal_warmup) / 5.0)
        effective_wtal_weight = wtal_weight * ramp_progress
    else:
        effective_wtal_weight = 0.0

    if accelerator.is_main_process:
        train_bar = tqdm(train_loader, desc="Training", leave=False)
    else:
        train_bar = train_loader

    for i, (videos, labels) in enumerate(train_bar):
        # CutMix (30%) / MixUp (20%) / Normal (50%)
        mixed = False
        if use_cutmix and np.random.random() < 0.3:
            videos, labels_a, labels_b, lam = cutmix_data(videos, labels, cutmix_alpha)
            mixed = True
        elif use_mixup and np.random.random() < 0.4:
            videos, labels_a, labels_b, lam = mixup_data(videos, labels, mixup_alpha)
            mixed = True
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        with accelerator.accumulate(model):
            with accelerator.autocast():
                if use_aux_loss:
                    outputs_dict = model(videos, return_all=True,
                                         compute_ortho_loss=use_ortho_loss,
                                         compute_wtal=wtal_active)
                    final_logits = outputs_dict['final']

                    if torch.isnan(final_logits).any():
                        nan_count += 1
                        if accelerator.is_main_process:
                            train_bar.set_postfix(loss='NaN_output', nan_count=nan_count)
                        continue

                    # Main loss
                    if mixed:
                        main_loss = mixup_criterion(criterion, final_logits, labels_a, labels_b, lam)
                    else:
                        main_loss = criterion(final_logits, labels)

                    # Auxiliary loss (per-head individual losses)
                    aux_loss = 0.0
                    num_aux = 0

                    for key in ['backbone', 'cross_attn', 'mgu', 'person', 'fused']:
                        if key in outputs_dict and outputs_dict[key] is not None:
                            if mixed:
                                aux_loss += mixup_criterion(criterion, outputs_dict[key], labels_a, labels_b, lam)
                            else:
                                aux_loss += criterion(outputs_dict[key], labels)
                            num_aux += 1

                    if num_aux > 0:
                        aux_loss = aux_loss / num_aux

                    loss = main_loss + AUX_LOSS_WEIGHT * aux_loss

                    # Orthogonal Loss
                    if use_ortho_loss and 'ortho_loss' in outputs_dict:
                        ortho_loss = outputs_dict['ortho_loss']
                        loss = loss + ortho_weight * ortho_loss
                        running_ortho_loss += ortho_loss.item()

                    # WTAL Loss
                    if wtal_active and 'tal_outputs' in outputs_dict:
                        unwrapped = accelerator.unwrap_model(model)
                        wtal_loss, wtal_loss_dict = unwrapped.wtal_loss_fn(
                            outputs_dict['tal_outputs'],
                            labels,
                            skip_mil=mixed  # Skip MIL CE loss during MixUp/CutMix
                        )
                        loss = loss + effective_wtal_weight * wtal_loss
                        running_wtal_loss += wtal_loss.item()

                    outputs = final_logits
                else:
                    if use_ortho_loss:
                        outputs, ortho_loss = model(videos, compute_ortho_loss=True)
                    else:
                        outputs = model(videos)
                        ortho_loss = None

                    if torch.isnan(outputs).any():
                        nan_count += 1
                        if accelerator.is_main_process:
                            train_bar.set_postfix(loss='NaN_output', nan_count=nan_count)
                        continue

                    if mixed:
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = criterion(outputs, labels)

                    if use_ortho_loss and ortho_loss is not None:
                        loss = loss + ortho_weight * ortho_loss
                        running_ortho_loss += ortho_loss.item()

            # NaN/Inf loss check
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if accelerator.is_main_process:
                    train_bar.set_postfix(loss='NaN', nan_count=nan_count)
                continue

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            if model_ema is not None and accelerator.sync_gradients:
                model_ema.update(accelerator.unwrap_model(model))

        _, predicted = torch.max(outputs, 1)
        total += labels_a.size(0)
        correct += (predicted == labels_a).sum().item()
        running_loss += loss.item()
        if accelerator.is_main_process:
            train_bar.set_postfix(loss=f'{loss.item():.4f}')

    if nan_count > 0 and accelerator.is_main_process:
        print(f"  NaN occurrences: {nan_count}")

    num_batches = len(train_loader)
    if num_batches > 0:
        avg_loss = running_loss / num_batches
        avg_ortho_loss = running_ortho_loss / num_batches if running_ortho_loss > 0 else 0
        avg_wtal_loss = running_wtal_loss / num_batches if running_wtal_loss > 0 else 0
        acc = 100 * correct / total if total > 0 else 0
        return avg_loss, acc, avg_ortho_loss, avg_wtal_loss
    return float('inf'), 0, 0, 0


def validate_epoch(model, val_loader, criterion, accelerator, use_tta=False,
                   compute_wtal_metrics=False):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    # WTAL proxy metrics
    wtal_fg_scores = []
    wtal_fg_sparsity = []
    wtal_peak_positions = []

    with torch.no_grad():
        if accelerator.is_main_process:
            val_bar = tqdm(val_loader, desc="Validation", leave=False)
        else:
            val_bar = val_loader

        for videos, labels in val_bar:
            with accelerator.autocast():
                if compute_wtal_metrics:
                    # Collect WTAL metrics: return_all + compute_wtal
                    outputs_dict = model(videos, return_all=True, compute_wtal=True)
                    outputs = outputs_dict['final']

                    if 'tal_outputs' in outputs_dict:
                        tal = outputs_dict['tal_outputs']
                        action_scores = tal['action_scores']  # (B, T)
                        t_cam = tal['t_cam']  # (B, C, T)

                        wtal_fg_scores.append(action_scores.mean().item())
                        wtal_fg_sparsity.append((action_scores > 0.5).float().mean().item())

                        # T-CAM peak position distribution (closer to clip center is better)
                        peak_pos = t_cam.max(dim=1)[0].argmax(dim=1).float()  # (B,)
                        T_len = t_cam.shape[2]
                        peak_center_ratio = (peak_pos / max(T_len - 1, 1)).mean().item()
                        wtal_peak_positions.append(peak_center_ratio)
                elif use_tta:
                    outputs_normal = model(videos)
                    outputs_flipped = model(torch.flip(videos, dims=[4]))
                    outputs = (outputs_normal + outputs_flipped) * 0.5
                else:
                    outputs = model(videos)

                loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    f1 = f1_score(all_labels, all_preds, average="macro")

    # WTAL metrics
    wtal_metrics = None
    if compute_wtal_metrics and wtal_fg_scores:
        wtal_metrics = {
            'mean_fg_score': np.mean(wtal_fg_scores),
            'fg_sparsity': np.mean(wtal_fg_sparsity),
            'peak_center_ratio': np.mean(wtal_peak_positions),
        }

    return avg_loss, accuracy, f1, wtal_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMDB-51 Full Dataset Training with STCA-ViT (VideoMAE)')

    # Data
    parser.add_argument('--data_root', type=str, default='../HMDB_dataset/videos')
    parser.add_argument('--clip_length', type=int, default=16)
    parser.add_argument('--frame_stride', type=int, default=2)

    # Batch / Workers
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override auto batch size (None=auto based on VRAM)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Override auto num_workers (None=auto)')

    # Training
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--backbone_lr_ratio', type=float, default=0.01)
    parser.add_argument('--min_lr_ratio', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation', type=int, default=2)

    # Augmentation
    parser.add_argument('--use_tta', action='store_true', default=True)
    parser.add_argument('--use_mixup', action='store_true', default=True)
    parser.add_argument('--mixup_alpha', type=float, default=0.4)
    parser.add_argument('--use_cutmix', action='store_true', default=True)
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    parser.add_argument('--use_randaugment', action='store_true', default=True)
    parser.add_argument('--randaug_n', type=int, default=2)
    parser.add_argument('--randaug_m', type=int, default=9)
    parser.add_argument('--use_random_erasing', action='store_true', default=True)
    parser.add_argument('--erasing_prob', type=float, default=0.25)
    parser.add_argument('--use_temporal_aug', action='store_true', default=True)
    parser.add_argument('--speed_range_min', type=float, default=0.8)
    parser.add_argument('--speed_range_max', type=float, default=1.2)

    # Regularization
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--val_ratio', type=float, default=0.2)

    # Orthogonal Loss / GRL / Multi-Head
    parser.add_argument('--use_ortho_loss', action='store_true', default=True)
    parser.add_argument('--ortho_weight', type=float, default=0.1)
    parser.add_argument('--use_grl', action='store_true', default=True)
    parser.add_argument('--grl_lambda_init', type=float, default=0.0)
    parser.add_argument('--grl_lambda_max', type=float, default=0.1)
    parser.add_argument('--use_multi_head', action='store_true', default=True)

    # WTAL (Temporal Action Localization)
    parser.add_argument('--use_wtal', action='store_true', default=False,
                        help='Enable WTAL head (weakly-supervised temporal action localization)')
    parser.add_argument('--wtal_weight', type=float, default=0.3,
                        help='WTAL loss weight')
    parser.add_argument('--wtal_warmup', type=int, default=5,
                        help='Epoch at which WTAL loss starts (after classification stabilizes)')

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--try_number', type=int, default=1)
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume training from (Best or Trainover .pth file)')

    args = parser.parse_args()

    # ========================
    # Environment setup
    # ========================
    # expandable_segments is only supported in PyTorch 2.1+
    torch_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    if torch_version >= (2, 1):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=args.gradient_accumulation,
    )

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    accelerate_set_seed(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True

    # ========================
    # GPU info and batch size
    # ========================
    image_size = 224  # 112x112 (speed optimization; VideoMAE backbone auto-resizes to 224 internally)

    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(accelerator.local_process_index).total_memory / 1e9
        if accelerator.is_main_process:
            print(f"GPU: {torch.cuda.get_device_name(accelerator.local_process_index)} | VRAM: {total_vram:.2f} GB")
            print(f"Accelerate: {accelerator.num_processes} GPU(s) detected")

        # Auto batch size (VideoMAE ViT-B + 112x112 input + backbone 224 resize + module_embed_dim=256)
        if args.batch_size is not None:
            optimal_batch_size = args.batch_size
            val_batch_size = args.batch_size
        else:
            optimal_batch_size = 1
            val_batch_size = 2
            # if total_vram < 12:
            #     optimal_batch_size = 1
            #     val_batch_size = 2
            # elif total_vram < 18:
            #     optimal_batch_size = 2
            #     val_batch_size = 4
            # elif total_vram < 24:
            #     optimal_batch_size = 4
            #     val_batch_size = 6
            # elif total_vram < 48:
            #     optimal_batch_size = 8
            #     val_batch_size = 10
            # elif total_vram < 80:
            #     optimal_batch_size = 12
            #     val_batch_size = 16
            # else:
            #     optimal_batch_size = 20
            #     val_batch_size = 24

        if accelerator.is_main_process:
            print(f"Batch size (per GPU): Train={optimal_batch_size}, Val={val_batch_size}")
            eff_batch = optimal_batch_size * accelerator.num_processes * args.gradient_accumulation
            print(f"Total effective batch size: {eff_batch}")
    else:
        if accelerator.is_main_process:
            print("CUDA not available - using CPU")
        optimal_batch_size = 1
        val_batch_size = 1

    # Auto num_workers
    # On Windows, conservative setting to avoid memory spike from PyTorch reload during worker spawn
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        cpu_count = multiprocessing.cpu_count()
        if platform.system() == 'Windows':
            num_workers = min(4, cpu_count // 2)  # Windows: prevent page file exhaustion
        else:
            num_workers = min(16, cpu_count // 2)

    # ========================
    # Checkpoint pre-validation (before dataset load)
    # ========================
    if args.resume:
        if not os.path.exists(args.resume):
            if accelerator.is_main_process:
                print(f"\nCheckpoint file not found: {args.resume}")
                print("   Aborting. Please verify the path.")
            exit(1)

        if accelerator.is_main_process:
            print("\n" + "=" * 80)
            print("Checkpoint pre-validation (before dataset load)")
            print(f"   Path: {args.resume}")

            checkpoint_preview = torch.load(args.resume, map_location='cpu', weights_only=False)
            if isinstance(checkpoint_preview, dict):
                ckpt_epoch = checkpoint_preview.get('epoch', '?')
                ckpt_loss = checkpoint_preview.get('val_loss', checkpoint_preview.get('best_loss', '?'))
                ckpt_acc = checkpoint_preview.get('val_accuracy', checkpoint_preview.get('best_accuracy', '?'))
                has_train_state = 'train_model_state_dict' in checkpoint_preview
                has_ema_state = 'model_state_dict' in checkpoint_preview
                has_optimizer = 'optimizer_state_dict' in checkpoint_preview
                has_scheduler = 'scheduler_state_dict' in checkpoint_preview

                print(f"   Checkpoint validation successful!")
                print(f"   Epoch: {ckpt_epoch}, Val Loss: {ckpt_loss}, Val Acc: {ckpt_acc}")
                print(f"   Included states:")
                print(f"      - Train Model: {'yes' if has_train_state else 'no (EMA fallback)'}")
                print(f"      - EMA Model:   {'yes' if has_ema_state else 'no'}")
                print(f"      - Optimizer:   {'yes' if has_optimizer else 'no'}")
                print(f"      - Scheduler:   {'yes' if has_scheduler else 'no'}")

                if not has_ema_state:
                    print(f"\n   model_state_dict not found. Checkpoint may be corrupted.")
                    print("   Aborting.")
                    del checkpoint_preview
                    exit(1)
            else:
                print(f"   Invalid checkpoint format (not a dict).")
                print("   Aborting.")
                del checkpoint_preview
                exit(1)

            del checkpoint_preview
            torch.cuda.empty_cache()
            print("   Pre-validation complete — starting dataset loading.")
            print("=" * 80)

    # ========================
    # Model name / WandB
    # ========================
    model_name = f"{args.try_number}_STCA_ViT_VideoMAE_HMDB51_{args.clip_length}f_{image_size}"

    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print(f"HMDB-51 Full Dataset - STCA-ViT (VideoMAE ViT-Base)")
        print(f"   Backbone: VideoMAE ViT-Base (768-d, Kinetics pretrained)")
        print(f"   embed_dim: 768, module_embed_dim: 256, num_heads: 12")
        print(f"   Frames: {args.clip_length}, Resolution: {image_size}x{image_size}")
        print(f"   Data Split: training_*.avi (Train+Val), testing_*.avi (Test)")
        print(f"   Accelerator Device: {accelerator.device}")
        print(f"   Num Processes: {accelerator.num_processes}")
        print(f"{'='*80}")

        print(f"\nKey improvements:")
        print(f"   1. Backbone: R2Plus1D-18 -> VideoMAE ViT-Base (768-d)")
        print(f"   2. embed_dim: 768, module_embed_dim: 256 (lightweight auxiliary modules)")
        print(f"   3. Resolution: {image_size}x{image_size} (backbone auto-resizes to 224 internally)")
        print(f"   4. Backbone freeze: 6/12 layers + gradient checkpointing")
        print(f"   5. Backbone LR: x{args.backbone_lr_ratio}")
        print(f"   6. Augmentation: RandAugment + CutMix + Random Erasing + Temporal Aug")

        wandb.init(
            project="HMDB51_STCA_ViT_VideoMAE",
            name=model_name,
            config={
                "version": "stca_vit_videomae",
                "model": "STCA-ViT (VideoMAE ViT-Base)",
                "backbone": "VideoMAE ViT-Base (MCG-NJU/videomae-base-finetuned-kinetics)",
                "embed_dim": 768,
                "module_embed_dim": 256,
                "num_heads": 12,
                "dataset": "HMDB-51",
                "num_classes": 51,
                "image_size": image_size,
                "frames_per_clip": args.clip_length,
                "frame_stride": args.frame_stride,
                "learning_rate": args.learning_rate,
                "backbone_lr_ratio": args.backbone_lr_ratio,
                "weight_decay": args.weight_decay,
                "batch_size": optimal_batch_size,
                "num_epochs": args.num_epochs,
                "warmup_epochs": args.warmup_epochs,
                "gradient_accumulation": args.gradient_accumulation,
                "dropout_rate": args.dropout,
                "seed": args.seed,
                "freeze_backbone_layers": 6,
                "gradient_checkpointing": True,
                "use_wtal": args.use_wtal,
                "wtal_weight": args.wtal_weight if args.use_wtal else 0,
                "wtal_warmup": args.wtal_warmup if args.use_wtal else 0,
            }
        )

    # ========================
    # Datasets
    # ========================
    device = accelerator.device
    speed_range = (args.speed_range_min, args.speed_range_max)

    with accelerator.main_process_first():
        train_dataset = HMDB51FullDataset(
            data_root=args.data_root, split='train',
            clip_length=args.clip_length, frame_stride=args.frame_stride,
            w_size=image_size, h_size=image_size,
            val_ratio=args.val_ratio, seed=args.seed, random_flip=True,
            use_randaugment=args.use_randaugment, randaug_n=args.randaug_n, randaug_m=args.randaug_m,
            use_random_erasing=args.use_random_erasing, erasing_prob=args.erasing_prob,
            use_temporal_aug=args.use_temporal_aug, speed_range=speed_range
        )
        val_dataset = HMDB51FullDataset(
            data_root=args.data_root, split='val',
            clip_length=args.clip_length, frame_stride=args.frame_stride,
            w_size=image_size, h_size=image_size,
            val_ratio=args.val_ratio, seed=args.seed, random_flip=False
        )
        test_dataset = HMDB51FullDataset(
            data_root=args.data_root, split='test',
            clip_length=args.clip_length, frame_stride=args.frame_stride,
            w_size=image_size, h_size=image_size,
            val_ratio=args.val_ratio, seed=args.seed, random_flip=False
        )

    # DataLoaders (prefetch_factor/persistent_workers only when num_workers>0)
    loader_extra = {}
    if num_workers > 0:
        loader_extra = {'prefetch_factor': 3, 'persistent_workers': True}

    train_loader = DataLoader(
        train_dataset, batch_size=optimal_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        **loader_extra
    )
    val_loader_extra = {}
    if num_workers > 0:
        val_loader_extra = {'prefetch_factor': 2, 'persistent_workers': True}

    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        **val_loader_extra
    )
    test_loader = DataLoader(
        test_dataset, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        **val_loader_extra
    )

    # ========================
    # Model
    # ========================
    num_classes = len(train_dataset.classes)
    if accelerator.is_main_process:
        print(f"\nNumber of classes: {num_classes}")
        print(f"Initializing STCA-ViT (VideoMAE ViT-Base) model...")

    model = stca_vit_v2_5_parallel(
        num_classes=num_classes,
        dropout=args.dropout,
        module_embed_dim=256,              # Auxiliary modules use 256-d (~75% memory reduction; fusion projects back to 768)
        use_gradient_reversal=args.use_grl,
        grl_lambda=args.grl_lambda_init,
        use_multi_head_ensemble=args.use_multi_head,
        freeze_backbone_layers=6,          # Freeze 6 of 12 layers (speed + memory savings)
        use_gradient_checkpointing=True,   # Memory savings -> larger batch size
        use_wtal=args.use_wtal             # WTAL head (temporal action localization)
    ).to(device)

    model_size = calculate_model_size(model)
    if accelerator.is_main_process:
        print(f"Model size: {model_size:.2f} MB")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1e6:.2f} M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")

    # ========================
    # Improvement features
    # ========================
    if accelerator.is_main_process:
        print(f"\nEnabled features:")
        print(f"   - Gradient Reversal: {'ON' if args.use_grl else 'OFF'} (lambda: {args.grl_lambda_init} -> {args.grl_lambda_max})")
        print(f"   - Orthogonal Loss: {'ON' if args.use_ortho_loss else 'OFF'} (weight: {args.ortho_weight})")
        print(f"   - Multi-Head Ensemble: {'ON' if args.use_multi_head else 'OFF'}")
        print(f"   - WTAL Head: {'ON' if args.use_wtal else 'OFF'}"
              f"{f' (weight: {args.wtal_weight}, warmup: {args.wtal_warmup} epochs)' if args.use_wtal else ''}")

    # ========================
    # EMA
    # ========================
    model_ema = ModelEMA(model, decay=args.ema_decay, device=device)
    if accelerator.is_main_process:
        print("EMA model initialized")

    # ========================
    # Optimizer (Layer-wise LR)
    # ========================
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay

    backbone_lr = learning_rate * args.backbone_lr_ratio  # 5e-7
    module_lr = learning_rate * 1.0                        # 5e-5
    fusion_lr = learning_rate * 0.8                        # 4e-5
    head_lr = learning_rate * 1.5                          # 7.5e-5

    backbone_params = list(model.backbone.parameters())
    cross_attn_params = list(model.stca_module.parameters())
    mgu_params = list(model.mgu_module.parameters())
    person_params = list(model.person_module.parameters()) if hasattr(model, 'person_module') else []
    fusion_params = list(model.fusion.parameters())
    classifier_params = list(model.classifier.parameters())
    wtal_params = list(model.tal_head.parameters()) if args.use_wtal else []

    wtal_lr = learning_rate * 2.0  # Higher LR for WTAL head (faster convergence)

    if accelerator.is_main_process:
        print(f"\nLayer-wise LR:")
        print(f"   - VideoMAE Backbone: {backbone_lr:.2e} (x{args.backbone_lr_ratio})")
        print(f"   - Cross-Attention/MGU/Person Modules: {module_lr:.2e}")
        print(f"   - Fusion: {fusion_lr:.2e}")
        print(f"   - Classification Heads: {head_lr:.2e}")
        if args.use_wtal:
            print(f"   - WTAL Head: {wtal_lr:.2e} (x2.0)")

    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': cross_attn_params, 'lr': module_lr, 'weight_decay': weight_decay},
        {'params': mgu_params, 'lr': module_lr, 'weight_decay': weight_decay},
        {'params': fusion_params, 'lr': fusion_lr, 'weight_decay': weight_decay},
        {'params': classifier_params, 'lr': head_lr, 'weight_decay': weight_decay * 2.0},
    ]
    if person_params:
        param_groups.append({'params': person_params, 'lr': module_lr, 'weight_decay': weight_decay})
    if wtal_params:
        param_groups.append({'params': wtal_params, 'lr': wtal_lr, 'weight_decay': weight_decay})

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-7)

    # Scheduler: Warmup + Cosine Annealing
    def warmup_cosine(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)
        return max(args.min_lr_ratio, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)
    if accelerator.is_main_process:
        print(f"Scheduler: Warmup ({args.warmup_epochs} epochs) + Cosine Annealing (min_lr_ratio={args.min_lr_ratio})")

    # ========================
    # Loss
    # ========================
    train_labels = [s[1] for s in train_dataset.samples]
    if args.use_class_weights:
        class_counts = Counter(train_labels)
        class_weights = {cls: np.sqrt(sum(class_counts.values()) / count) for cls, count in class_counts.items()}
        weights = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())], dtype=torch.float32).to(device)
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing, weight=weights)
        if accelerator.is_main_process:
            print(f"Loss: LabelSmoothing({args.label_smoothing}) with class weights")
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing, weight=None)
        if accelerator.is_main_process:
            print(f"Loss: LabelSmoothing({args.label_smoothing})")

    val_criterion = nn.CrossEntropyLoss()

    # ========================
    # Prepare with Accelerator
    # ========================
    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler
    )

    # ========================
    # Training loop
    # ========================
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    start_epoch = 0
    best_model_path = f"{args.try_number}_STCA_ViT_HMDB51_{args.clip_length}f_{image_size}_Best.pth"
    trained_model_path = f"{args.try_number}_STCA_ViT_HMDB51_{args.clip_length}f_{image_size}_Trainover.pth"

    # ========================
    # Resume from checkpoint
    # ========================
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if accelerator.is_main_process:
                print(f"\nResuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            # Load model (train_model_state_dict takes priority, falls back to model_state_dict)
            unwrapped_model = accelerator.unwrap_model(model)
            if 'train_model_state_dict' in checkpoint:
                unwrapped_model.load_state_dict(checkpoint['train_model_state_dict'])
            else:
                unwrapped_model.load_state_dict(checkpoint['model_state_dict'])

            # Load EMA model
            if 'ema_state_dict' in checkpoint:
                model_ema.module.load_state_dict(checkpoint['ema_state_dict'])
            elif 'model_state_dict' in checkpoint:
                model_ema.module.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    if accelerator.is_main_process:
                        print(f"  Optimizer load failed (ignoring): {e}")

            # Load scheduler
            if 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    if accelerator.is_main_process:
                        print(f"  Scheduler load failed (ignoring): {e}")

            # Restore training state
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
            best_val_acc = checkpoint.get('best_val_acc', checkpoint.get('val_acc', 0.0))
            patience_counter = checkpoint.get('patience_counter', 0)

            if accelerator.is_main_process:
                print(f"  Resuming from epoch {start_epoch}")
                print(f"  Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.2f}%")
                print(f"  Patience counter: {patience_counter}")

            del checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        else:
            if accelerator.is_main_process:
                print(f"\nWarning: checkpoint file not found: {args.resume}")
                print(f"Starting training from scratch.")

    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        if start_epoch > 0:
            print(f"Resuming training (Epoch {start_epoch + 1} ~ {args.num_epochs})")
        else:
            print(f"Starting training (total {args.num_epochs} epochs)")
        print(f"{'='*80}\n")

    for epoch in range(start_epoch, args.num_epochs):
        if accelerator.is_main_process:
            print(f"=== Epoch {epoch+1}/{args.num_epochs} ===")

        # GRL Lambda scheduling (gradual increase)
        progress = epoch / args.num_epochs
        grl_lambda = args.grl_lambda_init + (args.grl_lambda_max - args.grl_lambda_init) * progress
        unwrapped_model = accelerator.unwrap_model(model)
        if args.use_grl and hasattr(unwrapped_model, 'set_grl_lambda'):
            unwrapped_model.set_grl_lambda(grl_lambda)

        train_loss, train_acc, ortho_loss, wtal_loss = train_epoch(
            model, train_loader, criterion, optimizer, accelerator, device,
            model_ema=model_ema, max_grad_norm=args.max_grad_norm,
            use_aux_loss=True, use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha,
            use_cutmix=args.use_cutmix, cutmix_alpha=args.cutmix_alpha,
            use_ortho_loss=args.use_ortho_loss, ortho_weight=args.ortho_weight,
            use_wtal=args.use_wtal, wtal_weight=args.wtal_weight,
            current_epoch=epoch, wtal_warmup=args.wtal_warmup
        )

        # Collect WTAL metrics only when WTAL is enabled and warmup is complete
        wtal_active = args.use_wtal and (epoch >= args.wtal_warmup)
        val_loss, val_acc, val_f1, wtal_metrics = validate_epoch(
            model_ema.module, val_loader, val_criterion, accelerator,
            use_tta=args.use_tta, compute_wtal_metrics=wtal_active
        )

        current_lr = optimizer.param_groups[0]['lr']
        if accelerator.is_main_process:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            if ortho_loss > 0:
                print(f"Ortho Loss: {ortho_loss:.4f}, GRL lambda: {grl_lambda:.4f}")
            if wtal_loss > 0:
                print(f"WTAL Loss: {wtal_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
            if wtal_metrics is not None:
                print(f"WTAL Metrics: fg_score={wtal_metrics['mean_fg_score']:.3f}, "
                      f"sparsity={wtal_metrics['fg_sparsity']:.3f}, "
                      f"peak_center={wtal_metrics['peak_center_ratio']:.3f}")
            print(f"Learning Rate: {current_lr:.2e}")

            # Contribution weights logging
            if hasattr(unwrapped_model, 'get_contribution_weights'):
                contrib_weights = unwrapped_model.get_contribution_weights()
                if isinstance(contrib_weights, dict):
                    print(f"Contribution: alpha={contrib_weights.get('alpha (stca)', contrib_weights.get('alpha (cross)', 0)):.3f}, "
                          f"beta={contrib_weights.get('beta (mgu)', 0):.3f}, "
                          f"gamma={contrib_weights.get('gamma (person)', 0):.3f}")

            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'ortho_loss': ortho_loss,
                'grl_lambda': grl_lambda,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_f1': val_f1,
                'learning_rate': current_lr
            }
            if args.use_wtal:
                log_dict['wtal_loss'] = wtal_loss
            if wtal_metrics is not None:
                log_dict['wtal_fg_score'] = wtal_metrics['mean_fg_score']
                log_dict['wtal_fg_sparsity'] = wtal_metrics['fg_sparsity']
                log_dict['wtal_peak_center'] = wtal_metrics['peak_center_ratio']
            wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model_ema.module)
                unwrapped_train_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'train_model_state_dict': unwrapped_train_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'ema_state_dict': model_ema.module.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_acc': best_val_acc,
                    'patience_counter': patience_counter,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                }, best_model_path)
                print(f"Best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if accelerator.is_main_process:
                print(f"Patience: {patience_counter}/{args.patience}")

        scheduler.step()

        if patience_counter >= args.patience:
            if accelerator.is_main_process:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

        if accelerator.is_main_process:
            print()

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ========================
    # Save final model
    # ========================
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': args.num_epochs,
            'model_state_dict': unwrapped_model.state_dict(),
            'ema_state_dict': model_ema.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
        }, trained_model_path)
        print(f"Final model saved: {trained_model_path}")

    # ========================
    # Final Test
    # ========================
    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print(f"Final Test Set Evaluation")
        print(f"{'='*80}\n")

        checkpoint = torch.load(best_model_path, map_location=device)
        model_ema.module.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc, test_f1, _ = validate_epoch(
            model_ema.module, test_loader, val_criterion, accelerator, use_tta=args.use_tta
        )

        print(f"Final Test Results:")
        print(f"   Test Accuracy: {test_acc:.2f}%")
        print(f"   Test F1 Score: {test_f1:.4f}")

        wandb.log({
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_acc,
            'final_test_accuracy': test_acc,
            'final_test_f1': test_f1
        })

        wandb.finish()

        print(f"\n{'='*80}")
        print(f"Training complete!")
        print(f"   Best Val Acc: {best_val_acc:.2f}%")
        print(f"   Final Test Acc: {test_acc:.2f}%")
        print(f"{'='*80}")
