"""
STCA-ViT Training for Kinetics-400 Dataset
===========================================

Key improvements over the previous version:
-------------------------------------------
1. Backbone: R2Plus1D-18 -> VideoMAE ViT-Base (Kinetics pretrained)
2. embed_dim: 192 -> 768
3. Training settings: 30 epochs, weight_decay=0.05, lr=5e-5, backbone_lr_ratio=0.01

Data loading / split / Augmentation:
- Kinetics-400 standard train/val/test split
- decord-based fast video loading
- RandAugment + CutMix + Random Erasing + Temporal Aug
- Multi-clip inference (3-clip + TTA)
- EMA + Accelerate + Mixed Precision

Model input format:
- 16 frames: (Batch, 3, 16, 224, 224)
"""

import os
import random
import multiprocessing
import psutil
import cv2
import numpy as np
import torch
import gc
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from collections import Counter, OrderedDict
import torch.backends.cudnn as cudnn
from pathlib import Path
import torch.nn.functional as F
import decord
decord.bridge.set_bridge('torch')

from thop import profile

# Training utilities
import wandb
from tqdm import tqdm

# Accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

# STCA-ViT model import
import sys
sys.path.insert(0, '/home/user/data/codes')
from model import stca_vit_v2_5_parallel

# Suppress OpenCV/FFmpeg error messages
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

# Debug Mode
DEBUG_MODE = False

def debug_shape(x, layer_name):
    if DEBUG_MODE:
        if isinstance(x, torch.Tensor):
            print(f"[DEBUG] {layer_name} Shape: {x.shape}, Device: {x.device}, Dtype: {x.dtype}")
        else:
            print(f"[DEBUG] {layer_name} Type: {type(x)}")
    return x

def calculate_model_size(model):
    param_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4
    return param_size / (1024 * 1024)


# ============================================================================
# RandAugment for Video (Spatial)
# ============================================================================
class VideoRandAugment:
    """RandAugment for video - applies the same augmentation to each frame"""
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
        factor = max(factor, 0.1)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        result = cv2.addWeighted(img, factor, gray_3ch, 1 - factor, 0)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _adjust_contrast(self, img, factor):
        factor = max(factor, 0.1)
        mean = img.mean()
        result = mean + factor * (img.astype(np.float32) - mean)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _adjust_brightness(self, img, factor):
        factor = max(factor, 0.1)
        result = img.astype(np.float32) * factor
        return np.clip(result, 0, 255).astype(np.uint8)

    def _adjust_sharpness(self, img, factor):
        factor = max(factor, 0.1)
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        result = cv2.addWeighted(img, factor, blurred, 1 - factor, 0)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _shear(self, img, magnitude, axis='x'):
        h, w = img.shape[:2]
        if axis == 'x':
            M = np.array([[1, magnitude, 0], [0, 1, 0]], dtype=np.float32)
        else:
            M = np.array([[1, 0, 0], [magnitude, 1, 0]], dtype=np.float32)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _translate(self, img, magnitude, axis='x'):
        h, w = img.shape[:2]
        if axis == 'x':
            tx = int(w * magnitude)
            M = np.array([[1, 0, tx], [0, 1, 0]], dtype=np.float32)
        else:
            ty = int(h * magnitude)
            M = np.array([[1, 0, 0], [0, 1, ty]], dtype=np.float32)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


# ============================================================================
# Random Erasing for Video
# ============================================================================
class VideoRandomErasing:
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.43216, 0.394666, 0.37645)):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean

    def __call__(self, clip):
        if random.random() > self.probability:
            return clip
        C, T, H, W = clip.shape
        area = H * W
        for _ in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            if h < H and w < W:
                x1 = random.randint(0, W - w)
                y1 = random.randint(0, H - h)
                for c in range(C):
                    clip[c, :, y1:y1+h, x1:x1+w] = float(self.mean[c])
                return clip
        return clip


# ============================================================================
# Dataset (decord-based)
# ============================================================================
class EnhancedKinetics400Dataset(Dataset):
    KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
    KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

    def __init__(self, video_paths, labels, clip_length=16,
                 temporal_stride=4, w_size=224, h_size=224,
                 is_training=False, random_flip=True,
                 use_randaugment=True, randaug_n=2, randaug_m=9,
                 use_random_erasing=True, erasing_prob=0.25,
                 use_temporal_aug=True, speed_range=(0.8, 1.2),
                 num_clips=1, crop_mode='center'):
        self.clip_length = clip_length
        self.temporal_stride = temporal_stride
        self.w_size = w_size
        self.h_size = h_size
        self.is_training = is_training
        self.random_flip = random_flip
        self.use_temporal_aug = use_temporal_aug and is_training
        self.speed_range = speed_range
        self.num_clips = num_clips
        self.crop_mode = crop_mode

        self.randaugment = VideoRandAugment(n=randaug_n, m=randaug_m) if use_randaugment and is_training else None
        self.random_erasing = VideoRandomErasing(probability=erasing_prob) if use_random_erasing and is_training else None

        self.samples = []
        print(f"Initializing dataset (T={clip_length}, stride={temporal_stride}, size={h_size}x{w_size})...")
        if is_training:
            aug_info = []
            if use_randaugment: aug_info.append(f"RandAugment(n={randaug_n}, m={randaug_m})")
            if use_random_erasing: aug_info.append(f"RandomErasing(p={erasing_prob})")
            if use_temporal_aug: aug_info.append(f"TemporalAug(speed={speed_range})")
            if aug_info: print(f"   Augmentations: {', '.join(aug_info)}")

        min_required_frames = int(clip_length * temporal_stride * 0.75)
        skipped = 0
        corrupted = 0

        # Suppress decord C++ error messages
        import sys
        stderr_fd = sys.stderr.fileno()
        old_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)

        try:
            for video_path, label in zip(video_paths, labels):
                if not os.path.exists(video_path):
                    continue
                try:
                    file_size = os.path.getsize(video_path)
                    if file_size < 100 * 1024:
                        corrupted += 1
                        continue
                    vr = decord.VideoReader(video_path, num_threads=1)
                    total_frames = len(vr)
                    del vr
                    if total_frames <= 0 or total_frames > 100000:
                        corrupted += 1
                        continue
                    if total_frames >= min_required_frames:
                        self.samples.append((video_path, 0, label, total_frames))
                    else:
                        skipped += 1
                except Exception:
                    corrupted += 1
                    continue
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)

        print(f"Dataset created: {len(self.samples)} videos")
        if skipped > 0: print(f"   Videos skipped (insufficient frames): {skipped}")
        if corrupted > 0: print(f"   Videos corrupted/unreadable: {corrupted}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, _, label, total_frames = self.samples[idx]
        if self.num_clips > 1 and not self.is_training:
            clips = []
            for clip_idx in range(self.num_clips):
                clip = self._load_clip(video_path, total_frames, clip_idx)
                clips.append(clip)
            return torch.stack(clips), label
        else:
            clip = self._load_clip(video_path, total_frames, clip_idx=0)
            return clip, label

    def _load_clip(self, video_path, total_frames, clip_idx=0):
        try:
            vr = decord.VideoReader(video_path, num_threads=1)
            actual_total = len(vr)
            if actual_total == 0:
                raise ValueError(f"Empty video: {video_path}")

            effective_stride = self.temporal_stride
            if self.use_temporal_aug:
                speed_factor = random.uniform(*self.speed_range)
                effective_stride = max(1, int(self.temporal_stride * speed_factor))

            clip_span = self.clip_length * effective_stride

            if actual_total >= clip_span:
                if self.is_training:
                    max_start = actual_total - clip_span
                    start_frame = np.random.randint(0, max(1, max_start))
                else:
                    if self.num_clips > 1:
                        segment_size = (actual_total - clip_span) // max(1, self.num_clips - 1)
                        start_frame = clip_idx * segment_size
                        start_frame = min(start_frame, actual_total - clip_span)
                    else:
                        start_frame = (actual_total - clip_span) // 2
                frame_indices = list(range(start_frame, start_frame + clip_span, effective_stride))[:self.clip_length]
            else:
                effective_stride = max(1, actual_total // self.clip_length)
                frame_indices = list(range(0, actual_total, effective_stride))[:self.clip_length]

            frames_tensor = vr.get_batch(frame_indices)
            frames_np = frames_tensor.numpy()
            del vr

            do_flip = self.is_training and self.random_flip and random.random() < 0.5
            use_random_crop = self.is_training and self.crop_mode == 'random'
            new_w, new_h = None, None
            crop_x, crop_y = 0, 0

            if use_random_crop:
                orig_h, orig_w = frames_np.shape[1], frames_np.shape[2]
                scale = random.randint(256, 320)
                if orig_w < orig_h:
                    new_w = scale
                    new_h = int(orig_h * scale / orig_w)
                else:
                    new_h = scale
                    new_w = int(orig_w * scale / orig_h)
                crop_x = random.randint(0, max(0, new_w - self.w_size))
                crop_y = random.randint(0, max(0, new_h - self.h_size))

            frames_list = []
            for fi in range(frames_np.shape[0]):
                frame = frames_np[fi]
                if use_random_crop and new_w is not None:
                    frame = cv2.resize(frame, (new_w, new_h))
                    frame = frame[crop_y:crop_y+self.h_size, crop_x:crop_x+self.w_size]
                    if frame.shape[0] != self.h_size or frame.shape[1] != self.w_size:
                        frame = cv2.resize(frame, (self.w_size, self.h_size))
                else:
                    h, w = frame.shape[:2]
                    if w < h:
                        new_w_temp = 256
                        new_h_temp = int(h * 256 / w)
                    else:
                        new_h_temp = 256
                        new_w_temp = int(w * 256 / h)
                    frame = cv2.resize(frame, (new_w_temp, new_h_temp))
                    start_y = (new_h_temp - self.h_size) // 2
                    start_x = (new_w_temp - self.w_size) // 2
                    frame = frame[start_y:start_y+self.h_size, start_x:start_x+self.w_size]
                    if frame.shape[0] != self.h_size or frame.shape[1] != self.w_size:
                        frame = cv2.resize(frame, (self.w_size, self.h_size))
                if do_flip:
                    frame = cv2.flip(frame, 1)
                frames_list.append(frame)

            while len(frames_list) < self.clip_length:
                if len(frames_list) > 0:
                    frames_list.append(frames_list[-1].copy())
                else:
                    frames_list.append(np.zeros((self.h_size, self.w_size, 3), dtype=np.uint8))

            if self.randaugment is not None:
                frames_list = self.randaugment(frames_list)

            normalized_frames = []
            for frame in frames_list[:self.clip_length]:
                if frame.dtype != np.float32:
                    frame = frame.astype(np.float32) / 255.0
                else:
                    frame = frame / 255.0
                frame = (frame - self.KINETICS_MEAN) / self.KINETICS_STD
                normalized_frames.append(frame)

            clip = np.stack(normalized_frames, axis=0).transpose(3, 0, 1, 2)
            clip = torch.from_numpy(clip).float()

            if self.random_erasing is not None:
                clip = self.random_erasing(clip)

            return clip

        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return torch.zeros(3, self.clip_length, self.h_size, self.w_size, dtype=torch.float32)


# ============================================================================
# CutMix / MixUp
# ============================================================================
def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    _, _, T, H, W = x.shape
    cut_rat = np.sqrt(1. - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    mixed_x = x.clone()
    mixed_x[:, :, :, bby1:bby2, bbx1:bbx2] = x[index, :, :, bby1:bby2, bbx1:bbx2]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


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


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# Label Smoothing Loss
# ============================================================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ============================================================================
# EMA
# ============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        from copy import deepcopy
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


# ============================================================================
# Loss weights
# ============================================================================
AUX_LOSS_WEIGHT = 0.3
ORTHO_LOSS_WEIGHT = 0.1
GRL_LAMBDA_INIT = 0.0
GRL_LAMBDA_MAX = 0.1
WTAL_LOSS_WEIGHT = 0.3
WTAL_WARMUP_EPOCHS = 5  # Start WTAL after classification stabilizes


# ============================================================================
# Training function
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, accelerator, device,
                model_ema=None, max_grad_norm=1.0, use_aux_loss=True,
                use_mixup=True, mixup_alpha=0.4, use_cutmix=True, cutmix_alpha=1.0,
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

    # WTAL is only active after warmup + gradual ramp-up
    wtal_active = use_wtal and (current_epoch >= wtal_warmup)
    if wtal_active:
        ramp_progress = min(1.0, (current_epoch - wtal_warmup) / 5.0)
        effective_wtal_weight = wtal_weight * ramp_progress
    else:
        effective_wtal_weight = 0.0

    train_bar = tqdm(train_loader, desc="Training", leave=False, disable=not accelerator.is_main_process)

    for i, (videos, labels) in enumerate(train_bar):
        aug_choice = random.random()
        if aug_choice < 0.33 and use_cutmix:
            videos, labels_a, labels_b, lam = cutmix_data(videos, labels, cutmix_alpha)
            mixed = True
        elif aug_choice < 0.66 and use_mixup:
            videos, labels_a, labels_b, lam = mixup_data(videos, labels, mixup_alpha)
            mixed = True
        else:
            labels_a, labels_b, lam = labels, labels, 1.0
            mixed = False

        with accelerator.accumulate(model):
            with accelerator.autocast():
                if use_aux_loss:
                    outputs_dict = model(videos, return_all=True,
                                         compute_ortho_loss=use_ortho_loss,
                                         compute_wtal=wtal_active)
                    final_logits = outputs_dict['final']

                    if torch.isnan(final_logits).any():
                        nan_count += 1
                        continue

                    if mixed:
                        main_loss = mixup_criterion(criterion, final_logits, labels_a, labels_b, lam)
                    else:
                        main_loss = criterion(final_logits, labels)

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
                            skip_mil=mixed
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
                        continue

                    if mixed:
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = criterion(outputs, labels)

                    if use_ortho_loss and ortho_loss is not None:
                        loss = loss + ortho_weight * ortho_loss
                        running_ortho_loss += ortho_loss.item()

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
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

    valid_batches = len(train_loader) - nan_count
    if valid_batches > 0:
        avg_loss = running_loss / valid_batches
        avg_ortho_loss = running_ortho_loss / valid_batches if running_ortho_loss > 0 else 0
        avg_wtal_loss = running_wtal_loss / valid_batches if running_wtal_loss > 0 else 0
        acc = 100 * correct / total if total > 0 else 0
        return avg_loss, acc, avg_ortho_loss, avg_wtal_loss
    else:
        return float('inf'), 0, 0, 0


# ============================================================================
# Multi-clip Validation
# ============================================================================
def validate_epoch_multiclip(model, val_loader, criterion, device, accelerator=None,
                             use_tta=False, num_clips=1, compute_wtal_metrics=False):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # WTAL proxy metrics
    wtal_fg_scores = []
    wtal_fg_sparsity = []
    wtal_peak_positions = []

    is_main = accelerator is None or accelerator.is_main_process

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Validation", leave=False, disable=not is_main)
        for videos, labels in val_bar:
            if accelerator is None:
                videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if videos.dim() == 6:
                B, num_clips_dim, C, T, H, W = videos.shape
                all_outputs = []
                for clip_idx in range(num_clips_dim):
                    clip = videos[:, clip_idx]
                    if accelerator is not None:
                        with accelerator.autocast():
                            if compute_wtal_metrics and clip_idx == 0:
                                # Collect WTAL metrics from the first clip only (to save memory)
                                out_dict = model(clip, return_all=True, compute_wtal=True)
                                out = out_dict['final']
                                if 'tal_outputs' in out_dict:
                                    tal = out_dict['tal_outputs']
                                    action_scores = tal['action_scores']  # (B, T)
                                    t_cam = tal['t_cam']  # (B, C, T)
                                    wtal_fg_scores.append(action_scores.mean().item())
                                    wtal_fg_sparsity.append((action_scores > 0.5).float().mean().item())
                                    peak_pos = t_cam.max(dim=1)[0].argmax(dim=1).float()
                                    T_len = t_cam.shape[2]
                                    peak_center_ratio = (peak_pos / max(T_len - 1, 1)).mean().item()
                                    wtal_peak_positions.append(peak_center_ratio)
                            elif use_tta:
                                out_normal = model(clip)
                                out_flipped = model(torch.flip(clip, dims=[4]))
                                out = (out_normal + out_flipped) * 0.5
                            else:
                                out = model(clip)
                    else:
                        with autocast('cuda'):
                            if use_tta:
                                out_normal = model(clip)
                                out_flipped = model(torch.flip(clip, dims=[4]))
                                out = (out_normal + out_flipped) * 0.5
                            else:
                                out = model(clip)
                    all_outputs.append(out)
                outputs = torch.stack(all_outputs).mean(dim=0)
            else:
                if accelerator is not None:
                    with accelerator.autocast():
                        if compute_wtal_metrics:
                            out_dict = model(videos, return_all=True, compute_wtal=True)
                            outputs = out_dict['final']
                            if 'tal_outputs' in out_dict:
                                tal = out_dict['tal_outputs']
                                action_scores = tal['action_scores']
                                t_cam = tal['t_cam']
                                wtal_fg_scores.append(action_scores.mean().item())
                                wtal_fg_sparsity.append((action_scores > 0.5).float().mean().item())
                                peak_pos = t_cam.max(dim=1)[0].argmax(dim=1).float()
                                T_len = t_cam.shape[2]
                                peak_center_ratio = (peak_pos / max(T_len - 1, 1)).mean().item()
                                wtal_peak_positions.append(peak_center_ratio)
                        elif use_tta:
                            outputs_normal = model(videos)
                            outputs_flipped = model(torch.flip(videos, dims=[4]))
                            outputs = (outputs_normal + outputs_flipped) * 0.5
                        else:
                            outputs = model(videos)
                else:
                    with autocast('cuda'):
                        if use_tta:
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


def load_pretrained_model_for_continue_training(model, optimizer, scheduler, scaler, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found. Starting training from scratch.")
        return 0, float('inf'), 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        cleaned_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if 'total_ops' not in key and 'total_params' not in key:
                cleaned_state_dict[key] = value

        model.load_state_dict(cleaned_state_dict, strict=False)

        start_epoch = 0
        best_val_loss = float('inf')
        best_accuracy = 0

        if isinstance(checkpoint, dict):
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('val_loss', checkpoint.get('best_loss', float('inf')))
            best_accuracy = checkpoint.get('val_accuracy', checkpoint.get('best_accuracy', 0))
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                try: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except: pass
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                try: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except: pass

        print(f"Checkpoint loaded (Epoch {start_epoch}, Loss {best_val_loss:.4f}, Acc {best_accuracy:.2f}%)")
        return start_epoch, best_val_loss, best_accuracy

    except Exception as e:
        print(f"Checkpoint load failed: {e}")
        return 0, float('inf'), 0


# ============================================================================
# Main Training Script
# ============================================================================
if __name__ == "__main__":
    # ========================
    # Initialize Accelerator
    # ========================
    # Pre-check WTAL flag before Accelerator init (to determine batch/accumulation)
    import argparse as _argparse_early
    _early_parser = _argparse_early.ArgumentParser(add_help=False)
    _early_parser.add_argument('--use_wtal', action='store_true', default=False)
    _early_wtal, _ = _early_parser.parse_known_args()
    _grad_accum = 6 if _early_wtal.use_wtal else 4  # Increase accumulation when WTAL is on (compensate for smaller batch)

    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=_grad_accum,
    )

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    if accelerator.is_main_process:
        torch.cuda.empty_cache()
    gc.collect()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    import resource
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, 4096), hard))
    except Exception:
        pass

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # GPU info and batch size adjustment (based on VideoMAE ViT-B)
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(accelerator.local_process_index).total_memory / 1e9
        if accelerator.is_main_process:
            print(f"GPU: {torch.cuda.get_device_name(accelerator.local_process_index)} | VRAM: {total_vram:.2f} GB")
            print(f"Accelerate: {accelerator.num_processes} GPU(s) detected")

        # VideoMAE ViT-B + 3 modules have high memory usage
        # WTAL activation preserves temporal features, increasing memory -> reduce batch size
        # optimal_batch_size = 2 if _early_wtal.use_wtal else 3
        # val_batch_size = 3
        if total_vram < 16:
            optimal_batch_size = 1
            val_batch_size = 1
        elif total_vram < 24:
            optimal_batch_size = 1 if _early_wtal.use_wtal else 2
            val_batch_size = 2
        elif total_vram < 48:
            optimal_batch_size = 2 if _early_wtal.use_wtal else 3
            val_batch_size = 3
        elif total_vram < 80:
            optimal_batch_size = 3 if _early_wtal.use_wtal else 4
            val_batch_size = 4
        else:
            optimal_batch_size = 4 if _early_wtal.use_wtal else 6
            val_batch_size = 6

        if accelerator.is_main_process:
            print(f"Batch size (per GPU): Train={optimal_batch_size}, Val={val_batch_size}")
            print(f"Gradient Accumulation: {_grad_accum} steps" + (" (WTAL memory compensation)" if _early_wtal.use_wtal else ""))
            print(f"Total effective batch size: Train={optimal_batch_size * accelerator.num_processes * _grad_accum}")
    else:
        if accelerator.is_main_process:
            print("CUDA not available - using CPU")
        optimal_batch_size = 1
        val_batch_size = 1

    # ========================
    # Seed
    # ========================
    seedValue = 42
    accelerate_set_seed(seedValue)

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(seedValue)

    # ========================
    # Experiment configuration
    # ========================
    try_number = 8
    frame_number = 16
    temporal_stride = 4

    # Resolution
    w_size = 224
    h_size = 224

    # Backbone LR (VideoMAE is already well-pretrained, so use a very small value)
    backbone_lr_ratio = 0.01

    # Augmentation settings
    use_randaugment = True
    randaug_n = 2
    randaug_m = 9
    use_random_erasing = True
    erasing_prob = 0.25
    use_temporal_aug = True
    speed_range = (0.8, 1.2)
    use_cutmix = True
    cutmix_alpha = 1.0

    # Multi-clip
    num_val_clips = 3

    # Hyperparameters
    dropout_rate = 0.3
    learning_rate = 1e-4
    weight_decay = 0.05   # ViT standard
    warmup_epochs = 5
    label_smoothing = 0.1
    num_epochs = 30
    max_grad_norm = 1.0
    min_lr_ratio = 0.01
    use_mixup = True
    mixup_alpha = 0.4

    use_ortho_loss = True
    ortho_weight = 0.1
    use_grl = True
    grl_lambda_init = 0.0
    grl_lambda_max = 0.1
    use_multi_head = True

    # WTAL (Temporal Action Localization)
    use_wtal = False  # Activate with --use_wtal flag (default OFF)
    wtal_weight = 0.3
    wtal_warmup = 5  # Epoch at which WTAL loss starts

    # Handle CLI --use_wtal flag
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument('--use_wtal', action='store_true', default=False)
    _parser.add_argument('--wtal_weight', type=float, default=0.3)
    _parser.add_argument('--wtal_warmup', type=int, default=5)
    _parser.add_argument('--resume', type=str, default=None, help='Checkpoint path for explicit resume')
    _wtal_args, _ = _parser.parse_known_args()
    use_wtal = _wtal_args.use_wtal
    wtal_weight = _wtal_args.wtal_weight
    wtal_warmup = _wtal_args.wtal_warmup
    resume_path = _wtal_args.resume

    # Save paths (defined early for pre-validation)
    best_model_path = f"{try_number}_STCA_ViT_VideoMAE_{frame_number}frame_224_Best.pth"
    trained_model_path = f"{try_number}_STCA_ViT_VideoMAE_{frame_number}frame_224_Trainover.pth"

    # ========================
    # Checkpoint pre-validation (before dataset load)
    # ========================
    if resume_path:
        if not os.path.exists(resume_path):
            if accelerator.is_main_process:
                print(f"\nCheckpoint file not found: {resume_path}")
                print("   Aborting training. Please check the path.")
            exit(1)

        if accelerator.is_main_process:
            print("\n" + "=" * 80)
            print("Checkpoint pre-validation (before dataset load)")
            print(f"   Path: {resume_path}")

            checkpoint_preview = torch.load(resume_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint_preview, dict):
                ckpt_epoch = checkpoint_preview.get('epoch', '?')
                ckpt_loss = checkpoint_preview.get('val_loss', checkpoint_preview.get('best_loss', '?'))
                ckpt_acc = checkpoint_preview.get('val_accuracy', checkpoint_preview.get('best_accuracy', '?'))
                ckpt_version = checkpoint_preview.get('version', '?')
                has_train_state = 'train_model_state_dict' in checkpoint_preview
                has_ema_state = 'model_state_dict' in checkpoint_preview
                has_optimizer = 'optimizer_state_dict' in checkpoint_preview
                has_scheduler = 'scheduler_state_dict' in checkpoint_preview

                print(f"   Checkpoint validation successful!")
                print(f"   Epoch: {ckpt_epoch}, Val Loss: {ckpt_loss}, Val Acc: {ckpt_acc}")
                print(f"   Version: {ckpt_version}")
                print(f"   Included states:")
                print(f"      - Train Model: {'OK' if has_train_state else 'MISSING (EMA fallback)'}")
                print(f"      - EMA Model:   {'OK' if has_ema_state else 'MISSING'}")
                print(f"      - Optimizer:   {'OK' if has_optimizer else 'MISSING'}")
                print(f"      - Scheduler:   {'OK' if has_scheduler else 'MISSING'}")

                if not has_ema_state:
                    print(f"\n   model_state_dict is missing. Checkpoint may be corrupted.")
                    print("   Aborting training.")
                    del checkpoint_preview
                    exit(1)
            else:
                print(f"   Checkpoint format is invalid (not a dict).")
                print("   Aborting training.")
                del checkpoint_preview
                exit(1)

            del checkpoint_preview  # Free memory
            torch.cuda.empty_cache()
            print("   Pre-validation complete — starting dataset loading.")
            print("=" * 80)

    modelName = f"{try_number}_STCA_ViT_VideoMAE_Kinetics400_{frame_number}frame_224x224"

    if accelerator.is_main_process:
        print("=" * 80)
        print(f"STCA-ViT VideoMAE ViT-Base Experiment Configuration")
        print(f"   Backbone: VideoMAE ViT-Base (Kinetics-400 finetuned)")
        print(f"   embed_dim: 768, num_heads: 12")
        print("=" * 80)
        print(f"Key improvements:")
        print(f"   1. Backbone: R2Plus1D-18 -> VideoMAE ViT-Base (768-d)")
        print(f"   2. embed_dim: 192 -> 768")
        print(f"   3. Resolution: {h_size}x{w_size}")
        print(f"   4. Backbone LR: x{backbone_lr_ratio}")
        print(f"   5. Augmentation: RandAugment + CutMix + Random Erasing + Temporal Aug")
        print(f"   6. Multi-clip inference: {num_val_clips} clips")
        print(f"   7. WTAL Head: {'ON' if use_wtal else 'OFF'}"
              + (f" (weight={wtal_weight}, warmup={wtal_warmup})" if use_wtal else ""))
        print("=" * 80)

    # WandB
    if accelerator.is_main_process:
        wandb.init(
            project="Kinetics400_STCA_ViT_VideoMAE",
            name=modelName,
            config={
                "version": "stca_vit_videomae",
                "model": "STCA-ViT (VideoMAE ViT-Base)",
                "backbone": "VideoMAE ViT-Base (MCG-NJU/videomae-base-finetuned-kinetics)",
                "embed_dim": 768,
                "num_heads": 12,
                "dataset": "Kinetics-400",
                "num_classes": 400,
                "image_size": h_size,
                "frames_per_clip": frame_number,
                "temporal_stride": temporal_stride,
                "learning_rate": learning_rate,
                "backbone_lr_ratio": backbone_lr_ratio,
                "weight_decay": weight_decay,
                "batch_size": optimal_batch_size,
                "num_epochs": num_epochs,
                "warmup_epochs": warmup_epochs,
                "gradient_accumulation": 4,
                "dropout_rate": dropout_rate,
                "seed": seedValue,
                "use_wtal": use_wtal,
                "wtal_weight": wtal_weight if use_wtal else 0,
                "wtal_warmup": wtal_warmup if use_wtal else 0,
            }
        )

    # ========================
    # Kinetics-400 Dataset
    # ========================
    train_root = "/home/user/data/kinetics-dataset/k400/train"
    val_root = "/home/user/data/kinetics-dataset/k400/val"
    test_root = "/home/user/data/kinetics-dataset/k400/test"

    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("Kinetics-400 Dataset Loading")
        print("=" * 80)
        print(f"   Train path: {train_root}")
        print(f"   Val path: {val_root}")
        print(f"   Test path: {test_root}")
        print(f"   Frames/clip (T): {frame_number} frames")
        print(f"   Temporal Stride: {temporal_stride}")
        print(f"   Resolution: {w_size}x{h_size}")

    import pandas as pd
    annotation_dir = "/home/user/data/kinetics-dataset/k400/annotations"
    train_csv = os.path.join(annotation_dir, "train.csv")
    val_csv = os.path.join(annotation_dir, "val.csv")
    test_csv = os.path.join(annotation_dir, "test.csv")

    train_df = pd.read_csv(train_csv)
    train_video_paths = []
    train_labels = []

    all_classes = sorted(train_df['label'].unique())
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    print(f"\n   Number of classes: {len(all_classes)} (Kinetics-400 standard: 400)")

    for _, row in train_df.iterrows():
        youtube_id = row['youtube_id']
        time_start = str(row['time_start']).zfill(6)
        time_end = str(row['time_end']).zfill(6)
        video_name = f"{youtube_id}_{time_start}_{time_end}.mp4"
        video_path = os.path.join(train_root, video_name)
        if os.path.exists(video_path):
            train_video_paths.append(video_path)
            train_labels.append(class_to_idx[row['label']])

    print(f"   Train videos: {len(train_video_paths)} / {len(train_df)}")

    val_df = pd.read_csv(val_csv)
    val_video_paths = []
    val_labels = []
    for _, row in val_df.iterrows():
        youtube_id = row['youtube_id']
        time_start = str(row['time_start']).zfill(6)
        time_end = str(row['time_end']).zfill(6)
        video_name = f"{youtube_id}_{time_start}_{time_end}.mp4"
        video_path = os.path.join(val_root, video_name)
        if os.path.exists(video_path):
            val_video_paths.append(video_path)
            val_labels.append(class_to_idx[row['label']])

    print(f"   Validation videos: {len(val_video_paths)} / {len(val_df)}")

    test_df = pd.read_csv(test_csv)
    test_video_paths = []
    test_labels = []
    for _, row in test_df.iterrows():
        youtube_id = row['youtube_id']
        time_start = str(row['time_start']).zfill(6)
        time_end = str(row['time_end']).zfill(6)
        video_name = f"{youtube_id}_{time_start}_{time_end}.mp4"
        video_path = os.path.join(test_root, video_name)
        if os.path.exists(video_path):
            test_video_paths.append(video_path)
            test_labels.append(class_to_idx[row['label']])

    print(f"   Test videos: {len(test_video_paths)} / {len(test_df)}")
    print("=" * 80)

    # Dataset creation
    print("\nCreating dataset instances...")

    train_dataset = EnhancedKinetics400Dataset(
        train_video_paths, train_labels,
        clip_length=frame_number, temporal_stride=temporal_stride,
        w_size=w_size, h_size=h_size,
        is_training=True, random_flip=True,
        use_randaugment=use_randaugment, randaug_n=randaug_n, randaug_m=randaug_m,
        use_random_erasing=use_random_erasing, erasing_prob=erasing_prob,
        use_temporal_aug=use_temporal_aug, speed_range=speed_range,
        num_clips=1, crop_mode='random'
    )

    val_dataset = EnhancedKinetics400Dataset(
        val_video_paths, val_labels,
        clip_length=frame_number, temporal_stride=temporal_stride,
        w_size=w_size, h_size=h_size,
        is_training=False, random_flip=False,
        use_randaugment=False, use_random_erasing=False, use_temporal_aug=False,
        num_clips=num_val_clips, crop_mode='center'
    )

    test_dataset = EnhancedKinetics400Dataset(
        test_video_paths, test_labels,
        clip_length=frame_number, temporal_stride=temporal_stride,
        w_size=w_size, h_size=h_size,
        is_training=False, random_flip=False,
        use_randaugment=False, use_random_erasing=False, use_temporal_aug=False,
        num_clips=num_val_clips, crop_mode='center'
    )

    # DataLoader
    cpu_count = os.cpu_count() or 8
    num_workers = min(16, cpu_count // 2)
    if accelerator.is_main_process:
        print(f"DataLoader: num_workers={num_workers}, prefetch_factor=3")

    train_loader = DataLoader(
        train_dataset, batch_size=optimal_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True, prefetch_factor=3
    )

    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False, prefetch_factor=3
    )

    test_loader = DataLoader(
        test_dataset, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False, prefetch_factor=3
    )

    # ========================
    # Model initialization
    # ========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    num_classes = len(all_classes) if len(all_classes) > 0 else 400
    print(f"Number of classes: {num_classes}")

    print(f"\nInitializing STCA-ViT (VideoMAE ViT-Base) model...")

    model = stca_vit_v2_5_parallel(
        num_classes=num_classes,
        dropout=dropout_rate,
        use_gradient_reversal=use_grl,
        grl_lambda=grl_lambda_init,
        use_multi_head_ensemble=use_multi_head,
        use_wtal=use_wtal
    )

    model = model.to(device)

    model_size = calculate_model_size(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {model_size:.2f} MB")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # FLOPs calculation (thop may fail with transformer models)
    try:
        dummy_input = torch.randn(1, 3, frame_number, h_size, w_size).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
        print(f"Parameters: {params / 1e6:.2f} M")
    except Exception as e:
        print(f"FLOPs calculation failed (transformer compatibility issue): {e}")

    # ========================
    # Optimizer (ViT standard settings)
    # ========================
    backbone_lr = learning_rate * backbone_lr_ratio  # 5e-7
    module_lr = learning_rate * 1.0                   # 5e-5
    fusion_lr = learning_rate * 0.8                   # 4e-5
    head_lr = learning_rate * 1.5                     # 7.5e-5

    print(f"\nLR Strategy:")
    print(f"   - VideoMAE Backbone: {backbone_lr:.2e} (x{backbone_lr_ratio})")
    print(f"   - Cross-Attention/MGU/Person Modules: {module_lr:.2e}")
    print(f"   - Fusion: {fusion_lr:.2e}")
    print(f"   - Classification Heads: {head_lr:.2e}")

    backbone_params = list(model.backbone.parameters())
    cross_attn_params = list(model.stca_module.parameters())
    mgu_params = list(model.mgu_module.parameters())
    person_params = list(model.person_module.parameters()) if hasattr(model, 'person_module') else []
    fusion_params = list(model.fusion.parameters())
    classifier_params = list(model.classifier.parameters())
    wtal_params = list(model.tal_head.parameters()) if use_wtal else []

    wtal_lr = learning_rate * 2.0  # WTAL head uses higher LR

    if use_wtal and accelerator.is_main_process:
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

    # Scheduler
    def warmup_cosine(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        return max(min_lr_ratio, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)
    if accelerator.is_main_process:
        print(f"Using Warmup ({warmup_epochs} epochs) + Cosine Annealing scheduler")

    # Loss
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    val_criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    if accelerator.is_main_process:
        print(f"Using LabelSmoothingCrossEntropy (smoothing={label_smoothing})")

    # ========================
    # Prepare with Accelerate
    # ========================
    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler
    )

    # EMA
    if accelerator.is_main_process:
        print("Initializing EMA model...")
    model_ema = ModelEMA(accelerator.unwrap_model(model), decay=0.9999, device=accelerator.device)
    if accelerator.is_main_process:
        print("EMA model initialized")

    # ========================
    # Load checkpoint
    # ========================
    start_epoch = 0
    best_val_loss = float("inf")
    best_accuracy = 0
    patience = 20
    patience_counter = 0

    # Use --resume path if provided, otherwise auto-detect best_model_path
    checkpoint_to_load = resume_path if resume_path else (best_model_path if os.path.exists(best_model_path) else None)

    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        if accelerator.is_main_process:
            print("\n" + "=" * 80)
            print("Checkpoint found! Resuming training.")
            print(f"   Checkpoint path: {checkpoint_to_load}")
            print("=" * 80)

        checkpoint = torch.load(checkpoint_to_load, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict):
            # 1. Restore train model (prefer train_model_state_dict, fallback to model_state_dict)
            unwrapped_model = accelerator.unwrap_model(model)
            if 'train_model_state_dict' in checkpoint:
                unwrapped_model.load_state_dict(checkpoint['train_model_state_dict'], strict=False)
                if accelerator.is_main_process:
                    print("   Train model restored (train_model_state_dict)")
            elif 'model_state_dict' in checkpoint:
                unwrapped_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if accelerator.is_main_process:
                    print("   Train model restored (model_state_dict fallback)")

            # 2. Restore EMA model
            if 'model_state_dict' in checkpoint:
                model_ema.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
                if accelerator.is_main_process:
                    print("   EMA model restored")

            # 3. Restore Optimizer
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if accelerator.is_main_process:
                        print("   Optimizer restored")
                except Exception as e:
                    if accelerator.is_main_process:
                        print(f"   Optimizer restore failed (ignoring): {e}")

            # 4. Restore Scheduler
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    if accelerator.is_main_process:
                        print("   Scheduler restored")
                except Exception as e:
                    if accelerator.is_main_process:
                        print(f"   Scheduler restore failed (ignoring): {e}")

            # 5. Restore training state
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('val_loss', checkpoint.get('best_loss', float('inf')))
            best_accuracy = checkpoint.get('val_accuracy', checkpoint.get('best_accuracy', 0))
            patience_counter = checkpoint.get('patience_counter', 0)

            if accelerator.is_main_process:
                print(f"   Resuming from Epoch {start_epoch} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_accuracy:.2f}%)")
                print("=" * 80 + "\n")
        else:
            if accelerator.is_main_process:
                print("   Checkpoint format mismatch — starting fresh.")
                print("=" * 80 + "\n")
    else:
        if accelerator.is_main_process:
            print("\n" + "=" * 80)
            print("No checkpoint found. Starting training from scratch.")
            print("=" * 80 + "\n")

    # ========================
    # Training loop
    # ========================
    for epoch in range(start_epoch, num_epochs):
        if accelerator.is_main_process:
            print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        # GRL Lambda scheduling
        progress = epoch / num_epochs
        grl_lambda = grl_lambda_init + (grl_lambda_max - grl_lambda_init) * progress
        unwrapped_model = accelerator.unwrap_model(model)
        if use_grl and hasattr(unwrapped_model, 'set_grl_lambda'):
            unwrapped_model.set_grl_lambda(grl_lambda)

        # Training
        train_loss, train_accuracy, ortho_loss, wtal_loss = train_epoch(
            model, train_loader, criterion, optimizer, accelerator, device,
            model_ema=model_ema, max_grad_norm=max_grad_norm,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha,
            use_cutmix=use_cutmix, cutmix_alpha=cutmix_alpha,
            use_ortho_loss=use_ortho_loss, ortho_weight=ortho_weight,
            use_wtal=use_wtal, wtal_weight=wtal_weight,
            current_epoch=epoch, wtal_warmup=wtal_warmup
        )

        # Validation (Multi-clip + TTA)
        use_tta = (epoch >= num_epochs - 5)
        if accelerator.is_main_process:
            tta_msg = " + TTA" if use_tta else ""
            print(f"Starting validation (EMA model{tta_msg}, {num_val_clips}-clip)...")

        ema_device = accelerator.device
        model_ema.module.to(ema_device)

        wtal_active = use_wtal and (epoch >= wtal_warmup)
        val_loss, val_accuracy, val_f1, wtal_metrics = validate_epoch_multiclip(
            model_ema.module, val_loader, val_criterion, ema_device,
            accelerator=accelerator, use_tta=use_tta, num_clips=num_val_clips,
            compute_wtal_metrics=wtal_active
        )

        if scheduler is not None:
            scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        if accelerator.is_main_process:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            if ortho_loss > 0:
                print(f"Ortho Loss: {ortho_loss:.4f}, GRL lambda: {grl_lambda:.4f}")
            if wtal_loss > 0:
                print(f"WTAL Loss: {wtal_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val F1: {val_f1:.4f}")
            if wtal_metrics is not None:
                print(f"WTAL Metrics: fg_score={wtal_metrics['mean_fg_score']:.3f}, "
                      f"sparsity={wtal_metrics['fg_sparsity']:.3f}, "
                      f"peak_center={wtal_metrics['peak_center_ratio']:.3f}")
            print(f"Learning Rate: {current_lr:.2e}")
            print(f"Current best: Val Loss {best_val_loss:.4f}, Val Acc {best_accuracy:.2f}%")

            if hasattr(unwrapped_model, 'get_contribution_weights'):
                contrib_weights = unwrapped_model.get_contribution_weights()
                if isinstance(contrib_weights, dict):
                    print(f"Contribution: alpha={contrib_weights.get('alpha (stca)', contrib_weights.get('alpha (cross)', 0)):.3f}, "
                          f"beta={contrib_weights.get('beta (mgu)', 0):.3f}, "
                          f"gamma={contrib_weights.get('gamma (person)', 0):.3f}")

            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "ortho_loss": ortho_loss,
                "grl_lambda": grl_lambda,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
                "learning_rate": current_lr,
                "gpu_memory": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            }
            if use_wtal:
                log_dict['wtal_loss'] = wtal_loss
            if wtal_metrics is not None:
                log_dict['wtal_fg_score'] = wtal_metrics['mean_fg_score']
                log_dict['wtal_fg_sparsity'] = wtal_metrics['fg_sparsity']
                log_dict['wtal_peak_center'] = wtal_metrics['peak_center_ratio']
            wandb.log(log_dict)

        # Save best model
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            best_accuracy = val_accuracy
            patience_counter = 0

            if accelerator.is_main_process:
                unwrapped_train_model = accelerator.unwrap_model(model)
                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_ema.module.state_dict(),
                    'train_model_state_dict': unwrapped_train_model.state_dict(),
                    'ema_state_dict': {
                        'decay': model_ema.decay,
                        'shadow': {k: v.clone() for k, v in model_ema.module.state_dict().items()},
                    },
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'best_accuracy': val_accuracy,
                    'best_loss': val_loss,
                    'patience_counter': patience_counter,
                    'model_variant': 'VideoMAE_ViT-B',
                    'version': 'v2.5',
                    'resolution': f'{h_size}x{w_size}'
                }
                if scheduler is not None:
                    save_dict['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(save_dict, best_model_path)

                print(f"New best model saved! (EMA model)")
                print(f"   Improvement: {improvement:.4f} (val_loss)")
                print(f"   Saved to: {best_model_path}")
        else:
            patience_counter += 1
            if accelerator.is_main_process:
                print(f"Patience: {patience_counter}/{patience}")

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if patience_counter >= patience:
            if accelerator.is_main_process:
                print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    if accelerator.is_main_process:
        print("Training complete!")

    # Save final model
    if accelerator.is_main_process:
        unwrapped_train_model = accelerator.unwrap_model(model)
        final_save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model_ema.module.state_dict(),
            'train_model_state_dict': unwrapped_train_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'training_completed': True,
            'model_variant': 'VideoMAE_ViT-B',
            'version': 'v2.5',
            'resolution': f'{h_size}x{w_size}'
        }
        if scheduler is not None:
            final_save_dict['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(final_save_dict, trained_model_path)

    # ========================
    # Test evaluation (Multi-clip + TTA)
    # ========================
    if accelerator.is_main_process:
        print(f"\n=== Test with best model (Val Acc: {best_accuracy:.2f}%) ===")
        print(f"    Multi-clip: {num_val_clips}, TTA: Horizontal Flip")

        unwrapped_model = accelerator.unwrap_model(model)

        try:
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            unwrapped_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Checkpoint loaded successfully!")
        except Exception as e:
            print(f"Checkpoint loading failed: {e}")

        unwrapped_model.eval()

        print("Starting test evaluation...")
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            test_bar = tqdm(test_loader, desc="Testing")
            for i, (videos, labels) in enumerate(test_bar):
                if videos.dim() == 6:
                    B, num_clips, C, T, H, W = videos.shape
                    all_outputs = []
                    for clip_idx in range(num_clips):
                        clip = videos[:, clip_idx]
                        with accelerator.autocast():
                            out_normal = unwrapped_model(clip)
                            out_flipped = unwrapped_model(torch.flip(clip, dims=[4]))
                            out = (out_normal + out_flipped) / 2.0
                        all_outputs.append(out)
                    outputs = torch.stack(all_outputs).mean(dim=0)
                else:
                    with accelerator.autocast():
                        outputs_normal = unwrapped_model(videos)
                        outputs_flipped = unwrapped_model(torch.flip(videos, dims=[4]))
                        outputs = (outputs_normal + outputs_flipped) / 2.0

                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                test_bar.set_postfix(test_loss=loss.item())

                if i % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        test_accuracy = 100 * correct / total
        test_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        test_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        print(f"\n=== Final Test Results (VideoMAE) ===")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")

        wandb.log({
            "final_test_accuracy": test_accuracy,
            "final_test_precision": test_precision,
            "final_test_recall": test_recall,
            "final_test_f1": test_f1,
            "best_val_accuracy": best_accuracy,
            "best_val_loss": best_val_loss
        })

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix - STCA-ViT VideoMAE")
        plt.tight_layout()
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.savefig("confusion_matrix_stca_vit_VideoMAE_224.png", dpi=150)
        plt.close()

        wandb.finish()
        print("\nAll tasks completed!")
