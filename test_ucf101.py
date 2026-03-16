"""
STCA-ViT Test Script for UCF-101
====================================================

Loads a trained checkpoint and performs comprehensive evaluation on the test set.
Test set: group 20~25 (same split criterion as train_ucf101.py)

Generated visualizations:
---------
1. 06_ucf101_confusion.png           - Normalized Confusion Matrix (Reds, %)
2. 07_ucf101_f1score.png             - F1-Score per Class (color-coded bar chart)
3. 08_ucf101_class_performance_detailed.png - Top/Bottom 10, distribution, scatter
4. 09_ucf101_distribution_of_samples_per_class.png - Histogram of samples per class
5. 10_ucf101_prediction_confidence_analysis.png - Confidence analysis (includes ECE)

Usage:
---------
python test_ucf101.py --checkpoint STCA_ViT_UCF101_Best.pth
python test_ucf101.py --checkpoint STCA_ViT_UCF101_Best.pth --detailed --analyze_samples 10 --save_predictions
"""

import os
import sys
import re
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, OrderedDict
import argparse
import csv
import json
from datetime import datetime

import decord
decord.bridge.set_bridge('torch')

from tqdm import tqdm
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    accuracy_score, top_k_accuracy_score, precision_recall_fscore_support
)

# STCA-ViT model import
sys.path.insert(0, '/home/user/data/codes')
from model import stca_vit_v2_5_parallel

# Suppress OpenCV/FFmpeg error messages
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'


# ===========================
# UCF-101 Test Dataset
# ===========================
class UCF101TestDataset(Dataset):
    """
    UCF-101 Test Dataset

    Filename pattern: v_{ClassName}_g{group:02d}_c{clip:02d}.avi
    - group {train_group_max+1}~25 -> Test
      (default: group 20~25, same split as train_ucf101.py)
    """
    KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
    KINETICS_STD  = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

    def __init__(self, data_root, clip_length=16, frame_stride=4,
                 w_size=224, h_size=224, train_group_max=19,
                 multi_crop=False):
        self.data_root = data_root
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.w_size = w_size
        self.h_size = h_size
        self.train_group_max = train_group_max
        self.multi_crop = multi_crop

        # Class list (same sorted order as training, only folders containing .avi files)
        self.classes = sorted([d for d in os.listdir(data_root)
                               if os.path.isdir(os.path.join(data_root, d))
                               and any(f.endswith('.avi') for f in os.listdir(os.path.join(data_root, d)))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        self.samples = self._load_test_samples()
        print(f"UCF-101 Test: {len(self.samples)} videos, {len(self.classes)} classes")
        print(f"   (group {train_group_max + 1}~25 -> Test)")

    def _load_test_samples(self):
        test_samples = []
        group_pattern = re.compile(r'_g(\d+)_c\d+\.avi$', re.IGNORECASE)

        for class_name in self.classes:
            class_dir = os.path.join(self.data_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = self.class_to_idx[class_name]
            for video_name in sorted(os.listdir(class_dir)):
                if not video_name.endswith('.avi'):
                    continue
                m = group_pattern.search(video_name)
                if m is None:
                    continue
                group_num = int(m.group(1))
                if group_num > self.train_group_max:
                    video_path = os.path.join(class_dir, video_name)
                    test_samples.append((video_path, label))
        return test_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        if self.multi_crop:
            clips = self._load_video_multi_crop(video_path)
            return clips, label, video_path
        else:
            clip = self._load_video(video_path)
            return clip, label, video_path

    def _get_center_frame_indices(self, total_frames):
        stride = self.frame_stride
        span = self.clip_length * stride
        if total_frames >= span:
            start = (total_frames - span) // 2
        else:
            start = 0
        indices = [min(start + i * stride, max(total_frames - 1, 0))
                   for i in range(self.clip_length)]
        return indices

    def _load_frames(self, video_path):
        try:
            vr = decord.VideoReader(video_path, num_threads=1)
            total_frames = len(vr)
            if total_frames == 0:
                return None, 0
            frame_indices = self._get_center_frame_indices(total_frames)
            frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]
            frames_tensor = vr.get_batch(frame_indices)
            frames_np = frames_tensor.numpy().astype(np.uint8)
            return frames_np, total_frames
        except Exception as e:
            print(f"Warning: Failed to load {video_path}: {e}")
            return None, 0

    def _process_frames(self, frames_np, crop_type='center'):
        processed = []
        for t in range(frames_np.shape[0]):
            frame = frames_np[t]
            h_orig, w_orig = frame.shape[:2]
            if crop_type == 'center':
                frame = cv2.resize(frame, (self.w_size, self.h_size))
            elif crop_type == 'top_left':
                crop_h, crop_w = int(h_orig * 0.875), int(w_orig * 0.875)
                frame = frame[:crop_h, :crop_w]
                frame = cv2.resize(frame, (self.w_size, self.h_size))
            elif crop_type == 'top_right':
                crop_h, crop_w = int(h_orig * 0.875), int(w_orig * 0.875)
                frame = frame[:crop_h, w_orig - crop_w:]
                frame = cv2.resize(frame, (self.w_size, self.h_size))
            elif crop_type == 'bottom_left':
                crop_h, crop_w = int(h_orig * 0.875), int(w_orig * 0.875)
                frame = frame[h_orig - crop_h:, :crop_w]
                frame = cv2.resize(frame, (self.w_size, self.h_size))
            elif crop_type == 'bottom_right':
                crop_h, crop_w = int(h_orig * 0.875), int(w_orig * 0.875)
                frame = frame[h_orig - crop_h:, w_orig - crop_w:]
                frame = cv2.resize(frame, (self.w_size, self.h_size))
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - self.KINETICS_MEAN) / self.KINETICS_STD
            processed.append(frame)
        clip = np.stack(processed, axis=0).transpose(3, 0, 1, 2)
        return torch.from_numpy(clip).float()

    def _load_video(self, video_path):
        frames_np, _ = self._load_frames(video_path)
        if frames_np is None:
            return torch.zeros((3, self.clip_length, self.h_size, self.w_size))
        return self._process_frames(frames_np, crop_type='center')

    def _load_video_multi_crop(self, video_path):
        frames_np, _ = self._load_frames(video_path)
        if frames_np is None:
            zero_clip = torch.zeros((3, self.clip_length, self.h_size, self.w_size))
            return torch.stack([zero_clip] * 5)
        crops = ['center', 'top_left', 'top_right', 'bottom_left', 'bottom_right']
        clips = [self._process_frames(frames_np, crop_type=c) for c in crops]
        return torch.stack(clips)


# ===========================
# Evaluation Function
# ===========================
def evaluate(model, test_loader, device, use_tta=True, multi_crop=False, detailed=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    head_correct = defaultdict(int)
    head_total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            videos, labels, paths = batch
            labels = labels.to(device)

            if multi_crop:
                B, num_crops, C, T, H, W = videos.shape
                clips_flat = videos.view(B * num_crops, C, T, H, W).to(device)
                logits_flat = model(clips_flat)
                logits = logits_flat.view(B, num_crops, -1).mean(dim=1)
                clips_flip = torch.flip(clips_flat, dims=[4])
                logits_flip = model(clips_flip)
                logits_flip = logits_flip.view(B, num_crops, -1).mean(dim=1)
                logits = (logits + logits_flip) * 0.5
            else:
                videos = videos.to(device)
                if detailed:
                    outputs = model(videos, return_all=True)
                    logits = outputs['final']
                    if use_tta:
                        outputs_flip = model(torch.flip(videos, dims=[4]), return_all=True)
                        logits = (logits + outputs_flip['final']) * 0.5
                    for key in ['backbone', 'stca', 'mgu', 'person', 'fused']:
                        if key in outputs:
                            _, head_pred = torch.max(outputs[key], 1)
                            head_correct[key] += (head_pred == labels).sum().item()
                    head_total += labels.size(0)
                else:
                    logits = model(videos)
                    if use_tta:
                        logits_flip = model(torch.flip(videos, dims=[4]))
                        logits = (logits + logits_flip) * 0.5

            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    all_probs = np.array(all_probs)
    top1_acc = accuracy_score(all_labels, all_preds) * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    top5_acc = top_k_accuracy_score(all_labels, all_probs, k=5, labels=range(all_probs.shape[1])) * 100
    head_accuracies = {k: 100 * v / head_total for k, v in head_correct.items()} if head_total > 0 else {}

    return {
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs,
        'all_paths': all_paths,
        'head_accuracies': head_accuracies,
    }


# ===================================================================
# Visualization Functions
# ===================================================================
def _setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Tinos', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 20,
        'axes.titlesize': 23,
        'axes.labelsize': 21,
        'xtick.labelsize': 19,
        'ytick.labelsize': 19,
        'legend.fontsize': 19,
        'figure.dpi': 150,
    })
    return plt


# -----------------------------------------------------------------
# 06. Confusion Matrix (Normalized, Reds colormap, Percentage)
# -----------------------------------------------------------------
def plot_06_confusion_matrix(all_labels, all_preds, idx_to_class, save_path):
    plt = _setup_matplotlib()
    num_classes = len(idx_to_class)
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6) * 100

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Reds', vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Percentage (%)', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=13)
    ax.set_xlabel('Predicted Label', fontsize=18, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=18, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [06] Confusion Matrix saved: {save_path}")


# -----------------------------------------------------------------
# 07. F1-Score per Class (color-coded bar chart + average line)
# -----------------------------------------------------------------
def plot_07_f1score_per_class(all_labels, all_preds, num_classes, save_path):
    plt = _setup_matplotlib()
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(num_classes), zero_division=0
    )
    colors = []
    for f in f1_per_class:
        if f >= 0.9:
            colors.append('#2ecc71')
        elif f >= 0.8:
            colors.append('#3498db')
        elif f >= 0.6:
            colors.append('#f39c12')
        else:
            colors.append('#e74c3c')

    avg_f1 = np.mean(f1_per_class)
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(num_classes)
    ax.bar(x, f1_per_class, color=colors, width=0.8, edgecolor='none')
    ax.axhline(y=avg_f1, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    # Avg annotation
    mid_x = num_classes // 2
    ax.annotate(f'Avg: {avg_f1:.3f}', xy=(mid_x, avg_f1),
                xytext=(mid_x + 2, avg_f1 + 0.05),
                fontsize=13, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
    ax.set_xlabel('Class Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax.set_title('F1-Score per Class', fontsize=18, fontweight='bold')
    ax.set_xlim(-0.5, num_classes - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, num_classes, 5))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='≥ 0.9'),
        Patch(facecolor='#3498db', label='0.8~0.89'),
        Patch(facecolor='#f39c12', label='0.6~0.79'),
        Patch(facecolor='#e74c3c', label='< 0.6'),
    ]
    ax.legend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, -0.35), ncol=4, fontsize=18,
              frameon=True, edgecolor='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [07] F1-Score per Class saved: {save_path}")
    return f1_per_class


# -----------------------------------------------------------------
# 08. Class Performance Detailed (4 subplots)
# -----------------------------------------------------------------
def plot_08_class_performance_detailed(all_labels, all_preds, idx_to_class, num_classes, save_path):
    plt = _setup_matplotlib()
    prec, rec, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(num_classes), zero_division=0
    )
    class_names = [idx_to_class[i] for i in range(num_classes)]
    sorted_idx = np.argsort(f1_per_class)
    top10_idx = sorted_idx[-10:][::-1]
    bot10_idx = sorted_idx[:10]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # --- Top-left: Top 10 Classes by F1-Score ---
    ax = axes[0, 0]
    top10_names = [f"C{i}" for i in top10_idx][::-1]
    top10_f1 = [f1_per_class[i] for i in top10_idx][::-1]
    bars = ax.barh(range(10), top10_f1, color='#82d982', edgecolor='none', height=0.7)
    for bar, val in zip(bars, top10_f1):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    ax.set_yticks(range(10))
    ax.set_yticklabels(top10_names, fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title('Top 10 Classes by F1-Score', fontsize=14, fontweight='bold')

    # --- Top-right: Bottom 10 Classes by F1-Score ---
    ax = axes[0, 1]
    bot10_names = [f"C{i}" for i in bot10_idx][::-1]
    bot10_f1 = [f1_per_class[i] for i in bot10_idx][::-1]
    bars = ax.barh(range(10), bot10_f1, color='#f8a0a0', edgecolor='none', height=0.7)
    for bar, val in zip(bars, bot10_f1):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    ax.set_yticks(range(10))
    ax.set_yticklabels(bot10_names, fontsize=10)
    max_val = max(bot10_f1) if bot10_f1 else 1.0
    ax.set_xlim(0, max_val * 1.3 + 0.05)
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title('Bottom 10 Classes by F1-Score', fontsize=14, fontweight='bold')

    # --- Bottom-left: Performance Metrics Distribution ---
    ax = axes[1, 0]
    bins = np.arange(0, 1.05, 0.05)
    ax.hist(prec, bins=bins, alpha=0.5, label='Precision', color='#5dade2', edgecolor='white')
    ax.hist(rec, bins=bins, alpha=0.5, label='Recall', color='#58d68d', edgecolor='white')
    ax.hist(f1_per_class, bins=bins, alpha=0.5, label='F1-Score', color='#e74c3c', edgecolor='white')
    ax.axvline(np.median(prec), color='#2980b9', linestyle='--', linewidth=1.5)
    ax.axvline(np.median(rec), color='#27ae60', linestyle='--', linewidth=1.5)
    ax.axvline(np.median(f1_per_class), color='#c0392b', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Number of Classes', fontsize=12)
    ax.set_title('Performance Metrics Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=15)

    # --- Bottom-right: F1-Score vs Sample Support ---
    ax = axes[1, 1]
    scatter = ax.scatter(support, f1_per_class, c=f1_per_class, cmap='YlGn',
                         s=50, edgecolors='gray', linewidths=0.5, vmin=0, vmax=1)
    fig.colorbar(scatter, ax=ax, label='F1-Score', fraction=0.046, pad=0.04)
    # Correlation
    if len(support) > 1:
        corr = np.corrcoef(support, f1_per_class)[0, 1]
    else:
        corr = 0.0
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
            fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('Support (Number of Samples)', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score vs Sample Support', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [08] Class Performance Detailed saved: {save_path}")


# -----------------------------------------------------------------
# 09. Distribution of Samples per Class (histogram)
# -----------------------------------------------------------------
def plot_09_sample_distribution(all_labels, num_classes, save_path):
    plt = _setup_matplotlib()

    class_counts = np.zeros(num_classes, dtype=int)
    for label in all_labels:
        class_counts[label] += 1

    mean_count = np.mean(class_counts)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(class_counts, bins=10, color='#aed6f1', edgecolor='#3498db', linewidth=1.2)
    ax.axvline(mean_count, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {mean_count:.1f}')
    ax.set_xlabel('Number of Samples', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Classes', fontsize=16, fontweight='bold')
    ax.set_title('Distribution of Samples per Class', fontsize=20, fontweight='bold')
    ax.legend(fontsize=15)
    ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [09] Sample Distribution saved: {save_path}")


# -----------------------------------------------------------------
# 10. Prediction Confidence Analysis (4 subplots + ECE)
# -----------------------------------------------------------------
def plot_10_confidence_analysis(all_labels, all_preds, all_probs, idx_to_class, num_classes, save_path):
    plt = _setup_matplotlib()

    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    confidences = np.max(all_probs, axis=1)
    correct_mask = (all_labels_np == all_preds_np)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Top-left: Overall Prediction Confidence Distribution ---
    ax = axes[0, 0]
    ax.hist(confidences, bins=40, color='#7d8cff', edgecolor='white', alpha=0.85)
    mean_conf = np.mean(confidences)
    median_conf = np.median(confidences)
    ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_conf:.3f}')
    ax.axvline(median_conf, color='green', linestyle='--', linewidth=2, label=f'Median: {median_conf:.3f}')
    ax.set_xlabel('Confidence (Max Probability)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Overall Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=15)

    # --- Top-right: Confidence Distribution: Correct vs Incorrect ---
    ax = axes[0, 1]
    bins = np.linspace(0, 1, 30)
    ax.hist(confidences[correct_mask], bins=bins, density=True, alpha=0.7,
            color='#27ae60', edgecolor='white', label='Correct Predictions')
    ax.hist(confidences[~correct_mask], bins=bins, density=True, alpha=0.7,
            color='#e74c3c', edgecolor='white', label='Incorrect Predictions')
    if correct_mask.sum() > 0:
        ax.axvline(np.median(confidences[correct_mask]), color='#1e8449',
                   linestyle='--', linewidth=2)
    if (~correct_mask).sum() > 0:
        ax.axvline(np.median(confidences[~correct_mask]), color='#922b21',
                   linestyle='--', linewidth=2)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Confidence Distribution: Correct vs Incorrect', fontsize=14, fontweight='bold')
    ax.legend(fontsize=15)

    # --- Bottom-left: Reliability Diagram (ECE) ---
    ax = axes[1, 0]
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    ece = 0.0
    total_samples = len(all_labels)
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= low) & (confidences < high)
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        count = mask.sum()
        bin_counts.append(count)
        if count > 0:
            acc = correct_mask[mask].mean()
            conf = confidences[mask].mean()
            bin_accs.append(acc)
            bin_confs.append(conf)
            ece += (count / total_samples) * abs(acc - conf)
        else:
            bin_accs.append(0)
            bin_confs.append((low + high) / 2)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bar_width = 1.0 / n_bins * 0.85
    bars = ax.bar(bin_centers, bin_accs, width=bar_width, color='#9b59b6',
                  edgecolor='white', alpha=0.85, zorder=2)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration', zorder=1)
    for bc, ba, count in zip(bin_centers, bin_accs, bin_counts):
        if count > 0:
            ax.text(bc, ba + 0.02, f'{ba:.2f}\n({count})', ha='center', va='bottom',
                    fontsize=7, fontweight='bold')
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Reliability Diagram (ECE: {ece:.3f})', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=15, loc='lower right')

    # --- Bottom-right: Per-Class Confidence vs Accuracy ---
    ax = axes[1, 1]
    class_confs, class_accs, class_acc_vals = [], [], []
    for i in range(num_classes):
        mask = (all_labels_np == i)
        if mask.sum() > 0:
            c_conf = confidences[mask].mean()
            c_acc = correct_mask[mask].mean()
        else:
            c_conf, c_acc = 0.0, 0.0
        class_confs.append(c_conf)
        class_accs.append(c_acc)
        class_acc_vals.append(c_acc)
    class_confs = np.array(class_confs)
    class_accs = np.array(class_accs)
    class_acc_vals = np.array(class_acc_vals)
    scatter = ax.scatter(class_confs, class_accs, c=class_acc_vals, cmap='RdYlGn',
                         s=60, edgecolors='gray', linewidths=0.5, vmin=0, vmax=1, zorder=2)
    fig.colorbar(scatter, ax=ax, label='Accuracy', fraction=0.046, pad=0.04)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, zorder=1)
    ax.set_xlabel('Average Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Confidence vs Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [10] Confidence Analysis saved: {save_path}")
    return ece


# ===========================
# Console Output Helpers
# ===========================
def print_per_class_accuracy(all_labels, all_preds, idx_to_class):
    num_classes = len(idx_to_class)
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for label, pred in zip(all_labels, all_preds):
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1

    print(f"\n{'='*70}")
    print(f"{'Per-Class Accuracy':^70}")
    print(f"{'='*70}")
    print(f"{'No.':<5} {'Class':<30} {'Correct/Total':<15} {'Accuracy':>10}")
    print(f"{'-'*70}")

    accuracies = []
    for i in range(num_classes):
        total = class_total.get(i, 0)
        correct = class_correct.get(i, 0)
        acc = 100 * correct / total if total > 0 else 0
        accuracies.append(acc)
        print(f"{i+1:<5} {idx_to_class.get(i, f'class_{i}'):<30} {correct:>4}/{total:<8} {acc:>8.2f}%")

    print(f"{'-'*70}")
    print(f"{'Mean':<35} {'':<15} {np.mean(accuracies):>8.2f}%")
    print(f"{'='*70}")
    return accuracies


def analyze_misclassified(all_labels, all_preds, all_paths, idx_to_class, top_n=10):
    misclassified = []
    for label, pred, path in zip(all_labels, all_preds, all_paths):
        if label != pred:
            misclassified.append({'true': idx_to_class[label], 'pred': idx_to_class[pred], 'path': path})

    print(f"\n{'='*70}")
    print(f"Misclassified: {len(misclassified)}/{len(all_labels)} "
          f"({100*len(misclassified)/len(all_labels):.2f}%)")
    print(f"{'='*70}")

    confusion_pairs = defaultdict(int)
    for m in misclassified:
        confusion_pairs[(m['true'], m['pred'])] += 1

    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: -x[1])
    print(f"\nMost Confused Pairs (Top-{min(top_n, len(sorted_pairs))}):")
    for (true_cls, pred_cls), count in sorted_pairs[:top_n]:
        print(f"   {true_cls:<25} -> {pred_cls:<25} ({count})")
    return misclassified


# ===========================
# CSV Export for Figure Data
# ===========================
def save_figure_data_csv(all_labels, all_preds, all_probs, idx_to_class, num_classes, output_dir):
    csv_dir = os.path.join(output_dir, 'figure_data_csv')
    os.makedirs(csv_dir, exist_ok=True)

    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # 06: Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6) * 100
    path_06 = os.path.join(csv_dir, '06_confusion_matrix.csv')
    with open(path_06, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([''] + class_names)
        for i in range(num_classes):
            writer.writerow([class_names[i]] + [f'{v:.4f}' for v in cm_norm[i]])
    print(f"  [CSV] 06_confusion_matrix.csv saved")

    # 07 & 08: Per-class metrics
    prec, rec, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(num_classes), zero_division=0
    )
    path_07_08 = os.path.join(csv_dir, '07_08_class_metrics.csv')
    with open(path_07_08, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_idx', 'class_name', 'precision', 'recall', 'f1_score', 'support'])
        for i in range(num_classes):
            writer.writerow([i, class_names[i],
                             f'{prec[i]:.6f}', f'{rec[i]:.6f}',
                             f'{f1_per_class[i]:.6f}', int(support[i])])
    print(f"  [CSV] 07_08_class_metrics.csv saved")

    # 09: Sample distribution
    class_counts = np.zeros(num_classes, dtype=int)
    for label in all_labels:
        class_counts[label] += 1
    path_09 = os.path.join(csv_dir, '09_sample_distribution.csv')
    with open(path_09, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_idx', 'class_name', 'sample_count'])
        for i in range(num_classes):
            writer.writerow([i, class_names[i], int(class_counts[i])])
    print(f"  [CSV] 09_sample_distribution.csv saved")

    # 10a: Per-sample confidence
    confidences = np.max(all_probs, axis=1)
    correct_mask = (all_labels_np == all_preds_np)
    path_10a = os.path.join(csv_dir, '10a_sample_confidence.csv')
    with open(path_10a, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_idx', 'true_label', 'pred_label', 'confidence', 'correct'])
        for i in range(len(all_labels)):
            writer.writerow([i, all_labels[i], all_preds[i],
                             f'{confidences[i]:.6f}', int(correct_mask[i])])
    print(f"  [CSV] 10a_sample_confidence.csv saved")

    # 10b: Reliability diagram
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    total_samples = len(all_labels)
    path_10b = os.path.join(csv_dir, '10b_reliability_diagram.csv')
    with open(path_10b, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['bin_idx', 'bin_low', 'bin_high', 'bin_center',
                         'accuracy', 'avg_confidence', 'count', 'ece_contribution'])
        for i in range(n_bins):
            low, high = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences >= low) & (confidences < high)
            if i == n_bins - 1:
                mask = (confidences >= low) & (confidences <= high)
            count = int(mask.sum())
            if count > 0:
                acc = float(correct_mask[mask].mean())
                conf = float(confidences[mask].mean())
                ece_contrib = (count / total_samples) * abs(acc - conf)
            else:
                acc, conf, ece_contrib = 0.0, (low + high) / 2, 0.0
            center = (low + high) / 2
            writer.writerow([i, f'{low:.2f}', f'{high:.2f}', f'{center:.2f}',
                             f'{acc:.6f}', f'{conf:.6f}', count, f'{ece_contrib:.6f}'])
    print(f"  [CSV] 10b_reliability_diagram.csv saved")

    # 10c: Per-class confidence
    path_10c = os.path.join(csv_dir, '10c_per_class_confidence.csv')
    with open(path_10c, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class_idx', 'class_name', 'avg_confidence', 'accuracy', 'sample_count'])
        for i in range(num_classes):
            mask = (all_labels_np == i)
            count = int(mask.sum())
            if count > 0:
                c_conf = float(confidences[mask].mean())
                c_acc = float(correct_mask[mask].mean())
            else:
                c_conf, c_acc = 0.0, 0.0
            writer.writerow([i, class_names[i], f'{c_conf:.6f}', f'{c_acc:.6f}', count])
    print(f"  [CSV] 10c_per_class_confidence.csv saved")

    print(f"  [CSV] All figure data saved to: {csv_dir}/")
    return csv_dir


# ===========================
# Main
# ===========================
def main():
    parser = argparse.ArgumentParser(description='STCA-ViT Test for UCF-101')

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='../ucf101/ucf101')
    parser.add_argument('--train_group_max', type=int, default=19,
                        help='Maximum group number used during training (default: 19 -> group 20~25 = test)')

    # Model configuration
    parser.add_argument('--num_classes', type=int, default=101)
    parser.add_argument('--module_embed_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip_length', type=int, default=16)
    parser.add_argument('--frame_stride', type=int, default=4,
                        help='Frame sampling stride (recommended: 4 for UCF-101 at 30fps)')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--use_wtal', action='store_true', default=False)

    # Evaluation options
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_tta', action='store_true', default=False)
    parser.add_argument('--multi_crop', action='store_true', default=False)
    parser.add_argument('--detailed', action='store_true', default=False)

    # Analysis options
    parser.add_argument('--analyze_samples', type=int, default=0)

    # Output
    parser.add_argument('--output_dir', type=str, default='test_results_ucf101')
    parser.add_argument('--save_predictions', action='store_true', default=False)

    args = parser.parse_args()

    # ========================
    # Setup
    # ========================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    if args.batch_size is None:
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram < 12:
                args.batch_size = 2
            elif vram < 24:
                args.batch_size = 8
            elif vram < 48:
                args.batch_size = 12
            else:
                args.batch_size = 20
        else:
            args.batch_size = 1
    if args.multi_crop:
        args.batch_size = max(1, args.batch_size // 5)

    os.makedirs(args.output_dir, exist_ok=True)

    # ========================
    # Load Checkpoint
    # ========================
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"   Best Val Loss: {checkpoint['best_val_loss']:.4f}")
    if 'best_val_acc' in checkpoint:
        print(f"   Best Val Acc: {checkpoint['best_val_acc']:.2f}%")
    if 'val_f1' in checkpoint:
        print(f"   Val F1: {checkpoint['val_f1']:.4f}")

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
    else:
        raise KeyError("No model_state_dict or ema_state_dict in checkpoint.")

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        if name.endswith('total_ops') or name.endswith('total_params'):
            continue
        new_state_dict[name] = v

    # Auto-detect module_embed_dim
    probe_key = 'stca_module.out_proj.0.weight'
    if probe_key in new_state_dict:
        detected_embed_dim = new_state_dict[probe_key].shape[0]
        if detected_embed_dim != args.module_embed_dim:
            print(f"   [Auto-detect] module_embed_dim: {args.module_embed_dim} -> {detected_embed_dim}")
            args.module_embed_dim = detected_embed_dim

    # Auto-detect use_wtal
    has_wtal_keys = any(k.startswith('tal_head.') for k in new_state_dict)
    if has_wtal_keys and not args.use_wtal:
        print(f"   [Auto-detect] use_wtal: False -> True")
        args.use_wtal = True

    # ========================
    # Model
    # ========================
    print(f"\n{'='*60}")
    print(f"Model: STCA-ViT (VideoMAE ViT-Base)")
    print(f"   module_embed_dim: {args.module_embed_dim}")
    print(f"   num_classes: {args.num_classes}")
    print(f"   use_wtal: {args.use_wtal}")
    print(f"{'='*60}")

    model = stca_vit_v2_5_parallel(
        num_classes=args.num_classes,
        dropout=args.dropout,
        module_embed_dim=args.module_embed_dim,
        use_gradient_reversal=True,
        grl_lambda=0.0,
        use_multi_head_ensemble=True,
        freeze_backbone_layers=6,
        use_gradient_checkpointing=False,
        use_wtal=args.use_wtal,
    ).to(device)

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    print("Model loaded (EMA weights)")

    missing = [k for k in model.state_dict() if k not in new_state_dict]
    unexpected = [k for k in new_state_dict if k not in model.state_dict()]
    if missing:
        print(f"   Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"   Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f} M")

    # ========================
    # Dataset & DataLoader
    # ========================
    print(f"\nDataset: {args.data_root}")
    test_dataset = UCF101TestDataset(
        data_root=args.data_root,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        w_size=args.image_size,
        h_size=args.image_size,
        train_group_max=args.train_group_max,
        multi_crop=args.multi_crop,
    )

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs = {'prefetch_factor': 2, 'persistent_workers': True}

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, **loader_kwargs,
    )
    idx_to_class = test_dataset.idx_to_class
    num_classes = len(idx_to_class)

    # ========================
    # Evaluation
    # ========================
    use_tta = not args.no_tta
    mode = "Multi-Crop TTA" if args.multi_crop else ("TTA (Flip)" if use_tta else "Single-crop")
    print(f"\nEvaluation mode: {mode}")
    print(f"Batch size: {args.batch_size}\n")

    results = evaluate(model, test_loader, device,
                       use_tta=use_tta, multi_crop=args.multi_crop, detailed=args.detailed)

    # ========================
    # Print Results
    # ========================
    print(f"\n{'='*60}")
    print(f"{'TEST RESULTS (UCF-101)':^60}")
    print(f"{'='*60}")
    print(f"   Top-1 Accuracy:      {results['top1_acc']:.2f}%")
    print(f"   Top-5 Accuracy:      {results['top5_acc']:.2f}%")
    print(f"   F1 Score (Macro):    {results['f1_macro']:.4f}")
    print(f"   F1 Score (Weighted): {results['f1_weighted']:.4f}")
    print(f"   Total Samples:       {len(results['all_labels'])}")
    print(f"{'='*60}")

    if results['head_accuracies']:
        print(f"\nIndividual Head Accuracy (no TTA):")
        for head, acc in sorted(results['head_accuracies'].items()):
            print(f"   {head:<15} {acc:.2f}%")

    if hasattr(model, 'get_contribution_weights'):
        print(f"\nContribution Weights: {model.get_contribution_weights()}")
    if hasattr(model, 'get_ensemble_weights'):
        print(f"Ensemble Weights: {model.get_ensemble_weights()}")

    class_accuracies = print_per_class_accuracy(results['all_labels'], results['all_preds'], idx_to_class)

    target_names = [idx_to_class[i] for i in range(num_classes)]
    print(f"\n{'='*70}")
    print(f"{'Classification Report':^70}")
    print(f"{'='*70}")
    print(classification_report(results['all_labels'], results['all_preds'],
                                target_names=target_names, digits=4))

    if args.analyze_samples > 0:
        analyze_misclassified(results['all_labels'], results['all_preds'],
                              results['all_paths'], idx_to_class, top_n=args.analyze_samples)

    # ========================
    # Visualizations
    # ========================
    print(f"\n{'='*60}")
    print(f"Generating visualizations -> {args.output_dir}/")
    print(f"{'='*60}")

    plot_06_confusion_matrix(
        results['all_labels'], results['all_preds'], idx_to_class,
        os.path.join(args.output_dir, '06_ucf101_confusion.png')
    )
    plot_07_f1score_per_class(
        results['all_labels'], results['all_preds'], num_classes,
        os.path.join(args.output_dir, '07_ucf101_f1score.png')
    )
    plot_08_class_performance_detailed(
        results['all_labels'], results['all_preds'], idx_to_class, num_classes,
        os.path.join(args.output_dir, '08_ucf101_class_performance_detailed.png')
    )
    plot_09_sample_distribution(
        results['all_labels'], num_classes,
        os.path.join(args.output_dir, '09_ucf101_distribution_of_samples_per_class.png')
    )
    ece = plot_10_confidence_analysis(
        results['all_labels'], results['all_preds'], results['all_probs'],
        idx_to_class, num_classes,
        os.path.join(args.output_dir, '10_ucf101_prediction_confidence_analysis.png')
    )
    print(f"\n   ECE (Expected Calibration Error): {ece:.4f}")

    # ========================
    # Save Figure Data as CSV
    # ========================
    print(f"\nSaving figure data as CSV...")
    save_figure_data_csv(
        results['all_labels'], results['all_preds'], results['all_probs'],
        idx_to_class, num_classes, args.output_dir
    )

    # ========================
    # Save predictions JSON
    # ========================
    if args.save_predictions:
        predictions = []
        for i in range(len(results['all_labels'])):
            predictions.append({
                'path': results['all_paths'][i],
                'true_label': int(results['all_labels'][i]),
                'true_class': idx_to_class[results['all_labels'][i]],
                'pred_label': int(results['all_preds'][i]),
                'pred_class': idx_to_class[results['all_preds'][i]],
                'correct': results['all_labels'][i] == results['all_preds'][i],
                'confidence': float(np.max(results['all_probs'][i])),
                'top5_probs': {
                    idx_to_class[j]: float(results['all_probs'][i][j])
                    for j in np.argsort(results['all_probs'][i])[-5:][::-1]
                }
            })
        pred_path = os.path.join(args.output_dir, 'predictions.json')
        with open(pred_path, 'w', encoding='utf-8') as f:
            json.dump({
                'checkpoint': args.checkpoint,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'evaluation_mode': mode,
                'top1_accuracy': results['top1_acc'],
                'top5_accuracy': results['top5_acc'],
                'f1_macro': float(results['f1_macro']),
                'f1_weighted': float(results['f1_weighted']),
                'ece': float(ece),
                'total_samples': len(results['all_labels']),
                'predictions': predictions,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nPredictions saved: {pred_path}")

    # ========================
    # Save summary text
    # ========================
    summary_path = os.path.join(args.output_dir, 'test_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"STCA-ViT Test Results - UCF-101\n")
        f.write(f"{'='*60}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Evaluation Mode: {mode}\n")
        f.write(f"Test Split: group {args.train_group_max + 1}~25\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Top-1 Accuracy:      {results['top1_acc']:.2f}%\n")
        f.write(f"Top-5 Accuracy:      {results['top5_acc']:.2f}%\n")
        f.write(f"F1 Score (Macro):    {results['f1_macro']:.4f}\n")
        f.write(f"F1 Score (Weighted): {results['f1_weighted']:.4f}\n")
        f.write(f"ECE:                 {ece:.4f}\n")
        f.write(f"Total Test Samples:  {len(results['all_labels'])}\n\n")
        f.write(f"Per-Class Accuracy:\n")
        for i in range(num_classes):
            f.write(f"   {idx_to_class[i]:<30} {class_accuracies[i]:.2f}%\n")
    print(f"Summary saved: {summary_path}")

    print(f"\n{'='*60}")
    print(f"All done! Results saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
