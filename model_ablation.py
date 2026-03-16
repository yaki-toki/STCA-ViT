"""
STCA-ViT Ablation Model Variants
=================================

This module defines ablation study variants of the STCA-ViT architecture for video
action recognition. Each variant selectively enables or disables individual modules
(STCA, MGU, PersonAttention) to measure their contribution to overall performance.

Architecture overview:
    x (B, 3, T, H, W)
    ├── VideoMAE ViT-Base → b_back (B, 768)
    ├── STCA Module → f_stca (B, module_embed_dim)
    ├── MGU Module → f_mgu (B, module_embed_dim) + temporal_info (for WTAL)
    └── Person Attention Module → f_person (B, module_embed_dim)
             ↓
        4-way Residual Contribution Fusion → f_fused (B, 768)
             ↓
        ├── Multi-Head Ensemble Classifier → logits (B, num_classes)
        └── WTAL Head (use_wtal=True, optional) →
              ├── t_cam: (B, num_classes, T)      [per-frame class activation]
              ├── action_scores: (B, T)            [per-frame fg/bg probability]
              └── mil_logits: (B, num_classes)     [MIL-aggregated clip-level prediction]

Ablation configurations A0–A11 are defined in ABLATION_CONFIGS and can be
instantiated via stca_vit_ablation().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# HuggingFace transformers for VideoMAE
try:
    from transformers import VideoMAEModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. VideoMAE backbone will not be available.")

# YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Person attention will be disabled.")


# ===========================
# Constants
# ===========================
YOLO_PERSON_CLASS_ID = 0


# ===========================
# Gradient Reversal Layer
# ===========================
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


# ===========================
# Orthogonal Loss
# ===========================
class OrthogonalLoss(nn.Module):
    def __init__(self, lambda_decorr=0.005):
        super().__init__()
        self.lambda_decorr = lambda_decorr

    def _cosine_loss(self, feat_a, feat_b):
        a = F.normalize(feat_a, dim=1)
        b = F.normalize(feat_b, dim=1)
        cos_sim = torch.sum(a * b, dim=1)
        return (cos_sim ** 2).mean()

    def _decorrelation_loss(self, feat_a, feat_b):
        B = feat_a.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=feat_a.device)
        a = (feat_a - feat_a.mean(dim=0)) / (feat_a.std(dim=0) + 1e-5)
        b = (feat_b - feat_b.mean(dim=0)) / (feat_b.std(dim=0) + 1e-5)
        cross_corr = (a.T @ b) / B
        return (cross_corr ** 2).mean()

    def forward(self, b_back, f_stca=None, f_mgu=None, f_person=None):
        # Build list of active features (disabled modules pass None during ablation)
        active_feats = [b_back]
        if f_stca is not None:
            active_feats.append(f_stca)
        if f_mgu is not None:
            active_feats.append(f_mgu)
        if f_person is not None:
            active_feats.append(f_person)

        # Decorrelation is unnecessary with fewer than 2 features
        if len(active_feats) < 2:
            return torch.tensor(0.0, device=b_back.device)

        cos_loss = torch.tensor(0.0, device=b_back.device)
        decorr_loss = torch.tensor(0.0, device=b_back.device)
        for i in range(len(active_feats)):
            for j in range(i + 1, len(active_feats)):
                cos_loss = cos_loss + self._cosine_loss(active_feats[i], active_feats[j])
                decorr_loss = decorr_loss + self._decorrelation_loss(active_feats[i], active_feats[j])
        return cos_loss + self.lambda_decorr * decorr_loss


# ===========================
# WTAL Loss (Weakly-Supervised Temporal Action Localization)
# ===========================
class WTALLoss(nn.Module):
    """
    Weakly-supervised Temporal Action Localization Loss

    Five loss components:
    1. MIL Classification: clip-level CE loss on MIL-pooled logits
    2. Pseudo-Label: BCE on action_scores using MGU temp_weights as pseudo-labels (self-supervised)
    3. Sparsity: encourages action to occupy ~50% of the clip
    4. Smoothness: minimizes variation in action_score between adjacent frames
    5. Background Entropy: encourages uniform class distribution over background frames

    Total = lambda_mil * MIL + lambda_pseudo * Pseudo + lambda_sparsity * Sparsity
          + lambda_smooth * Smoothness + lambda_bg * BG_Entropy
    """
    def __init__(self, num_classes,
                 lambda_mil=1.0, lambda_pseudo=0.5, lambda_sparsity=0.1,
                 lambda_smooth=0.1, lambda_bg=0.05, target_sparsity=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_mil = lambda_mil
        self.lambda_pseudo = lambda_pseudo
        self.lambda_sparsity = lambda_sparsity
        self.lambda_smooth = lambda_smooth
        self.lambda_bg = lambda_bg
        self.target_sparsity = target_sparsity

    def forward(self, tal_outputs, labels, skip_mil=False):
        """
        Args:
            tal_outputs: dict from TemporalActionLocalizationHead
                - t_cam: (B, num_classes, T) per-frame class activation
                - action_scores: (B, T) per-frame fg/bg probability
                - temporal_attention: (B, T) MIL attention weights
                - mil_logits: (B, num_classes) MIL-aggregated clip-level prediction
                - pseudo_targets: (B, T) pseudo-labels derived from MGU w_temp
            labels: (B,) clip-level class labels
            skip_mil: skip MIL CE loss when using MixUp/CutMix
        Returns:
            total_loss: scalar
            loss_dict: dict of individual loss values (for logging)
        """
        t_cam = tal_outputs['t_cam']              # (B, C, T)
        action_scores = tal_outputs['action_scores']  # (B, T)
        mil_logits = tal_outputs['mil_logits']    # (B, C)
        pseudo_targets = tal_outputs['pseudo_targets']  # (B, T)

        loss_dict = {}
        total_loss = torch.tensor(0.0, device=t_cam.device)

        # 1. MIL Classification Loss
        if not skip_mil and labels is not None:
            mil_loss = F.cross_entropy(mil_logits, labels)
            total_loss = total_loss + self.lambda_mil * mil_loss
            loss_dict['mil_loss'] = mil_loss.item()

        # 2. Pseudo-Label Loss (self-supervised: MGU w_temp → action_scores)
        # BCE does not support float16 under AMP autocast, so disable it here
        with torch.cuda.amp.autocast(enabled=False):
            pseudo_loss = F.binary_cross_entropy(
                action_scores.float(), pseudo_targets.detach().float(), reduction='mean'
            )
        total_loss = total_loss + self.lambda_pseudo * pseudo_loss
        loss_dict['pseudo_loss'] = pseudo_loss.item()

        # 3. Sparsity Loss (mean of action_scores should be close to target_sparsity)
        mean_activation = action_scores.mean(dim=1)  # (B,)
        sparsity_loss = (mean_activation - self.target_sparsity).pow(2).mean()
        total_loss = total_loss + self.lambda_sparsity * sparsity_loss
        loss_dict['sparsity_loss'] = sparsity_loss.item()

        # 4. Smoothness Loss (minimize difference between adjacent frame action_scores)
        if action_scores.shape[1] > 1:
            smooth_loss = (action_scores[:, 1:] - action_scores[:, :-1]).pow(2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=t_cam.device)
        total_loss = total_loss + self.lambda_smooth * smooth_loss
        loss_dict['smooth_loss'] = smooth_loss.item()

        # 5. Background Entropy Loss (encourage uniform class distribution over background frames)
        bg_mask = (action_scores < 0.5).float()  # (B, T)
        if bg_mask.sum() > 0:
            # t_cam softmax → class distribution per frame
            t_cam_prob = F.softmax(t_cam, dim=1)  # (B, C, T)
            # class distribution entropy over background frames
            bg_mask_expanded = bg_mask.unsqueeze(1)  # (B, 1, T)
            bg_probs = (t_cam_prob * bg_mask_expanded).sum(dim=2)  # (B, C)
            bg_probs = bg_probs / (bg_mask.sum(dim=1, keepdim=True) + 1e-6)  # normalize
            bg_probs = bg_probs + 1e-6  # prevent log(0)
            # Maximize entropy → minimize negative entropy
            bg_entropy = -(bg_probs * bg_probs.log()).sum(dim=1).mean()
            max_entropy = np.log(self.num_classes)
            bg_entropy_loss = max_entropy - bg_entropy  # 0 means perfectly uniform
        else:
            bg_entropy_loss = torch.tensor(0.0, device=t_cam.device)
        total_loss = total_loss + self.lambda_bg * bg_entropy_loss
        loss_dict['bg_entropy_loss'] = bg_entropy_loss.item()

        loss_dict['total_wtal_loss'] = total_loss.item()
        return total_loss, loss_dict


# ===========================
# DropPath (Stochastic Depth)
# ===========================
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ===========================
# VideoMAE ViT-Base Backbone
# ===========================
class VideoMAEBackbone(nn.Module):
    """
    HuggingFace VideoMAE ViT-Base Backbone

    - Kinetics-400 finetuned: "MCG-NJU/videomae-base-finetuned-kinetics"
    - Self-supervised pretrained: "MCG-NJU/videomae-base"
    - Input: (B, C=3, T=16, H, W) — arbitrary resolution is accepted
      (internally resized to 224x224 via bilinear interpolation)
    - Output: b_back (B, 768)

    Note: VideoMAE internally requires 224x224 input. When a different resolution
    is provided, the input is automatically resized to 224x224 before processing.
    Other modules (STCA, MGU, PersonAttention) operate on the original resolution,
    so smaller inputs such as 112x112 can provide a speed benefit for those modules.
    """
    def __init__(self, pretrained_name="MCG-NJU/videomae-base-finetuned-kinetics",
                 freeze_layers=0, use_gradient_checkpointing=False):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for VideoMAE backbone. "
                              "Install with: pip install transformers")

        print(f"Loading VideoMAE backbone: {pretrained_name}")
        self.model = VideoMAEModel.from_pretrained(pretrained_name)
        self.embed_dim = self.model.config.hidden_size  # 768
        self.num_frames = self.model.config.num_frames  # 16

        # Gradient checkpointing (saves memory, allows larger batch sizes)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print(f"  Gradient checkpointing enabled")

        # Freeze early layers if specified
        if freeze_layers > 0:
            # Freeze embeddings (requires_grad handled in forward when using gradient checkpointing)
            for param in self.model.embeddings.parameters():
                param.requires_grad = False
            # Freeze first N encoder layers
            for i in range(min(freeze_layers, len(self.model.encoder.layer))):
                for param in self.model.encoder.layer[i].parameters():
                    param.requires_grad = False
            print(f"  Frozen: embeddings + {freeze_layers} encoder layers")

        print(f"  VideoMAE ViT-Base loaded (hidden_size={self.embed_dim})")

    def forward(self, x):
        """
        Args:
            x: (B, C, T, H, W) - input video (arbitrary resolution; resized to 224x224 internally)
        Returns:
            b_back: (B, 768) - avg pool over patch tokens
        """
        B, C, T, H, W = x.shape

        # VideoMAE requires 224x224 input — resize automatically for other resolutions
        if H != 224 or W != 224:
            # (B, C, T, H, W) → (B*T, C, H, W) → resize → (B, C, T, 224, 224)
            x = x.reshape(B * T, C, H, W)
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            x = x.reshape(B, C, T, 224, 224)

        # HuggingFace VideoMAE expects (B, T, C, H, W)
        pixel_values = x.permute(0, 2, 1, 3, 4).contiguous()

        # With gradient checkpointing and frozen layers, requires_grad=True is needed on input
        # (frozen embeddings produce outputs with no grad, causing checkpoint warnings)
        if self.use_gradient_checkpointing and self.training and not pixel_values.requires_grad:
            pixel_values = pixel_values.detach().requires_grad_(True)

        # Forward through VideoMAE
        outputs = self.model(pixel_values=pixel_values)

        # last_hidden_state: (B, num_patches, 768)
        # VideoMAE doesn't have a CLS token - use avg pooling over all patch tokens
        hidden_states = outputs.last_hidden_state  # (B, num_patches, 768)
        b_back = hidden_states.mean(dim=1)  # (B, 768)

        return b_back


# ===========================
# STCA Module (Spatio-Temporal Cross-Attention, 768-d)
# ===========================
class STCAModule(nn.Module):
    """
    STCA Module (Spatio-Temporal Cross-Attention) (embed_dim=768)

    Independently extracts spatial and temporal tokens from raw video x,
    then applies bidirectional cross-attention between them.
    """
    def __init__(self, in_channels=3, embed_dim=768, num_heads=12, dropout=0.1, drop_path=0.1,
                 num_cross_layers=2, num_temporal_tokens=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cross_layers = num_cross_layers
        self.num_temporal_tokens = num_temporal_tokens

        # Spatial token extraction: (B, 3, T, H, W) → spatial tokens
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(1, 7, 7), stride=(1, 4, 4), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.GELU(),
            nn.Conv3d(256, embed_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU()
        )

        # Temporal token extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 4, 4), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.GELU(),
            nn.Conv3d(256, embed_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU()
        )

        # Learnable Temporal Tokens
        self.learnable_temporal_tokens = nn.Parameter(torch.randn(1, num_temporal_tokens, embed_dim) * 0.02)
        self.temporal_token_proj = nn.Linear(embed_dim, embed_dim)

        # Multi-layer Bidirectional Cross-Attention
        self.bidirectional_cross_attn_layers = nn.ModuleList()
        for _ in range(num_cross_layers):
            self.bidirectional_cross_attn_layers.append(
                BidirectionalCrossAttentionLayer(embed_dim=embed_dim, num_heads=num_heads,
                                    dropout=dropout, drop_path=drop_path)
            )

        # Pooling-based Fusion
        self.spatial_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(start_dim=1))
        self.temporal_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(start_dim=1))

        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=-1)
        )

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, external_feat=None):
        B = x.shape[0]

        # Spatial tokens
        spatial_feat = self.spatial_conv(x)  # (B, D, T, H', W')
        spatial_feat = spatial_feat.mean(dim=2)  # temporal average → (B, D, H', W')
        S_tok = spatial_feat.flatten(2).transpose(1, 2)  # (B, N_s, D)

        # Temporal tokens
        temporal_feat = self.temporal_conv(x)  # (B, D, T', H', W')
        temporal_feat = temporal_feat.flatten(3).mean(dim=3)  # spatial average → (B, D, T')
        T_tok = temporal_feat.transpose(1, 2)  # (B, T', D)

        # Learnable temporal tokens
        L_T = self.learnable_temporal_tokens.expand(B, -1, -1)
        temporal_tokens_proj = self.temporal_token_proj(T_tok)
        attn_weights = torch.bmm(L_T, temporal_tokens_proj.transpose(1, 2))
        attn_weights = attn_weights / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        L_T = L_T + torch.bmm(attn_weights, temporal_tokens_proj)
        T_tok = torch.cat([T_tok, L_T], dim=1)

        # Multi-layer Bidirectional Cross-Attention
        for layer in self.bidirectional_cross_attn_layers:
            S_tok, T_tok = layer(S_tok, T_tok)

        # Pooling-based Fusion
        spatial_pooled = self.spatial_pool(S_tok.transpose(1, 2))
        temporal_pooled = self.temporal_pool(T_tok.transpose(1, 2))

        combined = torch.cat([spatial_pooled, temporal_pooled], dim=-1)
        weights = self.fusion_gate(combined)
        f_stca = weights[:, 0:1] * spatial_pooled + weights[:, 1:2] * temporal_pooled

        f_stca = torch.clamp(f_stca, min=-1e4, max=1e4)
        if torch.isnan(f_stca).any():
            f_stca = torch.nan_to_num(f_stca, nan=0.0)

        return self.out_proj(f_stca)


class BidirectionalCrossAttentionLayer(nn.Module):
    """Bidirectional Cross-Attention Layer + FFN"""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1, drop_path=0.1):
        super().__init__()

        self.s_norm = nn.LayerNorm(embed_dim)
        self.t_norm = nn.LayerNorm(embed_dim)

        self.s2t_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.t2s_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.s2t_norm = nn.LayerNorm(embed_dim)
        self.t2s_norm = nn.LayerNorm(embed_dim)

        mlp_hidden = embed_dim * 4
        self.s_ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim), nn.Dropout(dropout)
        )
        self.s_ffn_norm = nn.LayerNorm(embed_dim)

        self.t_ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim), nn.Dropout(dropout)
        )
        self.t_ffn_norm = nn.LayerNorm(embed_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, S_tok, T_tok):
        s_normed = self.s_norm(S_tok)
        t_normed = self.t_norm(T_tok)

        s_normed = torch.clamp(s_normed, min=-1e4, max=1e4)
        t_normed = torch.clamp(t_normed, min=-1e4, max=1e4)

        # S→T: temporal attends to spatial
        s2t_out, _ = self.s2t_attn(t_normed, s_normed, s_normed)
        s2t_out = torch.clamp(s2t_out, min=-1e4, max=1e4)
        T_tok = T_tok + self.drop_path(s2t_out)
        T_tok = self.s2t_norm(T_tok)

        # T→S: spatial attends to temporal
        t2s_out, _ = self.t2s_attn(s_normed, t_normed, t_normed)
        t2s_out = torch.clamp(t2s_out, min=-1e4, max=1e4)
        S_tok = S_tok + self.drop_path(t2s_out)
        S_tok = self.t2s_norm(S_tok)

        # FFN
        S_tok = S_tok + self.drop_path(self.s_ffn(self.s_ffn_norm(S_tok)))
        T_tok = T_tok + self.drop_path(self.t_ffn(self.t_ffn_norm(T_tok)))

        return S_tok, T_tok


# ===========================
# MGU Module (768-d)
# ===========================
class MGUModule(nn.Module):
    """
    Motion Gated Unit - Optical Flow Style (embed_dim=768)

    Independently extracts motion features from raw video x.
    Uses Sobel gradients and bidirectional temporal differences,
    distinct from the VideoMAE backbone approach.
    """
    def __init__(self, in_channels=3, embed_dim=768, dropout=0.3,
                 num_temporal_scales=4, num_heads=12, use_motion_token=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_temporal_scales = num_temporal_scales
        self.num_heads = num_heads
        self.use_motion_token = use_motion_token

        self.temporal_scales = [1, 2, 4, 8][:num_temporal_scales]

        # Sobel-like gradient filters
        self.gradient_conv = nn.Conv3d(
            in_channels, in_channels * 2,
            kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False, groups=in_channels
        )
        self._init_gradient_weights()

        # Motion magnitude and direction
        self.motion_magnitude_conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(32), nn.GELU(),
        )
        self.motion_direction_conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(32), nn.GELU(),
        )

        # Multi-scale bidirectional motion
        self.bidirectional_motion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels * 2, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
                nn.BatchNorm3d(32), nn.GELU(),
            ) for _ in self.temporal_scales
        ])

        # Scale fusion
        total_motion_channels = 32 + 32 + 32 * num_temporal_scales
        self.scale_attention = nn.Sequential(
            nn.Conv3d(total_motion_channels, num_temporal_scales + 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.scale_fusion = nn.Sequential(
            nn.Conv3d(total_motion_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm3d(64), nn.GELU(),
        )

        # Motion encoder → embed_dim
        self.motion_encoder = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128), nn.GELU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256), nn.GELU(),
            nn.Conv3d(256, embed_dim, kernel_size=(1, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim), nn.GELU()
        )

        # Motion boundary detection
        self.boundary_conv = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // 4, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim // 4), nn.GELU(),
            nn.Conv3d(embed_dim // 4, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Temporal attention
        self.temporal_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Flatten(2),
        )
        self.temporal_attn_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4), nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
        )

        # Learnable Motion Token
        if use_motion_token:
            self.motion_token = nn.Parameter(torch.randn(1, embed_dim, 1, 1, 1) * 0.02)
            self.token_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Output
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.out_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )

    def _init_gradient_weights(self):
        with torch.no_grad():
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            for c in range(self.gradient_conv.weight.shape[1]):
                self.gradient_conv.weight[c * 2, 0, 0] = sobel_x
                self.gradient_conv.weight[c * 2 + 1, 0, 0] = sobel_y

    def compute_optical_flow_features(self, x):
        B, C, T, H, W = x.shape
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        gradients = self.gradient_conv(x_reshaped.unsqueeze(2)).squeeze(2)
        gradients = gradients.reshape(B, T, C * 2, H, W).permute(0, 2, 1, 3, 4)

        grad_magnitude = torch.sqrt(gradients[:, ::2] ** 2 + gradients[:, 1::2] ** 2 + 1e-6)
        mag_diff = grad_magnitude[:, :, 1:] - grad_magnitude[:, :, :-1]
        mag_diff = F.pad(mag_diff, (0, 0, 0, 0, 0, 1), mode='replicate')
        magnitude_feat = self.motion_magnitude_conv(mag_diff)

        grad_diff = gradients[:, :, 1:] - gradients[:, :, :-1]
        grad_diff = F.pad(grad_diff, (0, 0, 0, 0, 0, 1), mode='replicate')
        direction_feat = self.motion_direction_conv(grad_diff)

        return magnitude_feat, direction_feat

    def compute_bidirectional_motion(self, x):
        B, C, T, H, W = x.shape
        motion_features = []
        for scale, conv in zip(self.temporal_scales, self.bidirectional_motion_convs):
            if scale >= T:
                forward_diff = x[:, :, -1:] - x[:, :, :1]
                forward_diff = forward_diff.expand(-1, -1, T, -1, -1)
                backward_diff = -forward_diff
            else:
                forward_diff = x[:, :, scale:] - x[:, :, :-scale]
                forward_diff = F.pad(forward_diff, (0, 0, 0, 0, 0, scale), mode='replicate')
                backward_diff = x[:, :, :-scale] - x[:, :, scale:]
                backward_diff = F.pad(backward_diff, (0, 0, 0, 0, scale, 0), mode='replicate')
            bidirectional = torch.cat([forward_diff, backward_diff], dim=1)
            motion_features.append(conv(bidirectional))
        return motion_features

    def forward(self, x, external_feat=None, return_temporal=False):
        """
        Args:
            x: (B, C, T, H, W) raw video
            external_feat: unused (kept for API compatibility)
            return_temporal: if True, also return temporal features needed by the WTAL head
        Returns:
            return_temporal=False: out (B, embed_dim) — standard output
            return_temporal=True:  (out, temporal_info_dict) — output plus temporal features
        """
        B, C, T, H, W = x.shape

        magnitude_feat, direction_feat = self.compute_optical_flow_features(x)
        bidirectional_feats = self.compute_bidirectional_motion(x)

        all_feats = [magnitude_feat, direction_feat] + bidirectional_feats
        concat_feat = torch.cat(all_feats, dim=1)

        scale_weights = self.scale_attention(concat_feat)
        weighted_feats = []
        for i, feat in enumerate(all_feats):
            weighted_feats.append(feat * scale_weights[:, i:i+1])
        weighted_concat = torch.cat(weighted_feats, dim=1)

        fused = self.scale_fusion(weighted_concat)
        M_enc = self.motion_encoder(fused)

        # Save M_enc before boundary application for WTAL head
        B_mask = self.boundary_conv(M_enc)
        if return_temporal:
            M_enc_raw = M_enc  # preserve feature before boundary gating
        M_enc = M_enc * (1 + B_mask)

        temp_pooled = self.temporal_attn(M_enc).transpose(1, 2)
        w_temp = F.softmax(self.temporal_attn_proj(temp_pooled), dim=1)
        w_temp = w_temp.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        M_enc = M_enc * w_temp

        if self.use_motion_token:
            _, _, T_feat, H_feat, W_feat = M_enc.shape
            m_tok = self.motion_token.expand(B, -1, T_feat, H_feat, W_feat)
            mot_flat = M_enc.flatten(2).transpose(1, 2)
            token_flat = m_tok.flatten(2).transpose(1, 2)
            mot_enhanced, _ = self.token_attn(mot_flat, token_flat, token_flat)
            M_enc = mot_enhanced.transpose(1, 2).view(B, self.embed_dim, T_feat, H_feat, W_feat)

        mot_pooled = self.pool(M_enc).flatten(1)
        out = self.out_proj[1:](mot_pooled)  # Skip Flatten

        if return_temporal:
            temporal_info = {
                'M_enc': M_enc_raw,      # (B, D, T', H', W') — before boundary gating
                'B_mask': B_mask,  # (B, 1, T', H', W')
                'w_temp': w_temp,    # (B, 1, T', 1, 1)
            }
            return out, temporal_info

        return out


# ===========================
# Person Attention Module (768-d)
# ===========================
class PersonAttentionModule(nn.Module):
    """
    YOLO-based Person-focused feature extraction (embed_dim=768)
    """
    def __init__(self, in_channels=3, embed_dim=768, dropout=0.3,
                 yolo_model_path='yolov8n.pt', topk=3, score_threshold=0.3,
                 img_mean=(0.43216, 0.394666, 0.37645), img_std=(0.22803, 0.22145, 0.216989)):
        super().__init__()
        self.embed_dim = embed_dim
        self.topk = topk
        self.score_threshold = score_threshold

        # Feature extraction with higher capacity
        self.feature_conv = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(64), nn.GELU(),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(128), nn.GELU(),
            nn.Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(256), nn.GELU(),
            nn.Conv3d(256, embed_dim, kernel_size=(1, 1, 1), bias=False),
            nn.BatchNorm3d(embed_dim), nn.GELU()
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.out_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )

        # YOLO - Lazy initialization
        self.yolo_available = YOLO_AVAILABLE
        self.yolo_model_path = yolo_model_path
        self.yolo_device = None
        self._yolo_initialized = False

        self.register_buffer("img_mean", torch.tensor(img_mean).view(1, 3, 1, 1, 1))
        self.register_buffer("img_std", torch.tensor(img_std).view(1, 3, 1, 1, 1))

    def _lazy_init_yolo(self):
        if self._yolo_initialized:
            return
        if self.yolo_available and YOLO_AVAILABLE:
            try:
                self.yolo_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                yolo_model = YOLO(self.yolo_model_path)
                yolo_model.to(self.yolo_device)
                yolo_model.eval()
                object.__setattr__(self, "_yolo_model", yolo_model)
                self._yolo_initialized = True
            except Exception as e:
                print(f"Warning: Failed to load YOLO: {e}")
                self.yolo_available = False
                self._yolo_initialized = True
        else:
            self._yolo_initialized = True

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_yolo_model' in state:
            del state['_yolo_model']
        state['_yolo_initialized'] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._yolo_initialized = False

    @torch.no_grad()
    def _get_person_attention(self, x):
        B, C, T, H, W = x.shape
        device = x.device

        if not self._yolo_initialized:
            self._lazy_init_yolo()

        if not self.yolo_available or not hasattr(self, '_yolo_model'):
            return torch.ones(B, 1, T, H, W, device=device)

        x_denorm = x * self.img_std.to(device) + self.img_mean.to(device)
        x_denorm = x_denorm.clamp(0, 1)

        attn_maps = []
        for b in range(B):
            indices = torch.linspace(0, T-1, min(self.topk, T)).long()
            frame_maps = []
            for t in indices:
                frame = x_denorm[b, :, t]
                frame_bgr = frame[[2, 1, 0]]
                frame_np = (frame_bgr * 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
                try:
                    results = self._yolo_model.predict(frame_np, verbose=False, device=self.yolo_device)[0]
                    heatmap = torch.zeros((H, W), device=device)
                    if len(results.boxes) > 0:
                        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                            if int(cls.item()) == YOLO_PERSON_CLASS_ID and conf.item() >= self.score_threshold:
                                x1, y1, x2, y2 = box.int()
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                sigma = max((x2 - x1).item(), (y2 - y1).item()) / 3.0
                                sigma = max(sigma, 5.0)
                                yy, xx = torch.meshgrid(torch.arange(H, device=device),
                                                        torch.arange(W, device=device), indexing='ij')
                                g = torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
                                heatmap = torch.maximum(heatmap, g)
                except Exception:
                    heatmap = torch.zeros((H, W), device=device)
                frame_maps.append(heatmap)

            avg_map = torch.stack(frame_maps).mean(0)
            avg_map = 0.5 + avg_map
            attn_maps.append(avg_map.unsqueeze(0).unsqueeze(0).expand(1, 1, T, H, W))

        return torch.cat(attn_maps, dim=0)

    def forward(self, x):
        if self.training:
            A_person = self._get_person_attention(x.detach())
        else:
            A_person = self._get_person_attention(x)

        F_person = self.feature_conv(x)
        attn_resized = F.interpolate(A_person, size=F_person.shape[2:], mode='trilinear', align_corners=False)
        F_person = F_person * attn_resized

        f_person = self.pool(F_person)
        f_person = self.out_proj(f_person)
        return f_person


# ===========================
# Temporal Action Localization Head (WTAL)
# ===========================
class TemporalActionLocalizationHead(nn.Module):
    """
    Weakly-Supervised Temporal Action Localization Head

    Takes temporal features from the MGU module and outputs per-frame action probabilities.

    Flow:
        M_enc (B, D, T, H, W)
            ↓ spatial_pool → (B, D, T)
            ↓ temporal_conv (depthwise + pointwise 1D) → (B, D, T) [residual]
            ↓ boundary_guidance (B_mask → gate) → (B, D, T)
            ├── t_cam_conv → (B, num_classes, T)  [T-CAM: per-frame class activation]
            ├── attention_branch → (B, T)  [action_scores: fg/bg probability]
            └── MIL aggregation: T-CAM × attention → (B, num_classes) [mil_logits]
    """
    def __init__(self, embed_dim=256, num_classes=51, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Spatial pooling: (B, D, T, H, W) → (B, D, T)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # Temporal convolution (depthwise separable 1D) with residual
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.temporal_norm = nn.LayerNorm(embed_dim)

        # Boundary guidance: B_mask spatial pool → temporal gate
        self.boundary_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # T-CAM: temporal class activation map
        self.t_cam_conv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(embed_dim, num_classes, kernel_size=1, bias=True),
        )

        # Attention branch: action_scores (fg/bg probability per frame)
        self.attention_branch = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim // 4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv1d(embed_dim // 4, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, M_enc, B_mask, w_temp):
        """
        Args:
            M_enc: (B, D, T, H, W) — MGU motion feature (before boundary gating)
            B_mask: (B, 1, T, H, W) — MGU boundary detection mask
            w_temp: (B, 1, T, 1, 1) — MGU temporal attention weights
        Returns:
            dict with:
                t_cam: (B, num_classes, T)
                action_scores: (B, T)
                temporal_attention: (B, T)
                mil_logits: (B, num_classes)
                pseudo_targets: (B, T) — w_temp converted to pseudo-labels
        """
        B, D, T_feat, H, W = M_enc.shape

        # 1. Spatial pooling → (B, D, T)
        feat = self.spatial_pool(M_enc).squeeze(-1).squeeze(-1)  # (B, D, T)

        # 2. Temporal convolution with residual
        feat_conv = self.temporal_conv(feat)  # (B, D, T)
        feat = feat + feat_conv  # residual
        feat = self.temporal_norm(feat.transpose(1, 2)).transpose(1, 2)  # (B, D, T)

        # 3. Boundary guidance: B_mask → spatial avg → gate
        boundary_gate = self.boundary_pool(B_mask).squeeze(-1).squeeze(-1)  # (B, 1, T)
        feat = feat * (1 + boundary_gate)  # boundary-enhanced feature

        # 4. T-CAM: per-frame class activation
        t_cam = self.t_cam_conv(feat)  # (B, num_classes, T)

        # 5. Action scores: fg/bg probability per frame
        action_scores = self.attention_branch(feat).squeeze(1)  # (B, T)

        # 6. MIL aggregation: attention-weighted T-CAM → clip-level logits
        # temporal_attention = normalized action_scores for MIL pooling
        temporal_attention = action_scores / (action_scores.sum(dim=1, keepdim=True) + 1e-6)  # (B, T)
        mil_logits = torch.bmm(
            t_cam,  # (B, C, T)
            temporal_attention.unsqueeze(-1)  # (B, T, 1)
        ).squeeze(-1)  # (B, C)

        # 7. Pseudo-targets from MGU w_temp (self-supervision signal)
        pseudo_targets = w_temp.squeeze(1).squeeze(-1).squeeze(-1)  # (B, T)
        # Normalize to [0, 1] range
        pseudo_min = pseudo_targets.min(dim=1, keepdim=True)[0]
        pseudo_max = pseudo_targets.max(dim=1, keepdim=True)[0]
        pseudo_targets = (pseudo_targets - pseudo_min) / (pseudo_max - pseudo_min + 1e-6)

        return {
            't_cam': t_cam,
            'action_scores': action_scores,
            'temporal_attention': temporal_attention,
            'mil_logits': mil_logits,
            'pseudo_targets': pseudo_targets,
        }


# ===========================
# Residual Contribution Fusion (768-d)
# ===========================
class ResidualContributionFusion(nn.Module):
    """
    Residual Contribution Fusion (backbone_dim=768, embed_dim=768)

    When module_embed_dim differs from embed_dim, a linear projection aligns dimensions.
    final = backbone + alpha*(stca_residual) + beta*(mgu_residual) + gamma*(person_residual)
    """
    def __init__(self, backbone_dim=768, embed_dim=768, module_embed_dim=None, num_heads=12, dropout=0.1,
                 use_gradient_reversal=True, grl_lambda=0.1):
        super().__init__()
        if module_embed_dim is None:
            module_embed_dim = embed_dim
        self.embed_dim = embed_dim
        self.module_embed_dim = module_embed_dim
        self.use_gradient_reversal = use_gradient_reversal

        if use_gradient_reversal:
            self.stca_grl = GradientReversalLayer(lambda_=grl_lambda)
            self.mgu_grl = GradientReversalLayer(lambda_=grl_lambda)
            self.person_grl = GradientReversalLayer(lambda_=grl_lambda)

        # Backbone projection (backbone_dim → embed_dim)
        self.backbone_proj = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Module projections (module_embed_dim → embed_dim)
        self.stca_proj = nn.Sequential(nn.Linear(module_embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.mgu_proj = nn.Sequential(nn.Linear(module_embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.person_proj = nn.Sequential(nn.Linear(module_embed_dim, embed_dim), nn.LayerNorm(embed_dim))

        # Learnable Contribution Weights
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.beta = nn.Parameter(torch.tensor(0.3))
        self.gamma = nn.Parameter(torch.tensor(0.1))

        # Residual extractors
        self.stca_residual = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim)
        )
        self.mgu_residual = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim)
        )
        self.person_residual = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim)
        )

        # Final Refinement
        self.fusion_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def set_grl_lambda(self, lambda_):
        if self.use_gradient_reversal:
            self.stca_grl.set_lambda(lambda_)
            self.mgu_grl.set_lambda(lambda_)
            self.person_grl.set_lambda(lambda_)

    def forward(self, b_back, f_stca, f_mgu, f_person):
        b_proj = self.backbone_proj(b_back)

        if self.use_gradient_reversal:
            c_proj = self.stca_proj(self.stca_grl(f_stca))
            m_proj = self.mgu_proj(self.mgu_grl(f_mgu))
            p_proj = self.person_proj(self.person_grl(f_person))
        else:
            c_proj = self.stca_proj(f_stca)
            m_proj = self.mgu_proj(f_mgu)
            p_proj = self.person_proj(f_person)

        b_proj = torch.clamp(b_proj, min=-1e4, max=1e4)
        c_proj = torch.clamp(c_proj, min=-1e4, max=1e4)
        m_proj = torch.clamp(m_proj, min=-1e4, max=1e4)
        p_proj = torch.clamp(p_proj, min=-1e4, max=1e4)

        b_det = b_proj.detach()

        r_stca = self.stca_residual(torch.cat([c_proj, b_det], dim=1))
        r_mgu = self.mgu_residual(torch.cat([m_proj, b_det], dim=1))
        r_person = self.person_residual(torch.cat([p_proj, b_det], dim=1))

        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        gamma = torch.sigmoid(self.gamma)

        f_fused = b_proj + alpha * r_stca + beta * r_mgu + gamma * r_person

        Z = torch.stack([b_proj, f_fused, r_stca, r_mgu, r_person], dim=1)

        attn_out, _ = self.fusion_attn(Z, Z, Z)
        attn_out = torch.clamp(attn_out, min=-1e4, max=1e4)
        Z = self.attn_norm(Z + attn_out)
        Z = self.ffn_norm(Z + self.ffn(Z))

        out = Z[:, 1]
        if torch.isnan(out).any():
            out = torch.nan_to_num(out, nan=0.0)

        return self.out_proj(out)

    def get_contribution_weights(self):
        return {
            'alpha (stca)': torch.sigmoid(self.alpha).item(),
            'beta (mgu)': torch.sigmoid(self.beta).item(),
            'gamma (person)': torch.sigmoid(self.gamma).item()
        }


# ===========================
# Multi-Head Ensemble Classifier (768-d)
# ===========================
class MultiHeadEnsembleClassifier(nn.Module):
    def __init__(self, backbone_dim=768, embed_dim=768, module_embed_dim=None, num_classes=400, dropout=0.3):
        super().__init__()
        if module_embed_dim is None:
            module_embed_dim = embed_dim
        self.num_classes = num_classes
        D_h = 384

        self.backbone_head = nn.Sequential(
            nn.Linear(backbone_dim, D_h), nn.LayerNorm(D_h), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(D_h, num_classes)
        )
        # Module heads: module_embed_dim input (detached raw module features)
        self.stca_head = nn.Sequential(
            nn.Linear(module_embed_dim, D_h), nn.LayerNorm(D_h), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(D_h, num_classes)
        )
        self.mgu_head = nn.Sequential(
            nn.Linear(module_embed_dim, D_h), nn.LayerNorm(D_h), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(D_h, num_classes)
        )
        self.person_head = nn.Sequential(
            nn.Linear(module_embed_dim, D_h), nn.LayerNorm(D_h), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(D_h, num_classes)
        )
        # Fused head: embed_dim input (fusion output)
        self.fused_head = nn.Sequential(
            nn.Linear(embed_dim, D_h), nn.LayerNorm(D_h), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(D_h, num_classes)
        )

        self.ensemble_weights = nn.Parameter(torch.tensor([0.2, 0.2, 0.2, 0.1, 0.3]))

        # Dynamic ensemble
        self.use_dynamic_ensemble = True
        total_dim = backbone_dim + module_embed_dim * 3 + embed_dim  # backbone + 3 modules + fused
        self.dynamic_weight_net = nn.Sequential(
            nn.Linear(total_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 5), nn.Softmax(dim=-1)
        )

    def forward(self, b_back, f_stca, f_mgu, f_person, f_fused,
                return_individual=False):
        backbone_logits = self.backbone_head(b_back)
        stca_logits = self.stca_head(f_stca)
        mgu_logits = self.mgu_head(f_mgu)
        person_logits = self.person_head(f_person)
        fused_logits = self.fused_head(f_fused)

        all_logits = torch.stack([
            backbone_logits, stca_logits, mgu_logits, person_logits, fused_logits
        ], dim=1)

        if self.use_dynamic_ensemble:
            all_feats = torch.cat([
                b_back, f_stca, f_mgu, f_person, f_fused
            ], dim=-1)
            weights = self.dynamic_weight_net(all_feats)
        else:
            weights = F.softmax(self.ensemble_weights, dim=0)
            weights = weights.unsqueeze(0).expand(b_back.size(0), -1)

        weights = weights.unsqueeze(-1)
        final_logits = (weights * all_logits).sum(dim=1)

        if return_individual:
            individual_logits = {
                'backbone': backbone_logits,
                'stca': stca_logits,
                'mgu': mgu_logits,
                'person': person_logits,
                'fused': fused_logits,
                'ensemble_weights': weights.squeeze(-1)
            }
            return final_logits, individual_logits

        return final_logits

    def get_ensemble_weights(self):
        if self.use_dynamic_ensemble:
            return "Dynamic (input-dependent)"
        return F.softmax(self.ensemble_weights, dim=0).detach().cpu().numpy()


# ===========================
# STCA-ViT Model
# ===========================
class STCAViT(nn.Module):
    """
    STCA-ViT - VideoMAE ViT-Base Backbone

    Architecture:
    - Backbone: VideoMAE ViT-Base (768-d, Kinetics pretrained)
    - STCA: Bidirectional spatial-temporal cross-attention (module_embed_dim)
    - MGU: Optical flow style motion features (module_embed_dim)
    - PersonAttention: YOLO-guided person features (module_embed_dim)
    - Fusion: Residual Contribution Fusion with GRL (projects module_embed_dim → embed_dim)
    - Classifier: Multi-Head Ensemble with dynamic weights
    - WTAL Head (optional): Weakly-supervised Temporal Action Localization

    Setting module_embed_dim smaller than embed_dim significantly reduces memory usage for
    auxiliary modules. Fusion and Classifier handle the projection from module_embed_dim
    to embed_dim internally.

    When use_wtal=True, the MGU temporal features are reused to output per-frame action
    probabilities. Pass compute_wtal=True to forward() to include tal_outputs in the result.
    """
    def __init__(self, num_classes=400, embed_dim=768, module_embed_dim=None, dropout=0.3, drop_path=0.2,
                 num_heads=12, module_num_heads=None, use_gradient_reversal=True, grl_lambda=0.1,
                 use_multi_head_ensemble=True,
                 videomae_name="MCG-NJU/videomae-base-finetuned-kinetics",
                 freeze_backbone_layers=0, use_gradient_checkpointing=False,
                 use_wtal=False,
                 # Ablation flags
                 backbone_only=False, disable_stca=False, disable_mgu=False, disable_person=False):
        super().__init__()

        # Default module_embed_dim to embed_dim when not specified
        if module_embed_dim is None:
            module_embed_dim = embed_dim
        if module_num_heads is None:
            module_num_heads = max(1, module_embed_dim // 64)  # e.g. 256 → 4 heads

        self.embed_dim = embed_dim
        self.module_embed_dim = module_embed_dim
        self.num_classes = num_classes
        self.use_multi_head_ensemble = use_multi_head_ensemble
        self.use_wtal = use_wtal

        # Ablation flags
        self.backbone_only = backbone_only
        self.disable_stca = disable_stca
        self.disable_mgu = disable_mgu
        self.disable_person = disable_person

        # ==================
        # 1. VideoMAE ViT-Base Backbone (always created)
        # ==================
        self.backbone = VideoMAEBackbone(
            pretrained_name=videomae_name,
            freeze_layers=freeze_backbone_layers,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        self.backbone_dim = self.backbone.embed_dim  # 768

        # ==================
        # A0: Backbone Only — use a simple linear classifier
        # ==================
        if self.backbone_only:
            self.simple_classifier = nn.Sequential(
                nn.Linear(self.backbone_dim, 384),
                nn.LayerNorm(384), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(384, num_classes)
            )
            return  # skip creation of remaining modules

        # ==================
        # 2. STCA Module (conditional)
        # ==================
        if not self.disable_stca:
            self.stca_module = STCAModule(
                in_channels=3,
                embed_dim=module_embed_dim,
                num_heads=module_num_heads,
                dropout=dropout,
                drop_path=drop_path,
                num_cross_layers=2,
                num_temporal_tokens=32
            )

        # ==================
        # 3. MGU Module (conditional)
        # ==================
        if not self.disable_mgu:
            self.mgu_module = MGUModule(
                in_channels=3,
                embed_dim=module_embed_dim,
                dropout=dropout,
                num_temporal_scales=3,
                num_heads=module_num_heads,
                use_motion_token=True
            )

        # ==================
        # 4. Person Attention Module (conditional)
        # ==================
        if not self.disable_person and YOLO_AVAILABLE:
            self.use_person_attention = True
            self.person_module = PersonAttentionModule(
                in_channels=3,
                embed_dim=module_embed_dim,
                dropout=dropout
            )
        else:
            self.use_person_attention = False

        # ==================
        # 5. Residual Contribution Fusion (module_embed_dim → embed_dim)
        # ==================
        self.fusion = ResidualContributionFusion(
            backbone_dim=self.backbone_dim,
            embed_dim=embed_dim,
            module_embed_dim=module_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_gradient_reversal=use_gradient_reversal,
            grl_lambda=grl_lambda
        )

        # Pin contribution weights of disabled modules close to zero
        with torch.no_grad():
            if self.disable_stca:
                self.fusion.alpha.fill_(-10.0)
                self.fusion.alpha.requires_grad = False
            if self.disable_mgu:
                self.fusion.beta.fill_(-10.0)
                self.fusion.beta.requires_grad = False
            if self.disable_person or not self.use_person_attention:
                self.fusion.gamma.fill_(-10.0)
                self.fusion.gamma.requires_grad = False

        # ==================
        # 6. Classifier
        # ==================
        if use_multi_head_ensemble:
            self.classifier = MultiHeadEnsembleClassifier(
                backbone_dim=self.backbone_dim,
                embed_dim=embed_dim,
                module_embed_dim=module_embed_dim,
                num_classes=num_classes,
                dropout=dropout
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, 384),
                nn.LayerNorm(384), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(384, num_classes)
            )

        self.orthogonal_loss = OrthogonalLoss()

        # ==================
        # 7. WTAL Head (optional, requires MGU)
        # ==================
        if use_wtal:
            if self.disable_mgu:
                print("  WARNING: WTAL requires MGU module. Disabling WTAL.")
                self.use_wtal = False
            else:
                self.tal_head = TemporalActionLocalizationHead(
                    embed_dim=module_embed_dim,
                    num_classes=num_classes,
                    dropout=dropout
                )
                self.wtal_loss_fn = WTALLoss(num_classes)
                print(f"  WTAL Head enabled (module_embed_dim={module_embed_dim}, num_classes={num_classes})")

    def set_grl_lambda(self, lambda_):
        if not self.backbone_only and hasattr(self, 'fusion'):
            self.fusion.set_grl_lambda(lambda_)

    def forward(self, x, return_all=False, compute_ortho_loss=False, compute_wtal=False):
        """
        Args:
            x: (B, 3, T, H, W) input video
            return_all: return all intermediate features as dict
            compute_ortho_loss: compute orthogonal regularization loss
            compute_wtal: compute WTAL temporal action localization outputs
        """
        B = x.shape[0]

        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        # A0: Backbone Only — simplified forward path
        if self.backbone_only:
            b_back = self.backbone(x)
            logits = self.simple_classifier(b_back)
            if return_all:
                return {'final': logits, 'b_back': b_back}
            return logits

        # 1. VideoMAE Backbone
        b_back = self.backbone(x)  # (B, backbone_dim=768)

        # 2. STCA Module (conditional)
        if not self.disable_stca:
            f_stca = self.stca_module(x)  # (B, module_embed_dim)
        else:
            f_stca = torch.zeros(B, self.module_embed_dim, device=x.device)

        # 3. MGU (conditional) — also return temporal features when WTAL is active
        if not self.disable_mgu:
            if self.use_wtal and compute_wtal:
                f_mgu, mgu_temporal_info = self.mgu_module(x, return_temporal=True)
            else:
                f_mgu = self.mgu_module(x)  # (B, module_embed_dim)
        else:
            f_mgu = torch.zeros(B, self.module_embed_dim, device=x.device)

        # 4. Person Attention (conditional)
        if not self.disable_person and self.use_person_attention:
            f_person = self.person_module(x)  # (B, module_embed_dim)
        else:
            f_person = torch.zeros(B, self.module_embed_dim, device=x.device)

        # 5. Fusion
        f_fused = self.fusion(b_back, f_stca, f_mgu, f_person)

        # 6. Classification
        if self.use_multi_head_ensemble:
            if return_all:
                logits, individual_logits = self.classifier(
                    b_back,
                    f_stca.detach(), f_mgu.detach(), f_person.detach(),
                    f_fused, return_individual=True
                )
            else:
                logits = self.classifier(
                    b_back,
                    f_stca.detach(), f_mgu.detach(), f_person.detach(),
                    f_fused, return_individual=False
                )
        else:
            logits = self.classifier(f_fused)

        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0)

        # 7. Orthogonal Loss (compare only active module feature pairs)
        ortho_loss = None
        if compute_ortho_loss:
            b_proj = self.fusion.backbone_proj(b_back)
            stca_proj_feat = self.fusion.stca_proj(f_stca) if not self.disable_stca else None
            mgu_proj_feat = self.fusion.mgu_proj(f_mgu) if not self.disable_mgu else None
            person_proj_feat = (self.fusion.person_proj(f_person)
                                if (not self.disable_person and self.use_person_attention) else None)
            ortho_loss = self.orthogonal_loss(
                b_proj.detach(),
                stca_proj_feat, mgu_proj_feat,
                person_proj_feat
            )

        # 8. WTAL Head
        tal_outputs = None
        if self.use_wtal and compute_wtal and not self.disable_mgu:
            tal_outputs = self.tal_head(
                mgu_temporal_info['M_enc'],
                mgu_temporal_info['B_mask'],
                mgu_temporal_info['w_temp']
            )

        # Return
        if return_all:
            outputs = {
                'final': logits,
                'b_back': b_back,
                'f_stca': f_stca,
                'f_mgu': f_mgu,
                'f_person': f_person,
                'f_fused': f_fused,
            }
            if self.use_multi_head_ensemble:
                outputs.update(individual_logits)
            if ortho_loss is not None:
                outputs['ortho_loss'] = ortho_loss
            if tal_outputs is not None:
                outputs['tal_outputs'] = tal_outputs
            return outputs

        if compute_ortho_loss:
            return logits, ortho_loss

        return logits

    def get_contribution_weights(self):
        if self.backbone_only:
            return None
        return self.fusion.get_contribution_weights()

    def get_ensemble_weights(self):
        if self.backbone_only or not self.use_multi_head_ensemble:
            return None
        return self.classifier.get_ensemble_weights()


# ===========================
# Factory Functions
# ===========================
def stca_vit(num_classes=400, **kwargs):
    """
    STCA-ViT with VideoMAE ViT-Base backbone
    embed_dim=768, num_heads=12

    module_embed_dim: dimension for auxiliary modules (STCA, MGU, PersonAttention).
                      Defaults to None, which matches embed_dim (768).
                      Setting to 256 reduces auxiliary module memory by ~75%.
    use_wtal: when True, enables the WTAL (Temporal Action Localization) head.
              Defaults to False, preserving identical behaviour to the base model.
    """
    kwargs.setdefault('dropout', 0.3)
    kwargs.setdefault('drop_path', 0.2)
    return STCAViT(
        num_classes=num_classes,
        embed_dim=768,
        num_heads=12,
        **kwargs
    )


# Alias for backward compatibility with training scripts
stca_vit_v2_5_parallel_ablation = stca_vit


# ===========================
# Ablation Study Configurations
# ===========================
ABLATION_CONFIGS = {
    'A0': {
        'name': 'Backbone Only',
        'description': 'VideoMAE backbone + simple linear classifier',
        'backbone_only': True,
        'disable_stca': True, 'disable_mgu': True, 'disable_person': True,
        'use_gradient_reversal': False, 'use_multi_head_ensemble': False, 'use_wtal': False,
    },
    'A1': {
        'name': 'Backbone + STCA',
        'description': 'Backbone with STCA module only',
        'backbone_only': False,
        'disable_stca': False, 'disable_mgu': True, 'disable_person': True,
        'use_gradient_reversal': True, 'use_multi_head_ensemble': True, 'use_wtal': False,
    },
    'A2': {
        'name': 'Backbone + MGU',
        'description': 'Backbone with MGU module only',
        'backbone_only': False,
        'disable_stca': True, 'disable_mgu': False, 'disable_person': True,
        'use_gradient_reversal': True, 'use_multi_head_ensemble': True, 'use_wtal': False,
    },
    'A3': {
        'name': 'Backbone + Person',
        'description': 'Backbone with Person Attention module only',
        'backbone_only': False,
        'disable_stca': True, 'disable_mgu': True, 'disable_person': False,
        'use_gradient_reversal': True, 'use_multi_head_ensemble': True, 'use_wtal': False,
    },
    'A4': {
        'name': 'Backbone + STCA + MGU',
        'description': 'Backbone with STCA and MGU, no Person',
        'backbone_only': False,
        'disable_stca': False, 'disable_mgu': False, 'disable_person': True,
        'use_gradient_reversal': True, 'use_multi_head_ensemble': True, 'use_wtal': False,
    },
    'A5': {
        'name': 'Backbone + STCA + Person',
        'description': 'Backbone with STCA and Person, no MGU',
        'backbone_only': False,
        'disable_stca': False, 'disable_mgu': True, 'disable_person': False,
        'use_gradient_reversal': True, 'use_multi_head_ensemble': True, 'use_wtal': False,
    },
    'A6': {
        'name': 'Backbone + MGU + Person',
        'description': 'Backbone with MGU and Person, no STCA',
        'backbone_only': False,
        'disable_stca': True, 'disable_mgu': False, 'disable_person': False,
        'use_gradient_reversal': True, 'use_multi_head_ensemble': True, 'use_wtal': False,
    },
    'A7': {
        'name': 'Full Model',
        'description': 'Complete STCA-ViT with all modules (baseline)',
        'backbone_only': False,
        'disable_stca': False, 'disable_mgu': False, 'disable_person': False,
        'use_gradient_reversal': True, 'use_multi_head_ensemble': True, 'use_wtal': False,
    },
    'A8': {
        'name': 'Full w/o Multi-Head',
        'description': 'Full model with simple fused classifier (no 5-head ensemble)',
        'backbone_only': False,
        'disable_stca': False, 'disable_mgu': False, 'disable_person': False,
        'use_gradient_reversal': True, 'use_multi_head_ensemble': False, 'use_wtal': False,
    },
    'A9': {
        'name': 'Full w/o Ortho Loss',
        'description': 'Full model without orthogonal decorrelation loss',
        'backbone_only': False,
        'disable_stca': False, 'disable_mgu': False, 'disable_person': False,
        'use_gradient_reversal': True, 'use_multi_head_ensemble': True, 'use_wtal': False,
        'disable_ortho_loss': True,
    },
    'A10': {
        'name': 'Full w/o GRL',
        'description': 'Full model without Gradient Reversal Layer',
        'backbone_only': False,
        'disable_stca': False, 'disable_mgu': False, 'disable_person': False,
        'use_gradient_reversal': False, 'use_multi_head_ensemble': True, 'use_wtal': False,
    },
    'A11': {
        'name': 'Full + WTAL',
        'description': 'Full model with WTAL temporal action localization head',
        'backbone_only': False,
        'disable_stca': False, 'disable_mgu': False, 'disable_person': False,
        'use_gradient_reversal': True, 'use_multi_head_ensemble': True, 'use_wtal': True,
    },
}


def stca_vit_ablation(num_classes=51, ablation_config='A7', **kwargs):
    """
    Factory function for ablation studies.

    Args:
        num_classes: number of output classes
        ablation_config: one of 'A0' through 'A11'
        **kwargs: additional arguments forwarded to STCAViT

    Returns:
        STCAViT instance with the specified ablation configuration applied
    """
    if ablation_config not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation config: {ablation_config}. "
                         f"Available: {list(ABLATION_CONFIGS.keys())}")

    config = ABLATION_CONFIGS[ablation_config]
    print(f"  Ablation: {ablation_config} — {config['name']}")
    print(f"  {config['description']}")

    # Extract model parameters from config (exclude name, description, disable_ortho_loss)
    model_keys = ['backbone_only', 'disable_stca', 'disable_mgu', 'disable_person',
                  'use_gradient_reversal', 'use_multi_head_ensemble', 'use_wtal']
    for key in model_keys:
        if key in config:
            kwargs.setdefault(key, config[key])

    kwargs.setdefault('dropout', 0.3)
    kwargs.setdefault('drop_path', 0.2)

    return STCAViT(
        num_classes=num_classes,
        embed_dim=768,
        num_heads=12,
        **kwargs
    )


# ===========================
# Test
# ===========================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\n" + "=" * 60)
    print("Test: STCA-ViT (VideoMAE ViT-Base, embed_dim=768)")
    print("=" * 60)

    model = stca_vit(
        num_classes=400,
        use_gradient_reversal=True,
        use_multi_head_ensemble=True
    ).to(device)

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")

    # Forward pass test
    dummy_input = torch.randn(2, 3, 16, 224, 224).to(device)
    print(f"\nInput shape: {dummy_input.shape}")

    with torch.no_grad():
        # Basic forward
        logits = model(dummy_input)
        print(f"Output logits shape: {logits.shape}")  # (2, 400)

        # With ortho loss
        logits, ortho_loss = model(dummy_input, compute_ortho_loss=True)
        print(f"Ortho loss: {ortho_loss.item():.6f}")

        # Full return
        outputs = model(dummy_input, return_all=True, compute_ortho_loss=True)
        print(f"Final logits: {outputs['final'].shape}")
        print(f"Backbone feat: {outputs['b_back'].shape}")
        print(f"STCA feat: {outputs['f_stca'].shape}")
        print(f"MGU feat: {outputs['f_mgu'].shape}")
        print(f"Person feat: {outputs['f_person'].shape}")
        print(f"Fused feat: {outputs['f_fused'].shape}")

        contrib = model.get_contribution_weights()
        print(f"Contribution weights: {contrib}")

    print("\nAll tests passed!")
