"""CycleDACS smoke tests with a tiny registered backbone."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import BACKBONES, build_train_model
from mmseg.utils.mmcv_shim.config import Config


@BACKBONES.register_module(force=True)
class CycleDAFormerToyBackbone(nn.Module):
    def __init__(self, out_channels=(4, 8, 16, 32), **kwargs):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(3, channels, kernel_size=1)
            for channels in out_channels
        ])

    def forward(self, x):
        h, w = x.shape[2:]
        feats = []
        for i, proj in enumerate(self.proj):
            scale = 2 ** i
            resized = F.interpolate(
                x,
                size=(max(h // scale, 1), max(w // scale, 1)),
                mode='bilinear',
                align_corners=False)
            feats.append(proj(resized))
        return feats


def _toy_cfg(
        tmp_path,
        *,
        adversarial=False,
        inference_mode='target2source',
        feature_distance=False):
    channels = [4, 8, 16, 32]
    return Config(
        dict(
            model=dict(
                type='EncoderDecoder',
                pretrained=None,
                backbone=dict(type='CycleDAFormerToyBackbone',
                              out_channels=channels),
                decode_head=dict(
                    type='DAFormerHead',
                    in_channels=channels,
                    in_index=[0, 1, 2, 3],
                    channels=8,
                    dropout_ratio=0.0,
                    num_classes=3,
                    norm_cfg=None,
                    align_corners=False,
                    decoder_params=dict(
                        embed_dims=8,
                        embed_cfg=dict(
                            type='conv',
                            kernel_size=1,
                            act_cfg=dict(type='ReLU'),
                            norm_cfg=None),
                        embed_neck_cfg='same_as_embed_cfg',
                        fusion_cfg=dict(
                            type='conv',
                            kernel_size=1,
                            act_cfg=dict(type='ReLU'),
                            norm_cfg=None),
                    ),
                    loss_decode=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0)),
                train_cfg=dict(work_dir=str(tmp_path)),
                test_cfg=dict(mode='whole')),
            uda=dict(
                type='CycleDACS',
                alpha=0.99,
                pseudo_threshold=0.5,
                pseudo_weight_ignore_top=0,
                pseudo_weight_ignore_bottom=0,
                imnet_feature_dist_lambda=0.01 if feature_distance else 0,
                imnet_feature_dist_classes=None,
                imnet_feature_dist_scale_min_ratio=None,
                mix='class',
                blur=False,
                color_jitter_strength=0.0,
                color_jitter_probability=1.0,
                debug_img_interval=10**9,
                print_grad_magnitude=False,
                cycle_feature_indices=[0, 1, 2, 3],
                lambda_cycle_feat=0.1,
                lambda_source_cycle_seg=0.1,
                lambda_source_trans_seg=0.1,
                lambda_target_consistency=0.1,
                enable_feature_adversarial=adversarial,
                lambda_feature_adv=0.05 if adversarial else 0.0,
                feature_disc_base_channels=4,
                inference_mode=inference_mode),
            runner=dict(type='IterBasedRunner', max_iters=2),
        ))


def _batch(device, h=64, w=64, num_classes=3):
    meta = dict(
        img_shape=(h, w, 3),
        ori_shape=(h, w, 3),
        pad_shape=(h, w, 3),
        scale_factor=1.0,
        flip=False,
        flip_direction=None,
        img_norm_cfg=dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True))
    return dict(
        img=torch.randn(1, 3, h, w, device=device),
        img_metas=[copy.deepcopy(meta)],
        gt_semantic_seg=torch.randint(
            0, num_classes, (1, 1, h, w), device=device),
        target_img=torch.randn(1, 3, h, w, device=device),
        target_img_metas=[copy.deepcopy(meta)],
    )


def test_cycle_dacs_train_step_logs_cycle_losses(tmp_path):
    model = build_train_model(_toy_cfg(tmp_path))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    outputs = model.train_step(_batch(torch.device('cpu')), optimizer)
    log_vars = outputs['log_vars']
    assert 'loss_cycle_feat_s' in log_vars
    assert 'loss_cycle_feat_t' in log_vars
    assert 'src_cycle.loss_seg' in log_vars
    assert 'src_trans.loss_seg' in log_vars
    assert 'loss_target_consistency' in log_vars


def test_cycle_dacs_train_step_with_feature_adversarial(tmp_path):
    model = build_train_model(_toy_cfg(tmp_path, adversarial=True))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    outputs = model.train_step(_batch(torch.device('cpu')), optimizer)
    log_vars = outputs['log_vars']
    assert 'loss_feat_adv_g' in log_vars
    assert 'loss_feat_adv_d' in log_vars


def test_cycle_dacs_train_step_with_imnet_feature_distance(tmp_path):
    model = build_train_model(_toy_cfg(tmp_path, feature_distance=True))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    outputs = model.train_step(_batch(torch.device('cpu')), optimizer)
    log_vars = outputs['log_vars']
    assert 'src.loss_imnet_feat_dist' in log_vars
    assert 'loss_cycle_feat_s' in log_vars


def test_cycle_dacs_target2source_and_original_inference_modes(tmp_path):
    img = torch.randn(1, 3, 64, 64)
    img_metas = [
        dict(
            ori_shape=(64, 64, 3),
            img_shape=(64, 64, 3),
            pad_shape=(64, 64, 3),
            scale_factor=1.0,
            flip=False,
            flip_direction=None,
        )
    ]
    for mode in ['target2source', 'original']:
        model = build_train_model(_toy_cfg(tmp_path, inference_mode=mode))
        model.eval()
        with torch.no_grad():
            logits = model.encode_decode(img, img_metas)
        assert logits.shape == (1, 3, 64, 64)
