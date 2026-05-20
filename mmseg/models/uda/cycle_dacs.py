# ---------------------------------------------------------------
# Feature-cycle extension for DAFormer/DACS.
# ---------------------------------------------------------------

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath
from torch import nn
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA
from mmseg.models.uda.cycle_losses import (
    confidence_masked_consistency_loss,
    cycle_feature_loss,
    discriminator_loss,
    generator_adversarial_loss,
)
from mmseg.models.uda.cycle_modules import (
    FeatureDiscriminator,
    ResidualFeatureTranslator,
    set_requires_grad,
)
from mmseg.models.uda.dacs import DACS, calc_grad_magnitude
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.ops import resize


@UDA.register_module()
class CycleDACS(DACS):
    """DACS with bidirectional feature translators and cycle losses."""

    def __init__(self, **cfg):
        super(CycleDACS, self).__init__(**cfg)
        model_cfg = cfg['model']
        decode_head_cfg = model_cfg['decode_head']
        feature_channels = cfg.get(
            'cycle_feature_channels',
            decode_head_cfg.get('in_channels'))
        if feature_channels is None:
            raise ValueError(
                'CycleDACS requires cycle_feature_channels or '
                'model.decode_head.in_channels.')
        self.cycle_feature_channels = [int(c) for c in feature_channels]
        self.cycle_feature_indices = tuple(
            int(i) for i in cfg.get(
                'cycle_feature_indices',
                list(range(len(self.cycle_feature_channels)))))
        if any(i < 0 or i >= len(self.cycle_feature_channels)
               for i in self.cycle_feature_indices):
            raise ValueError(
                'cycle_feature_indices must refer to valid feature levels.')

        hidden_channels = cfg.get('translator_hidden_channels', None)
        self.source2target = nn.ModuleList([
            ResidualFeatureTranslator(
                c, self._hidden_channels_for_level(hidden_channels, i, c))
            for i, c in enumerate(self.cycle_feature_channels)
        ])
        self.target2source = nn.ModuleList([
            ResidualFeatureTranslator(
                c, self._hidden_channels_for_level(hidden_channels, i, c))
            for i, c in enumerate(self.cycle_feature_channels)
        ])

        self.lambda_cycle_feat = float(cfg.get('lambda_cycle_feat', 1.0))
        self.lambda_source_cycle_seg = float(
            cfg.get('lambda_source_cycle_seg', 1.0))
        self.lambda_source_trans_seg = float(
            cfg.get('lambda_source_trans_seg', 0.0))
        self.lambda_target_consistency = float(
            cfg.get('lambda_target_consistency', 1.0))
        self.lambda_feature_adv = float(cfg.get('lambda_feature_adv', 0.0))

        self.cycle_feature_loss_type = cfg.get('cycle_feature_loss_type', 'l1')
        self.normalize_cycle_features = bool(
            cfg.get('normalize_cycle_features', True))
        self.target_consistency_loss_type = cfg.get(
            'target_consistency_loss_type', 'kl')
        self.target_consistency_detach_teacher = bool(
            cfg.get('target_consistency_detach_teacher', True))
        self.target_consistency_teacher_no_grad = bool(
            cfg.get('target_consistency_teacher_no_grad',
                    self.target_consistency_detach_teacher))
        target_threshold = cfg.get('target_consistency_threshold', None)
        self.target_consistency_threshold = (
            self.pseudo_threshold if target_threshold is None
            else float(target_threshold))

        self.feature_adv_loss_type = cfg.get('feature_adv_loss_type', 'bce')
        self.enable_feature_adversarial = bool(
            cfg.get('enable_feature_adversarial',
                    self.lambda_feature_adv > 0))
        self.inference_mode = cfg.get('inference_mode', 'target2source')
        if self.inference_mode not in ['target2source', 'original']:
            raise ValueError(
                "CycleDACS inference_mode must be 'target2source' or "
                f"'original', got {self.inference_mode!r}.")

        if self.enable_feature_adversarial:
            disc_base_channels = int(cfg.get('feature_disc_base_channels', 64))
            self.source_discriminators = nn.ModuleList([
                FeatureDiscriminator(c, disc_base_channels)
                for c in self.cycle_feature_channels
            ])
            self.target_discriminators = nn.ModuleList([
                FeatureDiscriminator(c, disc_base_channels)
                for c in self.cycle_feature_channels
            ])
        else:
            self.source_discriminators = None
            self.target_discriminators = None

    @staticmethod
    def _hidden_channels_for_level(hidden_channels, level, channels):
        if hidden_channels is None:
            return channels
        if isinstance(hidden_channels, (list, tuple)):
            return int(hidden_channels[level])
        return int(hidden_channels)

    def _has_cycle_generator_loss(self):
        return any(weight > 0 for weight in [
            self.lambda_cycle_feat,
            self.lambda_source_cycle_seg,
            self.lambda_source_trans_seg,
            self.lambda_target_consistency,
            self.lambda_feature_adv,
        ])

    def _translate_features(self, features, translators):
        translated = list(features)
        for idx in self.cycle_feature_indices:
            translated[idx] = translators[idx](features[idx])
        return translated

    def _decode_features(self, features):
        return self.get_model().decode_head.forward(features)

    def _seg_losses_from_features(self, features, gt_semantic_seg, prefix):
        logits = self._decode_features(features)
        losses = self.get_model().decode_head.losses(logits, gt_semantic_seg)
        return add_prefix(losses, prefix)

    def _zero_like_loss(self, device):
        return torch.tensor(0.0, device=device)

    def _compute_cycle_generator_losses(self, src_feat, target_img,
                                        gt_semantic_seg):
        dev = target_img.device
        losses = {}

        target_feat = self.get_model().extract_feat(target_img)
        src_to_target = self._translate_features(src_feat, self.source2target)
        src_cycle = self._translate_features(src_to_target, self.target2source)
        target_to_source = self._translate_features(target_feat,
                                                    self.target2source)
        target_cycle = self._translate_features(target_to_source,
                                                self.source2target)

        if self.lambda_cycle_feat > 0:
            src_cycle_loss = cycle_feature_loss(
                src_feat,
                src_cycle,
                self.cycle_feature_indices,
                loss_type=self.cycle_feature_loss_type,
                normalize=self.normalize_cycle_features)
            target_cycle_loss = cycle_feature_loss(
                target_feat,
                target_cycle,
                self.cycle_feature_indices,
                loss_type=self.cycle_feature_loss_type,
                normalize=self.normalize_cycle_features)
            losses['loss_cycle_feat_s'] = (
                self.lambda_cycle_feat * src_cycle_loss)
            losses['loss_cycle_feat_t'] = (
                self.lambda_cycle_feat * target_cycle_loss)

        if self.lambda_source_cycle_seg > 0:
            src_cycle_losses = self._seg_losses_from_features(
                src_cycle, gt_semantic_seg, 'src_cycle')
            src_cycle_losses['src_cycle.loss_seg'] = (
                self.lambda_source_cycle_seg *
                src_cycle_losses['src_cycle.loss_seg'])
            losses.update(src_cycle_losses)

        if self.lambda_source_trans_seg > 0:
            src_trans_losses = self._seg_losses_from_features(
                src_to_target, gt_semantic_seg, 'src_trans')
            src_trans_losses['src_trans.loss_seg'] = (
                self.lambda_source_trans_seg *
                src_trans_losses['src_trans.loss_seg'])
            losses.update(src_trans_losses)

        if self.lambda_target_consistency > 0:
            if self.target_consistency_teacher_no_grad:
                with torch.no_grad():
                    target_logits = self._decode_features(target_feat)
            else:
                target_logits = self._decode_features(target_feat)
            target_cycle_logits = self._decode_features(target_cycle)
            losses['loss_target_consistency'] = (
                self.lambda_target_consistency *
                confidence_masked_consistency_loss(
                    target_logits,
                    target_cycle_logits,
                    threshold=self.target_consistency_threshold,
                    loss_type=self.target_consistency_loss_type,
                    detach_teacher=self.target_consistency_detach_teacher))

        if self.lambda_feature_adv > 0:
            if not self.enable_feature_adversarial:
                raise RuntimeError(
                    'lambda_feature_adv > 0 requires '
                    'enable_feature_adversarial=True.')
            set_requires_grad(self.source_discriminators, False)
            set_requires_grad(self.target_discriminators, False)
            losses['loss_feat_adv_g'] = (
                self.lambda_feature_adv *
                self._compute_feature_adv_generator_loss(
                    src_to_target, target_to_source, dev))
            set_requires_grad(self.source_discriminators, True)
            set_requires_grad(self.target_discriminators, True)

        return losses, dict(
            src_to_target=src_to_target,
            src_cycle=src_cycle,
            target_feat=target_feat,
            target_to_source=target_to_source,
            target_cycle=target_cycle,
        )

    def _compute_feature_adv_generator_loss(self, src_to_target,
                                            target_to_source, device):
        loss = self._zero_like_loss(device)
        count = 0
        for idx in self.cycle_feature_indices:
            pred_target = self.target_discriminators[idx](src_to_target[idx])
            pred_source = self.source_discriminators[idx](
                target_to_source[idx])
            loss = loss + generator_adversarial_loss(
                pred_target, self.feature_adv_loss_type)
            loss = loss + generator_adversarial_loss(
                pred_source, self.feature_adv_loss_type)
            count += 2
        return loss / max(count, 1)

    def _compute_feature_adv_discriminator_loss(self, src_feat, cycle_outputs,
                                                device):
        if not self.enable_feature_adversarial or self.lambda_feature_adv <= 0:
            return self._zero_like_loss(device)
        loss = self._zero_like_loss(device)
        count = 0
        target_feat = cycle_outputs['target_feat']
        src_to_target = cycle_outputs['src_to_target']
        target_to_source = cycle_outputs['target_to_source']
        for idx in self.cycle_feature_indices:
            pred_real_target = self.target_discriminators[idx](
                target_feat[idx].detach())
            pred_fake_target = self.target_discriminators[idx](
                src_to_target[idx].detach())
            pred_real_source = self.source_discriminators[idx](
                src_feat[idx].detach())
            pred_fake_source = self.source_discriminators[idx](
                target_to_source[idx].detach())
            loss = loss + discriminator_loss(
                pred_real_target,
                pred_fake_target,
                self.feature_adv_loss_type)
            loss = loss + discriminator_loss(
                pred_real_source,
                pred_fake_source,
                self.feature_adv_loss_type)
            count += 2
        return self.lambda_feature_adv * loss / max(count, 1)

    def _run_cycle_losses(self, src_feat, target_img, gt_semantic_seg):
        if not self._has_cycle_generator_loss():
            return {}, None

        cycle_losses, cycle_outputs = self._compute_cycle_generator_losses(
            src_feat, target_img, gt_semantic_seg)
        cycle_loss, cycle_log = self._parse_losses(cycle_losses)
        cycle_log.pop('loss', None)
        cycle_loss.backward()

        disc_log = {}
        if self.lambda_feature_adv > 0:
            disc_loss_raw = self._compute_feature_adv_discriminator_loss(
                src_feat, cycle_outputs, target_img.device)
            disc_loss, disc_log = self._parse_losses(
                {'loss_feat_adv_d': disc_loss_raw})
            disc_log.pop('loss', None)
            disc_loss.backward()

        cycle_log.update(disc_log)
        return cycle_log, cycle_outputs

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training."""
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device
        cycle_enabled = self._has_cycle_generator_loss()

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()

        if self.local_iter > 0:
            self._update_ema(self.local_iter)

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist or cycle_enabled)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            from mmseg.utils import mmcv_compat as mmcv
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward(retain_graph=cycle_enabled)
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                from mmseg.utils import mmcv_compat as mmcv
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        cycle_log, _ = self._run_cycle_losses(
            src_feat, target_img, gt_semantic_seg)
        log_vars.update(cycle_log)

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_ema_model().encode_decode(
            target_img, target_img_metas)

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)

        if self.psweight_ignore_top > 0:
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars

    def encode_decode(self, img, img_metas):
        if self.inference_mode == 'original':
            return self.get_model().encode_decode(img, img_metas)
        features = self.get_model().extract_feat(img)
        translated = self._translate_features(features, self.target2source)
        out = self._decode_features(translated)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.get_model().align_corners)
        return out

    def _whole_inference(self, img, img_meta, rescale):
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.get_model().align_corners,
                warning=False)
        return seg_logit

    def _slide_inference(self, img, img_meta, rescale):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(
                    crop_seg_logit,
                    (int(x1), int(preds.shape[3] - x2), int(y1),
                     int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.get_model().align_corners,
                warning=False)
        return preds

    def inference(self, img, img_meta, rescale):
        if self.inference_mode == 'original':
            return self.get_model().inference(img, img_meta, rescale)
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self._slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self._whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))
        return output

    def simple_test(self, img, img_meta, rescale=True):
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            return seg_pred.unsqueeze(0)
        return list(seg_pred.cpu().numpy())

    def aug_test(self, imgs, img_metas, rescale=True):
        assert rescale
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            seg_logit += self.inference(imgs[i], img_metas[i], rescale)
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        return list(seg_pred.cpu().numpy())
