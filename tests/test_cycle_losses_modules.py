from __future__ import annotations

import torch

from mmseg.models.uda.cycle_losses import (
    confidence_masked_consistency_loss,
    cycle_feature_loss,
)
from mmseg.models.uda.cycle_modules import (
    FeatureDiscriminator,
    ResidualFeatureTranslator,
)


def test_residual_translator_preserves_shape_and_backpropagates():
    translator = ResidualFeatureTranslator(8)
    x = torch.randn(2, 8, 16, 16, requires_grad=True)
    y = translator(x)
    assert y.shape == x.shape
    y.mean().backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in translator.parameters())


def test_feature_discriminator_outputs_patch_logits_and_backpropagates():
    discriminator = FeatureDiscriminator(8, base_channels=4)
    x = torch.randn(2, 8, 8, 8, requires_grad=True)
    logits = discriminator(x)
    assert logits.shape[:2] == (2, 1)
    logits.mean().backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in discriminator.parameters())


def test_cycle_feature_loss_zero_for_identical_features():
    features = [
        torch.randn(1, 4, 16, 16),
        torch.randn(1, 8, 8, 8),
    ]
    loss = cycle_feature_loss(
        features,
        [f.clone() for f in features],
        feature_indices=[0, 1],
        loss_type='l1',
        normalize=True)
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_target_consistency_empty_confidence_mask_is_zero_with_grad_path():
    teacher_logits = torch.zeros(1, 3, 8, 8)
    student_logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    loss = confidence_masked_consistency_loss(
        teacher_logits,
        student_logits,
        threshold=0.99,
        loss_type='kl',
        detach_teacher=True)
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-6)
    loss.backward()
    assert student_logits.grad is not None
    assert torch.count_nonzero(student_logits.grad) == 0
