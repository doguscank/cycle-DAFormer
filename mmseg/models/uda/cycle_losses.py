import torch
import torch.nn.functional as F


def normalize_feature_map(x, eps=1e-6):
    return F.normalize(x, p=2, dim=1, eps=eps)


def feature_distance(pred, target, loss_type='l1', normalize=True):
    if normalize:
        pred = normalize_feature_map(pred)
        target = normalize_feature_map(target)
    loss_type = str(loss_type).lower()
    if loss_type == 'l1':
        return F.l1_loss(pred, target)
    if loss_type == 'mse':
        return F.mse_loss(pred, target)
    if loss_type == 'cosine':
        pred_flat = pred.flatten(start_dim=1)
        target_flat = target.flatten(start_dim=1)
        cosine = F.cosine_similarity(pred_flat, target_flat, dim=1, eps=1e-6)
        return 1 - cosine.mean()
    raise ValueError(f'Unsupported cycle feature loss type: {loss_type}')


def cycle_feature_loss(
        features,
        cycled_features,
        feature_indices,
        loss_type='l1',
        normalize=True):
    loss = None
    for idx in feature_indices:
        term = feature_distance(
            cycled_features[idx],
            features[idx],
            loss_type=loss_type,
            normalize=normalize)
        loss = term if loss is None else loss + term
    if loss is None:
        device = features[0].device if features else torch.device('cpu')
        return torch.tensor(0.0, device=device)
    return loss / max(len(feature_indices), 1)


def confidence_masked_consistency_loss(
        teacher_logits,
        student_logits,
        threshold,
        loss_type='kl',
        detach_teacher=True):
    if student_logits.shape[2:] != teacher_logits.shape[2:]:
        student_logits = F.interpolate(
            student_logits,
            size=teacher_logits.shape[2:],
            mode='bilinear',
            align_corners=False)

    teacher_prob = F.softmax(teacher_logits, dim=1)
    if detach_teacher:
        teacher_prob = teacher_prob.detach()
    confidence, teacher_label = teacher_prob.max(dim=1)
    mask = confidence.ge(float(threshold)).float()
    denom = mask.sum().clamp(min=1.0)

    loss_type = str(loss_type).lower()
    if loss_type == 'kl':
        per_pixel = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            teacher_prob,
            reduction='none').sum(dim=1)
    elif loss_type == 'ce':
        per_pixel = F.cross_entropy(
            student_logits,
            teacher_label,
            reduction='none',
            ignore_index=255)
    else:
        raise ValueError(f'Unsupported target consistency loss type: {loss_type}')
    return (per_pixel * mask).sum() / denom


def discriminator_loss(real_logits, fake_logits, loss_type='bce'):
    loss_type = str(loss_type).lower()
    if loss_type == 'mse':
        return 0.5 * (
            F.mse_loss(real_logits, torch.ones_like(real_logits)) +
            F.mse_loss(fake_logits, torch.zeros_like(fake_logits)))
    if loss_type == 'bce':
        return 0.5 * (
            F.binary_cross_entropy_with_logits(
                real_logits, torch.ones_like(real_logits)) +
            F.binary_cross_entropy_with_logits(
                fake_logits, torch.zeros_like(fake_logits)))
    raise ValueError(f'Unsupported adversarial loss type: {loss_type}')


def generator_adversarial_loss(fake_logits, loss_type='bce'):
    loss_type = str(loss_type).lower()
    if loss_type == 'mse':
        return F.mse_loss(fake_logits, torch.ones_like(fake_logits))
    if loss_type == 'bce':
        return F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits))
    raise ValueError(f'Unsupported adversarial loss type: {loss_type}')
