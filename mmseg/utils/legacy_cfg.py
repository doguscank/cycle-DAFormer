# Legacy config fixes for published DAFormer JSON checkpoints.


def update_legacy_cfg(cfg):
    """Normalize configs saved from older training runs."""
    if hasattr(cfg.data.test.pipeline[1], "get"):
        img_scale = cfg.data.test.pipeline[1].get("img_scale")
        if img_scale is not None and not isinstance(img_scale, tuple):
            cfg.data.test.pipeline[1]["img_scale"] = tuple(img_scale)
    decode_head = cfg.model.decode_head
    if decode_head.get("type") == "UniHead":
        decode_head["type"] = "DAFormerHead"
        fusion = decode_head.get("decoder_params", {}).get("fusion_cfg", {})
        if isinstance(fusion, dict):
            fusion.pop("fusion", None)
    backbone = cfg.model.backbone
    if isinstance(backbone, dict):
        backbone.pop("ema_drop_path_rate", None)
    elif hasattr(backbone, "pop"):
        backbone.pop("ema_drop_path_rate", None)
    return cfg
