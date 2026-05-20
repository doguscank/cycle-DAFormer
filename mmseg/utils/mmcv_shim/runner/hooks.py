# Training and evaluation hooks.

from __future__ import annotations

import os
import os.path as osp
import warnings

import numpy as np
import torch
import torch.distributed as dist

from mmseg.utils.mmcv_shim.runner.dist import get_dist_info


class Hook:
    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_train_iter(self, runner):
        pass

    def after_train_iter(self, runner):
        pass

    def after_train_epoch(self, runner):
        pass

    def after_evaluate(self, runner, eval_res=None, results=None, dataset=None):
        pass


class OptimizerHook(Hook):
    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        # grads applied in train_step for UDA models
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(runner.model.parameters(), self.grad_clip)
        runner.optimizer.step()


class LrUpdaterHook(Hook):
    def __init__(
        self,
        policy="poly",
        warmup=None,
        warmup_iters=0,
        warmup_ratio=1e-6,
        power=1.0,
        min_lr=0.0,
        by_epoch=False,
        **kwargs,
    ):
        self.policy = policy
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.power = power
        self.min_lr = min_lr
        self.by_epoch = by_epoch

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        max_iters = runner.max_iters
        base_lr = runner.optimizer.param_groups[0].get("_base_lr")
        if base_lr is None:
            for g in runner.optimizer.param_groups:
                g["_base_lr"] = g["lr"]
            base_lr = runner.optimizer.param_groups[0]["_base_lr"]
        if self.warmup and cur_iter < self.warmup_iters:
            alpha = cur_iter / max(self.warmup_iters, 1)
            lr = base_lr * (self.warmup_ratio * (1 - alpha) + alpha)
        elif self.policy == "poly":
            progress = cur_iter / max(max_iters, 1)
            lr = base_lr * (1 - progress) ** self.power
            lr = max(lr, self.min_lr)
        else:
            lr = base_lr
        for g in runner.optimizer.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)


class TextLoggerHook(Hook):
    def __init__(self, interval=50, by_epoch=True):
        self.interval = interval
        self.by_epoch = by_epoch

    def after_train_iter(self, runner):
        if runner.iter % self.interval != 0:
            return
        runner.log_buffer.average(self.interval)
        log_str = f"Iter [{runner.iter}/{runner.max_iters}]"
        for k, v in runner.log_buffer.output.items():
            log_str += f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
        if runner.logger:
            runner.logger.info(log_str)


class WandbLoggerHook(Hook):
    def __init__(
        self,
        interval=50,
        by_epoch=True,
        init_kwargs=None,
        log_checkpoint=False,
        log_model=True,
        log_model_freq=None,
        log_eval_images=True,
        num_eval_images=None,
        log_debug_images=True,
        num_debug_images=None,
        commit=True,
    ):
        self.interval = interval
        self.by_epoch = by_epoch
        self.init_kwargs = init_kwargs or {}
        self.log_checkpoint = log_checkpoint
        self.log_model = log_model
        self.log_model_freq = log_model_freq
        self.log_eval_images = log_eval_images
        self.num_eval_images = num_eval_images
        self.log_debug_images = log_debug_images
        self.num_debug_images = num_debug_images
        self.commit = commit
        self.wandb = None
        self._logged_debug_images = set()

    def before_run(self, runner):
        if runner.rank != 0:
            return
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "WandbLoggerHook requires wandb. Install it with "
                "`pip install wandb` or remove the WandbLoggerHook config."
            ) from exc
        self.wandb = wandb
        init_kwargs = dict(self.init_kwargs)
        init_kwargs.setdefault("dir", runner.work_dir)
        init_kwargs.setdefault(
            "name", runner.meta.get("config_name") or runner.meta.get("exp_name"))
        init_kwargs.setdefault("config", self._sanitize_config(runner.meta))
        self.wandb.init(**init_kwargs)
        if self.log_model:
            model = runner.model.module if hasattr(runner.model, "module") else runner.model
            self.wandb.watch(
                model,
                log="all",
                log_freq=self.log_model_freq or self.interval,
                log_graph=False,
            )
            self.wandb.run.summary["model/parameter_count"] = sum(
                p.numel() for p in model.parameters())
            self.wandb.run.summary["model/trainable_parameter_count"] = sum(
                p.numel() for p in model.parameters() if p.requires_grad)

    def after_train_iter(self, runner):
        if runner.rank != 0 or self.wandb is None:
            return
        if runner.iter % self.interval != 0:
            return
        if not runner.log_buffer.output:
            runner.log_buffer.average(self.interval)
        log_vars = {
            k: v
            for k, v in runner.log_buffer.output.items()
            if isinstance(v, (int, float))
        }
        payload = dict(log_vars)
        if self.log_debug_images:
            debug_images = self._get_new_debug_images(runner)
            if debug_images:
                payload["train/class_mix_debug"] = debug_images
        if payload:
            self.wandb.log(payload, step=runner.iter, commit=self.commit)

    def after_evaluate(self, runner, eval_res=None, results=None, dataset=None):
        if runner.rank != 0 or self.wandb is None:
            return
        eval_scalars = {
            f"eval/{k}": v
            for k, v in (eval_res or {}).items()
            if isinstance(v, (int, float))
        }
        if eval_scalars:
            self.wandb.log(eval_scalars, step=runner.iter, commit=False)
        if self.log_eval_images and results is not None and dataset is not None:
            images = self._build_eval_images(results, dataset)
            if images:
                self.wandb.log(
                    {"eval/predictions": images},
                    step=runner.iter,
                    commit=self.commit,
                )

    def _get_new_debug_images(self, runner):
        debug_dir = osp.join(runner.work_dir, "class_mix_debug")
        if not osp.isdir(debug_dir):
            return []
        image_paths = sorted(
            osp.join(debug_dir, name)
            for name in os.listdir(debug_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        new_paths = [p for p in image_paths if p not in self._logged_debug_images]
        if not new_paths:
            return []
        selected = new_paths if self.num_debug_images is None else new_paths[-self.num_debug_images:]
        self._logged_debug_images.update(selected)
        return [
            self.wandb.Image(path, caption=osp.basename(path))
            for path in selected
        ]

    def _build_eval_images(self, results, dataset):
        images = []
        palette = np.array(dataset.PALETTE or [], dtype=np.uint8)
        count = min(len(results), len(dataset.img_infos))
        if self.num_eval_images is not None:
            count = min(count, self.num_eval_images)
        for idx in range(count):
            pred = self._load_prediction(results[idx])
            if pred is None:
                continue
            pred_rgb = self._colorize_prediction(pred, palette)
            caption = dataset.img_infos[idx].get("filename", str(idx))
            image_path = osp.join(dataset.img_dir, caption)
            try:
                from mmseg.utils import mmcv_compat as mmcv

                image = mmcv.imread(image_path)
                image = image[:, :, ::-1]
                pred_rgb = self._resize_to(pred_rgb, image.shape[:2])
                overlay = (0.55 * image + 0.45 * pred_rgb).astype(np.uint8)
                images.append(self.wandb.Image(overlay, caption=caption))
            except Exception:
                images.append(self.wandb.Image(pred_rgb, caption=caption))
        return images

    def _load_prediction(self, result):
        if isinstance(result, str):
            result = np.load(result)
        if isinstance(result, (list, tuple)):
            if not result:
                return None
            result = result[0]
        pred = np.asarray(result)
        if pred.ndim == 3:
            pred = np.squeeze(pred)
        if pred.ndim != 2:
            return None
        return pred.astype(np.int64)

    def _colorize_prediction(self, pred, palette):
        if palette.size == 0:
            unique = np.unique(pred)
            palette = np.random.default_rng(0).integers(
                0, 255, size=(int(unique.max()) + 1, 3), dtype=np.uint8)
        pred_safe = np.clip(pred, 0, len(palette) - 1)
        return palette[pred_safe]

    def _resize_to(self, image, shape):
        if image.shape[:2] == tuple(shape):
            return image
        from mmseg.utils import mmcv_compat as mmcv

        return mmcv.imresize(image, (shape[1], shape[0]), interpolation="nearest")

    def _sanitize_config(self, value):
        if isinstance(value, dict):
            return {str(k): self._sanitize_config(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize_config(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def after_run(self, runner):
        if runner.rank != 0 or self.wandb is None:
            return
        if self.log_checkpoint:
            latest = osp.join(runner.work_dir, "latest.pth")
            if osp.isfile(latest):
                self.wandb.save(latest)
        self.wandb.finish()


class CheckpointHook(Hook):
    def __init__(self, interval=1, by_epoch=True, max_keep_ckpts=5,
                 meta=None, **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.max_keep_ckpts = max_keep_ckpts
        self.meta = meta or {}

    def after_train_iter(self, runner):
        if not self.by_epoch and runner.iter % self.interval == 0 and runner.iter > 0:
            self._save(runner)

    def _save(self, runner):
        if runner.rank != 0:
            return
        filename = osp.join(runner.work_dir, "latest.pth")
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        meta = {"iter": runner.iter, **self.meta, **runner.meta}
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": runner.optimizer.state_dict(),
                "meta": meta,
            },
            filename,
        )


class EvalHook(Hook):
    greater_keys = ["mIoU", "mAcc", "aAcc"]

    def __init__(
        self,
        dataloader,
        interval=1,
        by_epoch=True,
        metric="mIoU",
        save_best=None,
        rule=None,
        gpu_collect=False,
        efficient_test=False,
        **kwargs,
    ):
        self.dataloader = dataloader
        self.interval = interval
        self.by_epoch = by_epoch
        self.metric = metric
        self.save_best = save_best
        self.rule = rule
        self.gpu_collect = gpu_collect
        self.efficient_test = efficient_test
        self.broadcast_bn_buffer = kwargs.get("broadcast_bn_buffer", True)
        self.tmpdir = kwargs.get("tmpdir", None)
        self.best_score = -1

    def _should_evaluate(self, runner):
        if self.by_epoch:
            return False
        return runner.iter > 0 and runner.iter % self.interval == 0

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, metric=self.metric, logger=runner.logger
        )
        for k, v in eval_res.items():
            runner.log_buffer.output[k] = v
        key_score = eval_res.get(self.metric, eval_res.get("mIoU", 0))
        if runner.logger:
            runner.logger.info(f"Evaluation {eval_res}")
        for hook in runner._hooks:
            hook.after_evaluate(
                runner,
                eval_res=eval_res,
                results=results,
                dataset=self.dataloader.dataset,
            )
        return key_score

    def _save_ckpt(self, runner, key_score):
        if key_score > self.best_score:
            self.best_score = key_score
            runner.logger.info(f"Save best checkpoint with score {key_score:.4f}")

    def _do_evaluate(self, runner):
        raise NotImplementedError

    def after_train_iter(self, runner):
        if self._should_evaluate(runner):
            self._do_evaluate(runner)


class DistEvalHook(EvalHook):
    def _do_evaluate(self, runner):
        from torch.nn.modules.batchnorm import _BatchNorm

        if self.broadcast_bn_buffer:
            model = runner.model
            for module in model.modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)
        if not self._should_evaluate(runner):
            return
        rank, _ = get_dist_info()
        if rank != 0:
            return
        from mmseg.apis import multi_gpu_test

        tmpdir = self.tmpdir or osp.join(runner.work_dir, ".eval_hook")
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            efficient_test=self.efficient_test,
        )
        runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
        self.evaluate(runner, results)


# Re-export mmseg EvalHook subclasses will inherit - we set base in eval_hooks.py
