# Training and evaluation hooks.

from __future__ import annotations

import os
import os.path as osp
import warnings

import torch
import torch.distributed as dist

from mmseg.utils.mmcv_shim.runner.dist import get_dist_info


class Hook:
    def before_train_iter(self, runner):
        pass

    def after_train_iter(self, runner):
        pass

    def after_train_epoch(self, runner):
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
        runner.log_buffer.clear()


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
