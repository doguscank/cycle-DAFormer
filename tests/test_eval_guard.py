from types import SimpleNamespace

import pytest

from mmseg.utils.mmcv_shim.runner.hooks import EvalHook


class _Dataset:
    def __init__(self, eval_res):
        self.eval_res = eval_res

    def evaluate(self, results, metric='mIoU', logger=None):
        return self.eval_res


class _Logger:
    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


def _runner():
    return SimpleNamespace(logger=_Logger(), log_buffer=SimpleNamespace(
        output={}), _hooks=[])


def test_first_eval_guard_terminates_below_threshold():
    hook = EvalHook(
        SimpleNamespace(dataset=_Dataset({'mIoU': 0.49})),
        by_epoch=False,
        first_eval_min_miou=0.5)

    with pytest.raises(SystemExit):
        hook.evaluate(_runner(), results=[])


def test_first_eval_guard_can_warn_without_terminating():
    hook = EvalHook(
        SimpleNamespace(dataset=_Dataset({'mIoU': 0.49})),
        by_epoch=False,
        first_eval_min_miou=0.5,
        first_eval_kill=False)

    assert hook.evaluate(_runner(), results=[]) == 0.49

