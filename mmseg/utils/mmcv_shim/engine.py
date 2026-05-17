import os
import os.path as osp
import pickle
import tempfile

import torch
import torch.distributed as dist


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    tmpdir = osp.join(tmpdir, 'results')
    os.makedirs(tmpdir, exist_ok=True)
    part_file = osp.join(tmpdir, f'part_{rank}.pkl')
    with open(part_file, 'wb') as f:
        pickle.dump(result_part, f)
    dist.barrier()
    if rank != 0:
        return None
    results = []
    for i in range(world_size):
        with open(osp.join(tmpdir, f'part_{i}.pkl'), 'rb') as f:
            results.extend(pickle.load(f))
    ordered = [None] * size
    for res in results:
        ordered[res[0]] = res[1]
    return ordered


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    shape_max = int(torch.stack(shape_list).max())
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:part_tensor.shape[0]] = part_tensor
    part_list = [torch.zeros_like(part_send) for _ in range(world_size)]
    dist.all_gather(part_list, part_send)
    if rank == 0:
        ordered = [None] * size
        for recv, shape in zip(part_list, shape_list):
            part = pickle.loads(recv[:int(shape[0])].cpu().numpy().tobytes())
            for item in part:
                ordered[item[0]] = item[1]
        return ordered
    return None


def get_dist_info():
    from mmseg.utils.mmcv_shim.runner.dist import get_dist_info as _g
    return _g()
