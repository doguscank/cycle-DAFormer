import os

import torch
import torch.distributed as dist


def init_dist(launcher, backend='nccl', **kwargs):
    if launcher == 'pytorch':
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        os.environ['MASTER_ADDR'] = node_list.split(',')[0]
        os.environ['MASTER_PORT'] = str(kwargs.get('port', 29500))
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        dist.init_process_group(backend=backend)
    else:
        raise ValueError(f'Unsupported launcher: {launcher}')


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size
