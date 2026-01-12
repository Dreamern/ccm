# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel

import os
import sys

import utils
from fm_fid import FmFID

# python test_model.py model.pt cifar10
PATH = sys.argv[1]
task = sys.argv[2]  # fid, gen
dataset = sys.argv[3]

rank, world_size, local_rank = utils.init_distributed()

cond = False
if dataset == 'imagenet64':
    cond = True
state_key = 'ema_model'

model_name = PATH.split('/')[-1]
model_dir = os.path.dirname(PATH)
new_net = utils.get_eval_model(PATH, state_key, dataset, local_rank)
new_net = DistributedDataParallel(new_net, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

with torch.no_grad():
    if task == 'fid':
        fid_computer = FmFID().to(local_rank)
        fid = utils.get_fid(batch_size=100, 
                            num_samples=50000, 
                            intergration_steps=1, 
                            model=new_net, 
                            fid_computer=fid_computer, 
                            rank=rank, 
                            world_size=world_size, 
                            local_rank=local_rank,
                            dataset=dataset, 
                            train=False, 
                            t_span=None
                            )
        if rank == 0:
            print(f'{fid}')
    elif task == 'gen':
        if local_rank == 0:
            os.makedirs(os.path.join(model_dir, 'images'), exist_ok=True)
            utils.generate_samples(new_net, model_dir, dataset=dataset, local_rank=local_rank, integration_step=1, model_name=model_name)
    else:
        print('unsupported task')

