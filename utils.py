import copy
import os

import torch
from torchcfm.models.unet.unet import UNetModelWrapper
from fm_fid import FmFID
from fm_euler import FmEuler
from tqdm import trange
import numpy as np
from torchdyn.numerics import odeint

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image

def init_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")
    return rank, world_size, local_rank

@torch.no_grad()
def unnorm_image(x, mean=0.5, std=0.5) -> torch.Tensor:
    img = (x * std + mean) * 255
    return img.clamp(0, 255).byte()

@torch.no_grad()
def euler_ode_sample(model, xt, t, t_dt=1, integration_step=1, return_trajs=False, cond=False, label=None, t_span=None):
    
    euler_solver = FmEuler()
    if t_span is None:
        t_span = torch.linspace(t, t_dt, integration_step+1, device=xt.device)
    
    _, trajs = odeint(model, xt, t_span, solver=euler_solver, 
                      args={'cond': cond,'label': label})

    traj = trajs[-1,:].view(xt.shape).clamp(-1, 1)
    if return_trajs:
        return traj, trajs
    else:
        return traj

@torch.no_grad()
def generate_samples(model, save_dir, train_step='last', dataset='cifar10', local_rank=0, integration_step=1, model_name=''):
    model.eval()

    if dataset == 'cifar10':
        x0 = torch.randn(16, 3, 32, 32).to(local_rank)
        label = None
        cond = False
    elif dataset == 'imagenet64':
        x0 = torch.randn(16, 3, 64, 64).to(local_rank)
        label = torch.randint(0, 1000, (16,)).to(local_rank)
        cond = True
    
    traj = euler_ode_sample(model, x0, t=0, t_dt=1, integration_step=integration_step, cond=cond, label=label)
    img = traj / 2 + 0.5
    
    save_path = os.path.join(save_dir, 'images', model_name[:-3] + '_' + str(integration_step) + '_' + str(train_step) + '.png')
    print(save_path)
    img_array = save_image(img, save_path, nrow=4)

    model.train()
    return img_array

def create_model(dataset):
    if dataset == 'cifar10':
        net_model = UNetModelWrapper(
            dim=(3, 32, 32),
            num_res_blocks=2,
            num_channels=128,
            class_cond=False,
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
        )
    elif dataset == 'imagenet64':
        net_model = UNetModelWrapper(
            dim=(3, 64, 64),
            num_res_blocks=3,
            num_channels=192,
            class_cond=True,
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="32,16,8",
            dropout=0.0,
        )
    
    return net_model

@torch.no_grad()
def get_eval_model(PATH, state_key, dataset, local_rank):
    
    new_net = create_model(dataset)
    new_net = new_net.to(local_rank)

    print("path: ", PATH)
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    state_dict = checkpoint[state_key]
    new_net.load_state_dict(state_dict)

    del checkpoint
    torch.cuda.empty_cache()

    new_net.eval()
    return new_net

@torch.no_grad()
def get_fid(batch_size, model, fid_computer:FmFID, 
            rank, world_size, local_rank, num_samples=50000, dataset='cifar10', intergration_steps=1, train=True,
            t_span=None):
    model.eval()

    cond = False
    if dataset == 'cifar10':
        resolution = 32
        fpath = './assets/cifar10_legacy_tensorflow_train_32.npz'
    elif dataset == 'imagenet64':
        resolution = 64
        fpath = './assets/imagenet-64x64.npz'
        cond = True
    
    stats = np.load(fpath)
    mu, sigma = stats["mu"], stats["sigma"]
    label = None

    for i in trange(int(num_samples / batch_size), disable=(rank != 0 or train)):
        if i % world_size == rank:
            noise = torch.randn(batch_size, 3, resolution, resolution, device=local_rank)
            if cond:
                label = torch.randint(0, 1000, (batch_size, 1))
                label = label.squeeze(1).to(local_rank)
            samples = euler_ode_sample(model, noise, t=0, t_dt=1, integration_step=intergration_steps, cond=cond, label=label, t_span=t_span)
            image = unnorm_image(samples)
            fid_computer.update(image, real=False)
    
    current_fid = fid_computer.compute_with_ref(mu,sigma)

    fid_computer.reset()
    torch.cuda.empty_cache()
    
    model.train()
    return current_fid

