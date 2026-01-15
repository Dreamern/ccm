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
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from types import SimpleNamespace
import enc_dec_lib
from PIL import Image
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import pathlib

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
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    return ndarr # HWC

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
    img = (traj / 2 + 0.5).clamp(0, 1) * 255
    save_path = os.path.join(save_dir, 'images', model_name + '_' + str(integration_step) + '_' + str(train_step) + '.png')
    print(save_path)
    img_array = save_image(img.byte(), save_path, nrow=4)

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
            t_span=None, step=0):
    model.eval()

    if step == 0 and train:
        num_samples = 1000

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

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )
        
def infiniteloop(dataloader, return_x=False):
    while True:
        for x, y in iter(dataloader):
            if return_x:
                yield x
            else:
                yield x, y
                
def print0(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)
        
def get_dataset(dataset_name):
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    return dataset

def get_gan_args(use_parallel, image_size, local_rank):
    args_dict = dict(
        d_lr=0.002,
        gan_training=True,
        gan_real_free=True,
        discriminator_weight=1.0,
        discriminator_start_itr=0,
        use_d_fp16=False,
        d_architecture='StyleGAN-XL',
        g_learning_period=1,
        gan_fake_outer_type='no',
        gan_fake_inner_type='',
        gan_real_inner_type='',
        gan_target_matching=False,
        data_augment=True,
        d_backbone=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'],
        d_apply_adaptive_weight=True,
        shift_ratio=0.125,
        cutout_ratio=0.2,
        gan_training_frequency=1.,
        gaussian_filter=False,
        blur_fade_itr=1000,
        blur_init_sigma=2,
        prob_aug=1.0,
        gan_different_augment=True,
        gan_num_heun_step=17,
        gan_heun_step_strategy='uniform',
        gan_specific_time=False,
        gan_low_res_train=False,
        save_png=False,
        save_period=1000,
        image_size=image_size,
        loss_norm="lpips",
        parallel=use_parallel,
        local_rank=local_rank
    )
    args = SimpleNamespace(**args_dict)
    
    return args

def get_gan(args):
    discriminator, discriminator_feature_extractor = enc_dec_lib.load_discriminator_and_d_feature_extractor(args)
    return discriminator, discriminator_feature_extractor

def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

def calculate_adaptive_weight(loss1, loss2, last_layer=None):
    loss1_grad = torch.autograd.grad(loss1, last_layer, retain_graph=True)[0]
    loss2_grad = torch.autograd.grad(loss2, last_layer, retain_graph=True)[0]
    d_weight = torch.norm(loss1_grad) / (torch.norm(loss2_grad) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def euler_step_sample(k1, xt, t, t_dt=1):
    dt = t_dt - t
    dt = pad_t_like_x(dt, xt)
    x_sol = xt + dt * k1
    return x_sol.clamp(-1, 1)