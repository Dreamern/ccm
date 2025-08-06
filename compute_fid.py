# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import os
import sys

import torch
from absl import app, flags
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

from torchcfm.models.unet.unet import UNetModelWrapper
import utils

FLAGS = flags.FLAGS
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_string("input_dir", "./results", help="output_directory")
flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_integer("integration_steps", 100, help="number of inference steps")
flags.DEFINE_string("integration_method", "dopri5", help="integration method to use")
flags.DEFINE_integer("step", 400000, help="training steps")
flags.DEFINE_integer("num_gen", 50000, help="number of samples to generate")
flags.DEFINE_float("tol", 1e-5, help="Integrator tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size_fid", 1024, help="Batch size to compute FID")
flags.DEFINE_string("path", ' ', help="ckpt path")

FLAGS(sys.argv)


# Define the model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

new_net = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=FLAGS.num_channel,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
).to(device)


# Load the model
PATH = FLAGS.path
print("path: ", PATH)
checkpoint = torch.load(PATH)
state_dict = checkpoint["ema_model"]
try:
    new_net.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    new_net.load_state_dict(new_state_dict)

model_size = 0
for param in new_net.parameters():
    model_size += param.data.nelement()
print("Model params: %.2f M" % (model_size / 1024 / 1024))

new_net.eval()

utils.generate_samples(new_net, savedir='samples', net_='ema')

# Define the integration method if euler is used
if FLAGS.integration_method == "euler":
    node = NeuralODE(new_net, solver=FLAGS.integration_method)


def gen_1_img(unused_latent):
    with torch.no_grad():
        x = torch.randn(FLAGS.batch_size_fid, 3, 32, 32, device=device)
        if FLAGS.integration_method == "euler":
            t_span = torch.linspace(0, 1, FLAGS.integration_steps + 1, device=device)
            traj = node.trajectory(x, t_span=t_span)
        else:
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(0, 1, 2, device=device)
            traj = odeint(
                new_net, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method
            )
    traj = traj[-1, :] # .view([-1, 3, 32, 32]).clip(-1, 1)
    img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)  # .permute(1, 2, 0)
    return img


print("Start computing FID")
print("Use method: ", FLAGS.integration_method)
score = fid.compute_fid(
    gen=gen_1_img,
    dataset_name="cifar10",
    batch_size=FLAGS.batch_size_fid,
    dataset_res=32,
    num_gen=FLAGS.num_gen,
    dataset_split="train",
    mode="legacy_tensorflow",
)
print()
print("FID has been computed")
# print()
# print("Total NFE: ", new_net.nfe)
print()
print("FID: ", score)
