# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os
import torch
import lpips
from absl import app, flags
from tqdm import trange

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.image import PeakSignalNoiseRatio
from torch.utils.tensorboard import SummaryWriter

import utils
import loss
from fm_fid import FmFID
import enc_dec_lib
from utils import ema, infiniteloop, print0
from torchcfm.optimal_transport import OTPlanSampler

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "results", help="output_directory")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4 0.000008
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 300001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 16, help="workers of Dataloader")

# Evaluation
flags.DEFINE_integer("save_step",20000,help="frequency of saving checkpoints, 0 to disable during training")

# CCM
flags.DEFINE_string("ema_decay", '0.9,0.999,0.9999', help="ema decay rates")
flags.DEFINE_string("resume", "null", help="resume from last saved checkpoint")
flags.DEFINE_string("dataset", "cifar10", help="cifar10 or imagenet64")
flags.DEFINE_integer("kdc", 0, help="knowledge discrepancy of curriculum")
flags.DEFINE_integer("gan_start", 0, help="the number of steps to start using GAN")
flags.DEFINE_float("alpha", 1.0, help="bias of t")
flags.DEFINE_string("loss_type", "224", help="loss type")
flags.DEFINE_string("cm_type", "cd", help="consistency distillation or consistency training")
flags.DEFINE_float("timestep_size", "0.03", help="timestep size")
flags.DEFINE_string("teacher", 'null', help='checkpoint of teacher model')

def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup

def flags_to_dict():
    hparams = {
        'model': FLAGS.model,
        'lr': FLAGS.lr,
        'total_steps': FLAGS.total_steps,
        'batch_size': FLAGS.batch_size,
        'num_workers': FLAGS.num_workers,
        'ema_decay': FLAGS.ema_decay,
        'resume': FLAGS.resume,
        'dataset': FLAGS.dataset,
        'kdc': FLAGS.kdc,
        'gan_start': FLAGS.gan_start,
        'alpha': FLAGS.alpha,
        'loss_type': FLAGS.loss_type,
        'cm_type': FLAGS.cm_type,
        'timestep_size': FLAGS.timestep_size,
        'teacher': FLAGS.teacher
    }
    return hparams

def train(argv):

    rank, world_size, local_rank = utils.init_distributed()
    print0("lr, total_steps, ema decay, save_step:",FLAGS.lr,FLAGS.total_steps,FLAGS.ema_decay,FLAGS.save_step)
    
    dataset = utils.get_dataset(FLAGS.dataset)
    batch_size_per_gpu = FLAGS.batch_size // world_size
    train_sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset,batch_size=batch_size_per_gpu,shuffle=False,num_workers=FLAGS.num_workers,drop_last=True,sampler=train_sampler)
    datalooper = infiniteloop(dataloader)

    net_model = utils.create_model(FLAGS.dataset)
    net_model.to(local_rank)
    ema_decays_str = FLAGS.ema_decay.split(',')
    ema_decays = [float(ema_decay) for ema_decay in ema_decays_str]
    ema_models = [copy.deepcopy(net_model) for i in range(len(ema_decays))]
    target_model = ema_models[0]
    ema_model = ema_models[-1]

    # load teacher
    teacher_model = copy.deepcopy(net_model)
    print0(f'load teacher from {FLAGS.teacher}')
    teacher_ckpt = torch.load(FLAGS.teacher, map_location='cpu')
    teacher_model.load_state_dict(teacher_ckpt['ema_model'])
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    # cifar10/224-otcfm-cd-gan0-kdc0-t1.0-tsz0.03-bsz128
    basedir = f'{FLAGS.output_dir}/{FLAGS.dataset}/{FLAGS.loss_type}-{FLAGS.model}-{FLAGS.cm_type}-gan{FLAGS.gan_start}-kdc{FLAGS.kdc}-t{FLAGS.alpha}-tsz{FLAGS.timestep_size}-bsz{FLAGS.batch_size}'
    resume = FLAGS.resume
    savedir = basedir
    restore_steps = 0

    # load student
    student_ckpt = None
    if os.path.exists(resume):
        print0(f'resume from {resume}')
        student_ckpt = torch.load(resume, map_location='cpu')
        net_model.load_state_dict(student_ckpt['net_model'])
        ema_model.load_state_dict(student_ckpt['ema_model'])
        target_model.load_state_dict(student_ckpt['target_model'])
        for ema_decay, copy_model in zip(ema_decays_str[1:-1], ema_models[1:-1]):
            if ema_decay in student_ckpt.keys():
                copy_model.load_state_dict(student_ckpt[ema_decay])

        restore_steps = student_ckpt['step']
        savedir = os.path.join(basedir, 'restore')
    elif teacher_ckpt:
        net_model.load_state_dict(teacher_ckpt['ema_model'])
        for copy_model in ema_models:
            copy_model.load_state_dict(teacher_ckpt['ema_model'])
    del teacher_ckpt
    torch.cuda.empty_cache()

    net_model = DistributedDataParallel(net_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    net_model.train()
    for copy_model in ema_models:
        copy_model.train()
        copy_model.requires_grad_(False)

    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    ##################################################################
    gan_args = None
    discriminator, discriminator_feature_extractor = None, None
    optim_gan = None
    gan_training = False
    if FLAGS.gan_start > 0:
        image_size = 32 if FLAGS.dataset=='cifar10' else 64
        gan_args = utils.get_gan_args(use_parallel=True, image_size=image_size, local_rank=local_rank)
        discriminator, discriminator_feature_extractor = enc_dec_lib.load_discriminator_and_d_feature_extractor(gan_args)
        optim_gan = torch.optim.RAdam(discriminator.parameters(), lr=0.002, weight_decay=0.0, betas=(0.5, 0.9))
    
    if student_ckpt is not None:
        optim.load_state_dict(student_ckpt['optim'])
        sched.load_state_dict(student_ckpt['sched'])
        if 'optim_gan' in student_ckpt.keys():
            discriminator.load_state_dict(student_ckpt['discriminator'])
            optim_gan.load_state_dict(student_ckpt['optim_gan'])
            gan_training = True
    del student_ckpt
    torch.cuda.empty_cache()
    ##################################################################

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print0("Model params: %.2f M" % (model_size / 1024 / 1024))

    if dist.get_rank() == 0:
        os.makedirs(savedir, exist_ok=True)
        os.makedirs(os.path.join(savedir, 'images'), exist_ok=True)
        writer = SummaryWriter(log_dir=savedir)
        hparams = flags_to_dict()
        for key, value in hparams.items():
            writer.add_text(f'Hparams/{key}', f'{value}')

    ot_sampler = OTPlanSampler(method="exact")
    lpips_fn = lpips.LPIPS(net='vgg', version='0.1').to(local_rank)
    psnr_fn = None
    if FLAGS.kdc > 0:
        psnr_fn = PeakSignalNoiseRatio().to(local_rank)
    
    fid_computer = FmFID().to(local_rank)
    best_fid = 500
    shards = 1 if FLAGS.dataset == 'cifar10' else world_size
    timestep_size = FLAGS.timestep_size
    
    with trange(restore_steps, FLAGS.total_steps, dynamic_ncols=True, disable=(dist.get_rank() != 0)) as pbar:
        for step in pbar:
            optim.zero_grad()
            epoch = (step*FLAGS.batch_size//shards) // len(dataloader.dataset)
            dataloader.sampler.set_epoch(epoch)
            x1, labels = next(datalooper)
            x1 = x1.to(local_rank)
            labels = labels.to(local_rank)
            x0 = torch.randn_like(x1)
            
            ############################################################
            cond = None
            if FLAGS.dataset == 'imagenet64':
                cond = labels
            if FLAGS.model == 'otcfm':
                x0, x1, _, cond = ot_sampler.sample_plan_with_labels(x0, x1, y0=None, y1=cond)
            ############################################################

            t = torch.rand(x0.shape[0]).type_as(x0) * (1-timestep_size)
            # t = torch.rand(1).type_as(x0).repeat(x0.shape[0]) * (1-timestep_size)
            t = t ** FLAGS.alpha

            losses = loss.cm_gan_kdc(x0, x1, net_model, target_model, teacher_model, t, timestep_size, FLAGS.loss_type, lpips_fn,
                gan_training, gan_args, discriminator, discriminator_feature_extractor, 
                kdc=FLAGS.kdc, psnr_fn=psnr_fn,
                cm_type=FLAGS.cm_type, step=step, cond=cond)

            cm_loss = losses['cm_loss']
            if gan_training:
                gan_loss = losses['gan_loss']
                if step % 2 == 1:
                    total_loss = gan_loss
                else:
                    total_loss = gan_loss + cm_loss
            else:
                total_loss = cm_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            if gan_training and step % 2 == 1:
                optim_gan.step()
                optim_gan.zero_grad()
            else:
                optim.step()
                sched.step()

            if dist.get_rank() == 0:
                writer.add_scalar('total_loss/total_loss', total_loss.item(), step)
                for param_group in optim.param_groups:
                    current_lr = param_group['lr']
                    writer.add_scalar('training/learning_rate', current_lr, global_step=step)

            for i, copy_model in enumerate(ema_models):
                ema(net_model.module, copy_model, ema_decays[i])

            pbar.set_postfix(loss=total_loss.item())
            pbar.update()

            # compute fid
            if step % (FLAGS.save_step // 8) == 0 or (FLAGS.gan_start > 0 and step == FLAGS.gan_start):
                save_dict = {
                        "net_model": net_model.module.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "target_model": target_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                }
                if gan_training:
                    save_dict['optim_gan'] = optim_gan.state_dict()
                    save_dict['discriminator'] = discriminator.state_dict()

                for ema_decay, copy_model in zip(ema_decays_str[1:-1], ema_models[1:-1]):
                    save_dict[ema_decay] = copy_model.state_dict()

                if step == FLAGS.gan_start and step > 0:
                    if dist.get_rank() == 0:
                        torch.save(save_dict, os.path.join(savedir, f'weights_ganstart_{step}.pt'))
                    gan_training = True

                ema_fids = []
                for i, copy_model in enumerate(ema_models):
                    copy_fid = utils.get_fid(50, copy_model, fid_computer, 
                                                rank=rank, world_size=world_size, local_rank=local_rank, 
                                                dataset=FLAGS.dataset, intergration_steps=1, step=step)

                    if dist.get_rank() == 0:
                        ema_fids.append(round(copy_fid, 2))
                        writer.add_scalar(f'metric/fid-{ema_decays[i]}', copy_fid, step)
                        
                if dist.get_rank() == 0:
                    with open(os.path.join(savedir, 'fid.txt'), 'a') as f:
                        msg = f'step:{step}, {ema_fids[0]}, {ema_fids[1]}, {ema_fids[2]}'
                        f.writelines(msg)
                        f.writelines('\n')
                        print(msg)

                if dist.get_rank() == 0:
                    # print()
                    # print(f'step:{step}, fid-{ema_decays}:{ema_fids}')
                    # sample and Saving the weights
                    if FLAGS.save_step > 0 and step % FLAGS.save_step == 0: # 20000
                        img_array = utils.generate_samples(ema_model, savedir, train_step=step, dataset=FLAGS.dataset, local_rank=local_rank, model_name=FLAGS.model)
                        writer.add_image('train', img_array, step, dataformats='HWC')
                        if step > 0:
                            torch.save(save_dict, os.path.join(savedir, f'weights_last.pt'))
                            for i, ema_fid in enumerate(ema_fids):
                                if ema_fid < best_fid:
                                    best_fid = ema_fid
                                    torch.save(save_dict, os.path.join(savedir, f'weights_best.pt'))
                                    break


if __name__ == "__main__":
    app.run(train)
