import torch
import torch.nn.functional as F
import enc_dec_lib

from utils import euler_step_sample, pad_t_like_x
from utils import calculate_adaptive_weight, adopt_weight

def compute_loss(pred, target, loss_type, return_abs=False, lpips_fn=None):
    if loss_type == 'l1':
        consist_loss = torch.abs(pred - target)
    elif loss_type == 'pips':
        consist_loss = lpips_fn(pred, target)
    elif loss_type == 'l2':
        consist_loss = (pred - target) ** 2
    elif loss_type == '224':
        est = F.interpolate(pred, size=224, mode="bilinear")
        target = F.interpolate(target, size=224, mode="bilinear")   
        consist_loss = lpips_fn((est+1)/2.0, (target+1)/2.0)   # 4G
    if return_abs:
        return consist_loss
    else:
        return torch.mean(consist_loss)

def cd(x0, x1, student_model, target_model, teacher_model, interval, loss_type):
    # consistency distillation
    t = torch.rand(1).repeat(x0.shape[0]).type_as(x0)
    u = (t + interval).clip(0, 1)
    
    pad_t = pad_t_like_x(t, x0)
    pad_u = pad_t_like_x(u, x0)
    
    xt = pad_t * x1 + (1 - pad_t) * x0
    # xu = pad_u * x1 + (1 - pad_u) * x0

    x1_pred_student = euler_step_sample(k1=student_model(t, xt), xt=xt, t=t, t_dt=1)
    
    with torch.no_grad():
        v = teacher_model(t, xt)
        xu = (pad_u-pad_t)*v+xt
        x1_pred_target = euler_step_sample(k1=target_model(u, xu), xt=xu, t=u, t_dt=1)

    consist_loss = compute_loss(x1_pred_student, x1_pred_target, loss_type)
    return consist_loss

@torch.no_grad()
def get_kdc_target(x0, x1, xt, t, timestep_size, teacher_model, target_model, kdc, psnr_fn, x1_pred_student, cm_type, cond=None):

    kdc_xt_xu = 100
    u = t
    xu = xt
    dt = timestep_size

    while kdc_xt_xu > 100-kdc and u[0] < 1:
        if u[0] > 1 - dt:
            dt = 1 - u[0]
        if cm_type == 'ct':
            xu = dt * (x1 - x0) + xu
        elif cm_type == 'cd':
            xu = dt * teacher_model(u, xu, cond) + xu
        u = u + dt
    
        v_target = target_model(u, xu, cond)
        x1_pred_target = euler_step_sample(k1=v_target, xt=xu, t=u, t_dt=1)

        kdc_xt_xu = psnr_fn(x1_pred_student, x1_pred_target)

    return x1_pred_target

def get_gan_loss(args, model, real=None, fake=None, consistency_loss=None,
                 learn_generator=True, discriminator=None,
                 step=0, init_step=0, discriminator_feature_extractor=None, **model_kwargs):
    if model_kwargs is None:
        model_kwargs = {}
    if learn_generator:
        logits_fake = enc_dec_lib.get_xl_feature(args=args, estimate=fake, feature_extractor=discriminator_feature_extractor,
            discriminator=discriminator, step=step, **model_kwargs)
        g_loss = sum([(-l).mean() for l in logits_fake]) / len(logits_fake)     
        CTM_loss = consistency_loss.mean()
        if args.image_size == 32:
            layer_num = 11
        elif args.image_size == 64:
            layer_num = 15
        if args.parallel:
            d_weight = calculate_adaptive_weight(CTM_loss.mean(),
                                                        g_loss.mean(),
                                                        last_layer=
                                                        model.module.output_blocks[layer_num][0].out_layers[
                                                            3].weight)
        else:
            d_weight = calculate_adaptive_weight(CTM_loss.mean(),
                                                        g_loss.mean(),
                                                        last_layer=
                                                        model.output_blocks[layer_num][0].out_layers[
                                                            3].weight)
        d_weight = torch.clip(d_weight, 0.01, 10.)
        discriminator_loss = adopt_weight(d_weight, step, threshold=init_step + args.discriminator_start_itr) * g_loss
    else:
        logits_fake, logits_real = enc_dec_lib.get_xl_feature(args, fake.detach(), target=real.detach(),
                                                    feature_extractor=discriminator_feature_extractor,
                                                    discriminator=discriminator, step=step, **model_kwargs)
        loss_Dgen = sum([(F.relu(torch.ones_like(l) + l)).mean() for l in logits_fake]) / len(logits_fake)
        loss_Dreal = sum([(F.relu(torch.ones_like(l) - l)).mean() for l in logits_real]) / len(logits_real)
        discriminator_loss = loss_Dreal + loss_Dgen
    return discriminator_loss



def cm_gan_kdc(x0, x1, student_model, target_model, teacher_model, t, timestep_size, loss_type='l2', lpips_fn=None,
                gan_training=False, gan_args=None, discriminator=None, discriminator_feature_extractor=None,
                kdc=65, psnr_fn=None,
                cm_type='cd', step=0, cond=None):

    pad_t = pad_t_like_x(t, x0)
    xt = pad_t * x1 + (1 - pad_t) * x0
    v_pred = student_model(t, xt, cond)
    x1_pred_student = euler_step_sample(k1=v_pred, xt=xt, t=t, t_dt=1)
    
    cm_loss = None
    gan_loss = None
    
    if gan_training and step % 2 == 1:
        gan_loss = get_gan_loss(args=gan_args,model=student_model,
                    fake=x1_pred_student, real=x1, consistency_loss=None,
                    learn_generator=False, discriminator=discriminator, discriminator_feature_extractor=discriminator_feature_extractor,
                    step=step, init_step=0)
    else:
        if kdc > 0:
            x1_pred_target = get_kdc_target(x0, x1, xt, t, timestep_size,
                                            teacher_model, target_model, kdc, psnr_fn, x1_pred_student, cm_type, cond)
        else:
            u = (t + timestep_size).clamp(0, 1)
            pad_u = pad_t_like_x(u, x0)
            
            if cm_type == 'ct':
                xu = pad_u * x1 + (1 - pad_u) * x0
            elif cm_type == 'cd':
                v = teacher_model(t, xt, cond)
                xu = (pad_u-pad_t)*v+xt
            else:
                v = teacher_model(t, xt, cond)
                xu = ((pad_u-pad_t)*v+xt + pad_u*x1 + (1-pad_u)*x0)/2

            v_target = target_model(u, xu, cond)
            x1_pred_target = euler_step_sample(k1=v_target, xt=xu, t=u, t_dt=1)

        cm_loss = compute_loss(x1_pred_student, x1_pred_target, loss_type, lpips_fn=lpips_fn)

        if gan_training:
            gan_loss = get_gan_loss(args=gan_args,model=student_model,
                        fake=x1_pred_student, real=None, consistency_loss=cm_loss,
                        learn_generator=True, discriminator=discriminator, discriminator_feature_extractor=discriminator_feature_extractor,
                        step=step, init_step=0)
    
    losses = {
        'cm_loss': cm_loss,
        'gan_loss': gan_loss,
    }
    
    return losses
