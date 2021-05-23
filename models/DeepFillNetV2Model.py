import logging
import os

import numpy as np
import torch
from torch import autograd

from models import get_scheduler
from models.NN.DeepFill2.DeepFillNet import InpaintSNNet
from models.NN.DeepFill2.SNPatchGan import SnPatchGanDirciminator
from utils.PerceptualLoss import PerceptualLoss
from utils.loss import SNDisLoss, SNGenLoss
from utils.modelUtils import weights_init
from utils.tensor_ref import tensor2img


class DeepFillNetV2Model():
    def __init__(self, cfg):
        m_cfg = cfg.copy()
        m_cfg.update(cfg['model'])
        self.cfg = m_cfg
        gpu_ids = [int(_) for _ in m_cfg.get('gpu_ids', [])]
        if len(gpu_ids) == 0:
            self.gpu_ids = None
        else:
            self.gpu_ids = [int(_) for _ in gpu_ids]
        self.isTrain = (cfg['stage'] == "train")
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids is not None else torch.device(
            'cpu')  # get device name: CPU or GPU

        G_cfg = cfg.copy()
        G_cfg.update(cfg['nn'])
        G_cfg.update(cfg['nn']['G'])
        self.netG = InpaintSNNet(
            in_channels = G_cfg.get('in_chanels', 5),
            out_channels = G_cfg.get('out_chanels', 3),
            basic_num = G_cfg.get('basic_num', 48)
        )
        self.netG.to(self.device)
        weights_init(self.netG, cfg['init_type'])

        if self.isTrain:
            D_cfg = cfg.copy()
            D_cfg.update(cfg['nn'])
            D_cfg.update(cfg['nn']['D'])
            self.gan_with_mask = m_cfg.get('gan_with_mask', False)
            self.netD = SnPatchGanDirciminator(
                in_channels = D_cfg.get('in_chanels', 7),
                basic_num = D_cfg.get('basic_num', 48)
            )
            self.netD.to(self.device)
            self.lr = m_cfg['learning_rate']
            self.l1 = torch.nn.L1Loss()
            self.criterionDis = SNDisLoss().to(self.device)
            self.criterionGen = SNGenLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.perceptualLoss = PerceptualLoss(torch.nn.MSELoss(), ['conv2_1', 'conv3_1', 'conv4_1']).to(self.device)

            beta1 = m_cfg['beta1']
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]
            self.schedulers = [get_scheduler(optimizer, m_cfg) for optimizer in self.optimizers]
        else:
            self.netG.eval()

    def set_input(self, image, mask):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.images = image.to(self.device).float()
        self.masks = mask.to(self.device).float()
        self.incomplete = self.images * (1 - self.masks)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.coarse, self.recon_imgs, self.offset_flow = self.netG(self.incomplete, self.masks)  # G(A)
        self.complete_imgs = self.recon_imgs * self.masks + self.incomplete * (1 - self.masks)
        return self.complete_imgs

    def optimize_parameters(self):
        self.current_loss = {}
        self.forward()                   # compute fake images: G(A)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.optimizer_D.zero_grad()     # set D's gradients to zero

        pos = self.images
        neg = self.complete_imgs
        if self.gan_with_mask:
            pos = torch.cat([pos, self.masks], dim=1)
            neg = torch.cat([neg, self.masks], dim=1)
        pos_neg_imgs = torch.cat([pos, neg.detach()], dim=0)

        pred_pos_neg = self.netD(pos_neg_imgs)
        pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
        d_loss = self.criterionDis(pred_pos, pred_neg)
        d_loss.backward(retain_graph=True)
        d_penalty = self.calc_gradient_penalty(
            self.netD, pos, neg.detach()) * self.cfg.get('wgan_gp_lambda', 1.0)
        d_loss += d_penalty
        self.optimizer_D.step()

        # optim G
        self.optimizer_D.zero_grad()
        self.optimizer_G.zero_grad()

        pred_neg = self.netD(neg)
        # pred_pos, pred_neg = torch.chunk(pred_pos_neg,  2, dim=0)
        g_loss = self.criterionGen(pred_neg) * self.cfg.get('lambda_construct', 1.0)
        r_loss = self.l1(self.images, self.recon_imgs)
        r_loss += self.l1(self.images, self.coarse) # tf.reduce_mean(tf.abs(batch_pos - x1)) ?
        r_loss *= self.cfg.get('lambda_construct', 1.0)

        vgg_loss = self.perceptualLoss(neg[:,:3,...], pos[:,:3,...])
        total_loss = g_loss + r_loss + vgg_loss
        total_loss.backward()
        self.optimizer_G.step()

        self.current_loss = {
            'loss-d': d_loss.item(),
            'loss-g': g_loss.item(),
            'loss-r': r_loss.item(),
            'vgg': vgg_loss.item()
        }

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data).to(self.device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_().clone()

        disc_interpolates = netD(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size()).to(self.device)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=grad_outputs, create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty


    def save(self, save_dir, name, iter):
        for model, net_name in zip([self.netG, self.netD], ['G', 'D']):
            if hasattr(model, 'module'):
                model = model.module
            ckpt_path = os.path.join(save_dir, f"{iter}_{name}_{net_name}.pth")
            logging.info(f"Save ckpt:{ckpt_path}")
            torch.save(model.state_dict(), ckpt_path)

    def load(self, save_dir, name, iter):
        models = [self.netG]
        names = ['G']
        if self.isTrain:
            models.append(self.netD)
            names.append("D")
        for model, net_name in zip(models, names):
            logging.info(f"Load Net:{net_name} epoch:{iter}")
            ckpt_path = os.path.join(save_dir ,f"{iter}_{name}_{net_name}.pth")
            x = torch.load(ckpt_path)
            if hasattr(x, 'module'):
                x = x.module
                x = x.state_dict()
            model.load_state_dict(x)

    def get_current_visual(self, n):
        real_A = tensor2img(self.incomplete, n)
        real_B = tensor2img(self.complete_imgs, n)
        fake_B = tensor2img(self.images, n)
        return np.hstack([real_A, real_B, fake_B])

    def get_current_error(self):
        return self.current_loss

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.cfg['model']['lr_policy'] == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        return old_lr, lr

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
