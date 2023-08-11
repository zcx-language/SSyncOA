#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : self_sync.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/8/8 15:49

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tvn
import kornia as K
from lightning import LightningModule, seed_everything
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanAbsoluteError, JaccardIndex
from omegaconf import DictConfig
from lpips import LPIPS
from typing import Tuple, List, Any, Dict

from src.utils import get_pylogger
log = get_pylogger(__name__)


def min_max_norm(x: torch.Tensor):
    return (x - x.min()) / (x.max() - x.min())


LPIPS_FN = LPIPS(net='vgg')


class SelfSync(LightningModule):
    def __init__(self, model_cfg: DictConfig,
                 encoder: nn.Module,
                 augmenter: nn.Module,
                 syncor: nn.Module,
                 decoder: nn.Module,
                 loss_fn: nn.ModuleDict,
                 loss_cfg: DictConfig):
        super().__init__()
        torch.set_float32_matmul_precision('high')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False)

        self.model_cfg = model_cfg
        self.encoder = encoder
        self.augmenter = augmenter
        self.syncor = syncor
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.loss_cfg = loss_cfg

        # Metrics
        self.train_psnr = PeakSignalNoiseRatio()
        self.train_theta_err = MeanAbsoluteError()
        self.train_iou = JaccardIndex(task='binary')

        self.val_psnr = PeakSignalNoiseRatio()
        self.val_theta_err = MeanAbsoluteError()
        self.val_iou = JaccardIndex(task='binary')

    def forward(self, x: torch.Tensor):
        pass

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # Att: reset() will clear the cache of metrics
        self.val_psnr.reset()
        self.val_theta_err.reset()
        self.val_iou.reset()

        # Using global variable to avoid saving model in pickle
        global LPIPS_FN
        LPIPS_FN = LPIPS_FN.to(self.device)

    def shared_step(self, batch: Any):
        host, mask, msg, bg_img = batch
        host[1:] = host[0].repeat(host.shape[0]-1, 1, 1, 1)
        mask[1:] = mask[0].repeat(mask.shape[0]-1, 1, 1, 1)
        # Encode
        container, residual = self.encoder(host, mask, msg, normalize=True)
        # Augment
        aug_container, aug_mask = self.augmenter(container, mask, bg_img, self.trainer.fit_loop.total_batch_idx)
        theta_gt = self.augmenter.geometric_aug_dict.affine._params

        # Crop the mask

        # Sync
        sync_object_patch, sync_mask_patch, theta = self.syncor(aug_container, aug_mask, normalize=True)
        # Decode
        msg_hat_logit = self.decoder(sync_object_patch, sync_mask_patch, normalize=True)

        return (host, mask, msg, container, residual, aug_container, aug_mask, sync_object_patch,
                sync_mask_patch, theta, theta_gt, msg_hat_logit)

    def training_step(self, batch: Any, batch_idx: int):
        (host, mask, msg, container, residual, aug_container, aug_mask, sync_object_patch,
         sync_mask_patch, theta, theta_gt, msg_hat_logit) = self.shared_step(batch)

        # Calculate loss
        # Encode loss
        mask_weight = (1 - mask) * 1e4 + torch.ones_like(mask)
        vis_loss = (self.loss_fn.pix_loss(residual*mask_weight, torch.zeros_like(residual)) +
                    LPIPS_FN(container, host, normalize=True).mean())
        vis_loss = vis_loss * self.loss_cfg.vis_weight if self.current_epoch > self.loss_cfg.vis_delay_epoch else 0.
        # Decode loss
        # msg_loss = F.binary_cross_entropy_with_logits(msg_hat_logit, msg.float()) * self.loss_cfg.msg_weight
        msg_loss = 0.
        # Sync loss
        sync_loss = F.mse_loss(theta, theta_gt) * self.loss_cfg.sync_weight

        loss = vis_loss + msg_loss + sync_loss
        self.log('train/vis_loss', vis_loss)
        self.log('train/msg_loss', msg_loss)
        self.log('train/sync_loss', sync_loss)
        self.log('train/loss', loss)

        # Calculate metrics
        with torch.no_grad():
            psnr = self.train_psnr(container, host)
            theta_err = self.train_theta_err(theta, theta_gt)
        self.log('train/psnr', psnr)
        self.log('train/theta_err', theta_err)

        # Visualize
        current_batch_idx = self.trainer.fit_loop.total_batch_idx
        image_size = self.model_cfg.image_shape[-2:]
        if current_batch_idx % 300 == 0:
            show_image = dict(
                exp1=torch.cat([
                    host[:1] * mask[:1], min_max_norm(residual[:1]), container[:1],
                    F.interpolate(aug_container[:1], size=tuple(image_size)),
                    F.interpolate(aug_container[:1] * aug_mask[:1], size=tuple(image_size)),
                    sync_object_patch[:1] * sync_mask_patch[:1]], dim=0),
                exp2=torch.cat([
                    host[1:2] * mask[1:2], min_max_norm(residual[1:2]), container[1:2],
                    F.interpolate(aug_container[1:2], size=tuple(image_size)),
                    F.interpolate(aug_container[1:2] * aug_mask[1:2], size=tuple(image_size)),
                    sync_object_patch[1:2] * sync_mask_patch[1:2]], dim=0),
                exp3=torch.cat([
                    host[2:3] * mask[2:3], min_max_norm(residual[2:3]), container[2:3],
                    F.interpolate(aug_container[2:3], size=tuple(image_size)),
                    F.interpolate(aug_container[2:3] * aug_mask[2:3], size=tuple(image_size)),
                    sync_object_patch[2:3] * sync_mask_patch[2:3]], dim=0),
            )
            self.visualize2logger('train', show_image, current_batch_idx)
        return loss

    def on_train_epoch_end(self) -> None:
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        (host, mask, msg, container, residual, aug_container, aug_mask, sync_object_patch,
         sync_mask_patch, theta, theta_gt, msg_hat_logit) = self.shared_step(batch)

        with torch.no_grad():
            self.val_psnr.update(container, host)
            self.val_theta_err.update(theta, theta_gt)
            self.val_iou.update(sync_mask_patch, aug_mask)

        # Visualize
        image_size = self.model_cfg.image_shape[-2:]
        if batch_idx == 0:
            show_image = dict(
                exp1=torch.cat([
                    host[:1] * mask[:1], min_max_norm(residual[:1]), container[:1],
                    F.interpolate(aug_container[:1], size=tuple(image_size)),
                    F.interpolate(aug_container[:1] * aug_mask[:1], size=tuple(image_size)),
                    sync_object_patch[:1] * sync_mask_patch[:1]], dim=0),
                exp2=torch.cat([
                    host[1:2] * mask[1:2], min_max_norm(residual[1:2]), container[1:2],
                    F.interpolate(aug_container[1:2], size=tuple(image_size)),
                    F.interpolate(aug_container[1:2] * aug_mask[1:2], size=tuple(image_size)),
                    sync_object_patch[1:2] * sync_mask_patch[1:2]], dim=0),
                exp3=torch.cat([
                    host[2:3] * mask[2:3], min_max_norm(residual[2:3]), container[2:3],
                    F.interpolate(aug_container[2:3], size=tuple(image_size)),
                    F.interpolate(aug_container[2:3] * aug_mask[2:3], size=tuple(image_size)),
                    sync_object_patch[2:3] * sync_mask_patch[2:3]], dim=0),
            )
            self.visualize2logger('val', show_image, self.current_epoch)
        pass

    def on_validation_epoch_end(self) -> None:
        psnr = self.val_psnr.compute()
        theta_err = self.val_theta_err.compute()
        iou = self.val_iou.compute()

        self.val_psnr.reset()
        self.val_theta_err.reset()
        self.val_iou.reset()

        self.logger_instance.add_scalar('val/psnr', psnr, self.current_epoch)
        self.logger_instance.add_scalar('val/theta_err', theta_err, self.current_epoch)
        self.logger_instance.add_scalar('val/iou', iou, self.current_epoch)

        # Log to file
        log.info(f'Epoch {self.current_epoch}: psnr={psnr:.4f}, theta_err={theta_err:.4f}, iou={iou:.4f}')
        pass

    def on_test_start(self) -> None:
        seed_everything(42, workers=True)

    def test_step(self, batch: Any, batch_idx: int):
        (host, mask, msg, container, residual, aug_container, aug_mask, sync_object_patch,
         sync_mask_patch, theta, theta_gt, msg_hat_logit) = self.shared_step(batch)
        pass

    def on_test_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        optim = torch.optim.Adam([
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *self.segmenter.parameters(),
        ], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)
        return {'optimizer': optim}

    @property
    def logger_instance(self):
        return self.logger.experiment

    def visualize2logger(self, stage: str, image_dict: Dict, step: int):
        for label, image in image_dict.items():
            if image.ndim == 4:
                self.logger_instance.add_images(f'{stage}/{label}', image, step)
            elif image.ndim == 3:
                self.logger_instance.add_image(f'{stage}/{label}', image, step)
            else:
                raise ValueError


def run():
    pass


if __name__ == '__main__':
    run()
