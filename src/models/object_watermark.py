#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : object_watermark.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/5/31 14:19

# Import lib here
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule, seed_everything
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.classification import MultilabelAccuracy
from omegaconf import DictConfig
from lpips import LPIPS
from typing import Tuple, List, Any, Dict

from src.utils import get_pylogger

log = get_pylogger(__name__)


def min_max_norm(x: torch.Tensor):
    return (x - x.min()) / (x.max() - x.min())


LPIPS_FN = LPIPS(net='vgg')


class ObjectWatermark(LightningModule):
    def __init__(self, image_shape: Tuple[int, int, int],
                 msg_len: int,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 augmenter: nn.Module,
                 syncor: nn.Module,
                 loss_cfg: DictConfig):
        super().__init__()
        torch.set_float32_matmul_precision('high')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False)

        self.encoder = encoder
        self.decoder = decoder
        self.augmenter = augmenter
        self.syncor = syncor
        self.loss_cfg = loss_cfg

        # Metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.train_bar = MultilabelAccuracy(num_labels=msg_len, threshold=0.5)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_bar = MultilabelAccuracy(num_labels=msg_len, threshold=0.5)
        self.test_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_bar = MultilabelAccuracy(num_labels=msg_len, threshold=0.5)

    def forward(self, x: torch.Tensor):
        pass

    def encode(self, image: torch.Tensor, mask: torch.Tensor, msg: torch.Tensor):
        if len(image.shape) == 3 and len(mask.shape) == 3 and len(msg.shape) == 1:
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            msg = msg.unsqueeze(0)
        assert len(image.shape) == 4 and len(mask.shape) == 4 and len(msg.shape) == 2
        container, residual = self.encoder(image, mask, msg, normalize=True)
        return container.squeeze(0), residual.squeeze(0)

    def decode(self, container: torch.Tensor, mask: torch.Tensor):
        if len(container.shape) == 3 and len(mask.shape) == 3:
            container = container.unsqueeze(0)
            mask = mask.unsqueeze(0)
        assert len(container.shape) == 4 and len(mask.shape) == 4
        container, mask = container.to(self.device), mask.to(self.device)
        # FIXME:
        # sync_container, sync_mask = self.syncor(container, mask, normalize=True)
        return self.decoder(container, mask, normalize=True).squeeze(0)

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # TODO: reset valid metrics
        self.val_psnr.reset()
        self.val_ssim.reset()
        self.val_bar.reset()

        # Using global variable to avoid saving model in pickle
        global LPIPS_FN
        LPIPS_FN = LPIPS_FN.to(self.device)

    def shared_step(self, batch: Any):
        host, mask, msg = batch
        # Encode
        container, residual = self.encoder(host, mask, msg, normalize=True)
        # Augment
        aug_container, aug_mask = self.augmenter(container, mask, self.trainer.fit_loop.total_batch_idx)
        # Decode
        sync_container, sync_mask = self.syncor(aug_container, aug_mask, normalize=True)
        msg_hat_logit = self.decoder(sync_container, sync_mask, normalize=False)

        sync_container = (sync_container + 1) / 2.
        return host, mask, msg, residual, container, aug_container, sync_container, aug_mask, msg_hat_logit

    def training_step(self, batch: Any, batch_idx: int):
        (host, mask, msg, residual, container, aug_container,
         sync_container, aug_mask, msg_hat_logit) = self.shared_step(batch)
        msg_hat = torch.sigmoid(msg_hat_logit)

        # Calculate loss
        # Encode loss
        vis_loss = F.l1_loss(host, container) + LPIPS_FN(host, container, normalize=True).mean()
        vis_loss = vis_loss * self.loss_cfg.vis_weight if self.current_epoch >= self.loss_cfg.vis_delay_epoch else 0.
        # Decode loss
        msg_loss = F.binary_cross_entropy_with_logits(msg_hat_logit, msg.float()) * self.loss_cfg.msg_weight
        loss = vis_loss + msg_loss
        self.log('train/vis_loss', vis_loss)
        self.log('train/msg_loss', msg_loss)
        self.log('train/loss', loss)

        # Calculate metrics
        with torch.no_grad():
            psnr = self.train_psnr(container, host)
            bar = self.train_bar(msg_hat, msg)
        self.log('train/psnr', psnr)
        self.log('train/bar', bar)

        # Visualize
        current_batch_idx = self.trainer.fit_loop.total_batch_idx
        if current_batch_idx % 300 == 0:
            show_image = dict(
                exp1=torch.cat([
                    host[:1]*mask[:1], min_max_norm(residual[:1]), container[:1],
                    aug_container[:1], aug_container[:1]*aug_mask[:1], sync_container[:1]], dim=0),
                exp2=torch.cat([
                    host[1:2]*mask[1:2], min_max_norm(residual[1:2]), container[1:2],
                    aug_container[1:2], aug_container[1:2]*aug_mask[1:2], sync_container[1:2]], dim=0),
                exp3=torch.cat([
                    host[2:3]*mask[2:3], min_max_norm(residual[2:3]), container[2:3],
                    aug_container[2:3], aug_container[2:3]*aug_mask[2:3], sync_container[2:3]], dim=0),
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
        (host, mask, msg, residual, container, aug_container,
         sync_container, aug_mask, msg_hat_logit) = self.shared_step(batch)
        msg_hat = torch.sigmoid(msg_hat_logit)

        # Update metrics
        with torch.no_grad():
            self.val_psnr.update(container, host)
            self.val_ssim.update(container, host)
            self.val_bar.update(msg_hat, msg)

        # Visualize
        if batch_idx == 0:
            show_image = dict(
                exp1=torch.cat([
                    host[:1] * mask[:1], min_max_norm(residual[:1]), container[:1],
                    aug_container[:1], aug_container[:1]*aug_mask[:1], sync_container[:1]], dim=0),
                exp2=torch.cat([
                    host[1:2] * mask[1:2], min_max_norm(residual[1:2]), container[1:2],
                    aug_container[1:2], aug_container[1:2]*aug_mask[1:2], sync_container[1:2]], dim=0),
                exp3=torch.cat([
                    host[2:3] * mask[2:3], min_max_norm(residual[2:3]), container[2:3],
                    aug_container[2:3], aug_container[2:3]*aug_mask[2:3], sync_container[2:3]], dim=0),
            )
            self.visualize2logger('val', show_image, self.current_epoch)
        pass

    def on_validation_epoch_end(self) -> None:
        psnr = self.val_psnr.compute()
        ssim = self.val_ssim.compute()
        bar = self.val_bar.compute()

        self.val_psnr.reset()
        self.val_ssim.reset()
        self.val_bar.reset()
        self.logger_instance.add_scalar('val/psnr', psnr, self.current_epoch)
        self.logger_instance.add_scalar('val/ssim', ssim, self.current_epoch)
        self.logger_instance.add_scalar('val/bar', bar, self.current_epoch)

        # Log to file
        log.info(f'Epoch {self.current_epoch}: psnr={psnr:.4f}, ssim={ssim:.4f}, bar={bar:.4f}')
        pass

    def on_test_start(self) -> None:
        seed_everything(42, workers=True)

    def test_step(self, batch: Any, batch_idx: int):
        (host, mask, msg, residual, container, aug_container,
         sync_container, aug_mask, msg_hat_logit) = self.shared_step(batch)
        msg_hat = torch.sigmoid(msg_hat_logit)

        # Update metrics
        with torch.no_grad():
            self.test_psnr.update(container, host)
            self.test_ssim.update(container, host)
            self.test_bar.update(msg_hat, msg)

        # Visualize
        if batch_idx % 200 == 0:
            show_image = dict(
                exp1=torch.cat([
                    host[:1] * mask[:1], min_max_norm(residual[:1]), container[:1],
                    aug_container[:1], aug_container[:1]*aug_mask[:1], sync_container[:1]], dim=0),
                exp2=torch.cat([
                    host[1:2] * mask[1:2], min_max_norm(residual[1:2]), container[1:2],
                    aug_container[1:2], aug_container[1:2]*aug_mask[1:2], sync_container[1:2]], dim=0),
                exp3=torch.cat([
                    host[2:3] * mask[2:3], min_max_norm(residual[2:3]), container[2:3],
                    aug_container[2:3], aug_container[2:3]*aug_mask[2:3], sync_container[2:3]], dim=0),
            )
            self.visualize2logger('test', show_image, batch_idx)

    def on_test_epoch_end(self) -> None:
        psnr = self.test_psnr.compute()
        ssim = self.test_ssim.compute()
        bar = self.test_bar.compute()

        self.test_psnr.reset()
        self.test_ssim.reset()
        self.test_bar.reset()

        self.logger_instance.add_scalar('test/psnr', psnr, self.current_epoch)
        self.logger_instance.add_scalar('test/ssim', ssim, self.current_epoch)
        self.logger_instance.add_scalar('test/bar', bar, self.current_epoch)

        # Log to file
        log.info(f'Test: psnr={psnr:.4f}, ssim={ssim:.4f}, bar={bar:.4f}')

    def configure_optimizers(self):
        optim = torch.optim.Adam([
            *self.encoder.parameters(),
            *self.decoder.parameters()
        ], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)
        return {'optimizer': optim}

    @property
    def logger_instance(self):
        return self.logger.experiment

    def visualize2logger(self, stage: str, image_dict: Dict, step: int):
        for label, image in image_dict.items():
            self.logger_instance.add_images(f'{stage}/{label}', image, step)


def run():
    pass


if __name__ == '__main__':
    run()
