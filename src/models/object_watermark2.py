#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : object_watermark2.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/7/27 16:39

# Import lib here
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tvn
import kornia as K
from lightning import LightningModule, seed_everything
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.classification import MultilabelAccuracy, BinaryJaccardIndex
from omegaconf import DictConfig
from lpips import LPIPS
from typing import Tuple, List, Any, Dict

from src.utils import get_pylogger

log = get_pylogger(__name__)


def min_max_norm(x: torch.Tensor):
    return (x - x.min()) / (x.max() - x.min())


LPIPS_FN = LPIPS(net='vgg')


class ObjectWatermark2(LightningModule):
    def __init__(self, model_cfg: DictConfig,
                 encoder: nn.Module,
                 augmenter: nn.Module,
                 segmenter: nn.Module,
                 syncor: nn.Module,
                 syncor2: nn.Module,
                 decoder: nn.Module,
                 loss_fn: nn.ModuleDict,
                 loss_cfg: DictConfig):
        super().__init__()
        torch.set_float32_matmul_precision('high')
        self.automatic_optimization = False

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False)

        self.model_cfg = model_cfg
        self.encoder = encoder
        self.augmenter = augmenter
        self.segmenter = segmenter
        self.syncor = syncor
        self.syncor2 = syncor2
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.loss_cfg = loss_cfg

        # Metrics
        msg_len = model_cfg.msg_len
        img_size = model_cfg.image_shape[-2] * model_cfg.image_shape[-1]
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.train_bar = MultilabelAccuracy(num_labels=msg_len, threshold=0.5)
        self.train_iou = BinaryJaccardIndex(threshold=0.5)
        self.train_iou2 = BinaryJaccardIndex(threshold=0.5)
        # self.train_par = MultilabelAccuracy(num_labels=512*512, threshold=0.5)
        # self.train_par = MultilabelAccuracy(num_labels=256*256, threshold=0.5)

        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_bar = MultilabelAccuracy(num_labels=msg_len, threshold=0.5)
        self.val_iou = BinaryJaccardIndex(threshold=0.5)
        self.val_iou2 = BinaryJaccardIndex(threshold=0.5)
        # self.val_par = MultilabelAccuracy(num_labels=512*512, threshold=0.5)
        # self.val_par = MultilabelAccuracy(num_labels=256*256, threshold=0.5)

        self.test_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_bars = nn.ModuleDict({
            key: MultilabelAccuracy(num_labels=msg_len, threshold=0.5) for key in self.augmenter.augment_types})
        self.test_ious = nn.ModuleDict({
            key: BinaryJaccardIndex(threshold=0.5) for key in self.augmenter.augment_types})
        # self.test_par = MultilabelAccuracy(num_labels=256*256, threshold=0.5)

        # Log model arch to file
        from torchinfo import summary
        log.info(f'Encoder Summary:\n'
                 f'{summary(self.encoder, input_size=((1, 3, 256, 256), (1, 1, 256, 256), (1, 30)), depth=5, verbose=0)}')
        log.info(f'Decoder Summary:\n'
                 f'{summary(self.decoder, input_size=((1, 3, 256, 256), (1, 1, 256, 256)), depth=5, verbose=0)}')

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
        # Att: reset() will clear the cache of metrics
        self.val_psnr.reset()
        self.val_ssim.reset()
        self.val_bar.reset()
        self.val_iou.reset()
        self.val_iou2.reset()

        # Using global variable to avoid saving model in pickle
        global LPIPS_FN
        LPIPS_FN = LPIPS_FN.to(self.device)

    def shared_step(self, batch: Any, phase: str = 'train'):
        host, mask, msg, bg_img = batch
        # host: (1, 3, H, W)
        # mask: (1, 1, H, W)
        # msg: (1, real_batch_size, msg_len)
        # bg_img: (1, real_batch_size, 3, 512, 512)
        true_batch_size = msg.shape[1]
        host = host.repeat(true_batch_size, 1, 1, 1)
        mask = mask.repeat(true_batch_size, 1, 1, 1)
        msg = msg.squeeze(dim=0)
        bg_img = bg_img.squeeze(dim=0)

        # Encode
        container, residual = self.encoder(host, mask, msg, normalize=True)
        # Augment

        def seg_sync_decode(_aug_container, _aug_mask):
            # Segment
            # Att: aug_container is detached from the compute graph
            seg_mask_logits = self.segmenter(_aug_container.detach(), normalize=True)

            # Sync
            object_mask = _aug_mask.ge(0.5).int()
            if self.current_epoch >= self.model_cfg.adopt_seg_delay_epoch:
                for idx, seg_mask_logit in enumerate(seg_mask_logits):
                    seg_mask = seg_mask_logit.sigmoid().ge(0.5).int()
                    if seg_mask.sum() >= 1e4:   # Filter out abnormal seg mask
                        object_mask[idx] = seg_mask

            sync_object_patch, sync_mask_patch = self.syncor(_aug_container, object_mask)
            sync_object_patch2, sync_mask_patch2 = self.syncor2(_aug_container, seg_mask_logits.sigmoid().detach())
            # Decode
            msg_hat_logit = self.decoder(sync_object_patch, sync_mask_patch, normalize=True)
            return seg_mask_logits, sync_object_patch, sync_mask_patch, msg_hat_logit, sync_mask_patch2

        if phase == 'train' or phase == 'val':
            aug_dict = self.augmenter(container, mask, bg_img, self.trainer.fit_loop.total_batch_idx)
            aug_container, aug_mask = list(aug_dict.values())[0]
            return (host, mask, msg, container, residual,
                    aug_container, aug_mask, *seg_sync_decode(aug_container, aug_mask))
        elif phase == 'test':
            aug_dict = self.augmenter(container, mask, bg_img, self.trainer.fit_loop.total_batch_idx, return_all=True)
            seg_mask_logits_dict, sync_object_patch_dict, sync_mask_patch_dict, msg_hat_logit_dict = {}, {}, {}, {}
            for aug_name, (aug_container, aug_mask) in aug_dict.items():
                (seg_mask_logits_dict[aug_name], sync_object_patch_dict[aug_name],
                 sync_mask_patch_dict[aug_name], msg_hat_logit_dict[aug_name], _) = seg_sync_decode(aug_container, aug_mask)
            return (host, mask, msg, container, residual, aug_dict, seg_mask_logits_dict,
                    sync_object_patch_dict, sync_mask_patch_dict, msg_hat_logit_dict)
        else:
            raise ValueError(f'Invalid phase: {phase}')

    def training_step(self, batch: Any, batch_idx: int):
        # print(self.encoder.residual.conv.bias.grad)
        (host, mask, msg, container, residual, aug_container, aug_mask, seg_mask_logits,
         sync_object_patch, sync_mask_patch, msg_hat_logit, sync_mask_patch2) = self.shared_step(batch, phase='train')

        current_batch_idx = self.trainer.fit_loop.total_batch_idx
        enc_dec_optim, seg_sync_optim = self.optimizers()

        # Calculate loss
        mask_weight = (1 - mask) * 1e4 + torch.ones_like(mask)
        vis_loss = (self.loss_fn.pix_loss(residual*mask_weight, torch.zeros_like(residual)) +
                    LPIPS_FN(container, host, normalize=True).mean())
        vis_loss = vis_loss * self.loss_cfg.vis_weight if current_batch_idx > self.loss_cfg.vis_delay_batch else 0.
        # Decode loss
        msg_loss = F.binary_cross_entropy_with_logits(msg_hat_logit, msg.float()) * self.loss_cfg.msg_weight
        enc_dec_optim.zero_grad()
        self.manual_backward(vis_loss + msg_loss)
        enc_dec_optim.step()

        # Segment loss
        # seg_loss = F.binary_cross_entropy_with_logits(seg_mask_logits, aug_mask.float()) * self.loss_cfg.seg_weight
        seg_loss = K.losses.lovasz_hinge_loss(seg_mask_logits, aug_mask[:, 0]) * self.loss_cfg.seg_weight
        # Sync loss
        sync_loss = K.losses.lovasz_hinge_loss(sync_mask_patch2, mask[:, 0]) * self.loss_cfg.sync_weight
        seg_sync_optim.zero_grad()
        self.manual_backward(seg_loss + sync_loss)
        seg_sync_optim.step()

        self.log('train/vis_loss', vis_loss)
        self.log('train/msg_loss', msg_loss)
        self.log('train/seg_loss', seg_loss)
        self.log('train/sync_loss', sync_loss)

        # Calculate metrics
        with torch.no_grad():
            psnr = self.train_psnr(container, host)
            bar = self.train_bar(msg_hat_logit.sigmoid(), msg)
            iou = self.train_iou(seg_mask_logits.sigmoid(), aug_mask)
            iou2 = self.train_iou2(sync_mask_patch2.sigmoid(), mask)
        self.log('train/psnr', psnr)
        self.log('train/bar', bar)
        self.log('train/iou', iou)
        self.log('train/iou2', iou2)

        # Visualize
        image_size = self.model_cfg.image_shape[-2:]
        if current_batch_idx % 300 == 0:
            show_image = dict(
                bg1=torch.cat([
                    host[:1] * mask[:1], min_max_norm(residual[:1]), container[:1],
                    F.interpolate(aug_container[:1], size=tuple(image_size)),
                    F.interpolate(aug_container[:1] * aug_mask[:1], size=tuple(image_size)),
                    sync_object_patch[:1] * sync_mask_patch[:1],
                    sync_mask_patch2.sigmoid()[:1].repeat(1, 3, 1, 1)
                ], dim=0),
                bg2=torch.cat([
                    host[1:2] * mask[1:2], min_max_norm(residual[1:2]), container[1:2],
                    F.interpolate(aug_container[1:2], size=tuple(image_size)),
                    F.interpolate(aug_container[1:2] * aug_mask[1:2], size=tuple(image_size)),
                    sync_object_patch[1:2] * sync_mask_patch[1:2],
                    sync_mask_patch2.sigmoid()[1:2].repeat(1, 3, 1, 1)
                ], dim=0),
            )
            self.visualize2logger('train', show_image, current_batch_idx)

    def on_train_epoch_end(self) -> None:
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        (host, mask, msg, container, residual, aug_container, aug_mask, seg_mask_logits,
         sync_object_patch, sync_mask_patch, msg_hat_logit, sync_mask_patch2) = self.shared_step(batch, phase='val')
        # import pdb; pdb.set_trace()
        # Update metrics
        with torch.no_grad():
            self.val_psnr.update(container, host)
            self.val_ssim.update(container, host)
            self.val_bar.update(msg_hat_logit.sigmoid(), msg)
            self.val_iou.update(seg_mask_logits.sigmoid(), aug_mask)
            self.val_iou2.update(sync_mask_patch2.sigmoid(), mask)

        # Visualize
        image_size = self.model_cfg.image_shape[-2:]
        if batch_idx == 0:
            # import pdb; pdb.set_trace()
            show_image = dict(
                bg1=torch.cat([
                    host[:1] * mask[:1], min_max_norm(residual[:1]), container[:1],
                    F.interpolate(aug_container[:1], size=tuple(image_size)),
                    F.interpolate(aug_container[:1] * aug_mask[:1], size=tuple(image_size)),
                    sync_object_patch[:1] * sync_mask_patch[:1],
                    sync_mask_patch2.sigmoid()[:1].repeat(1, 3, 1, 1)
                ], dim=0),
            )
            self.visualize2logger('val/ob1', show_image, self.current_epoch)
        if batch_idx == 1:
            show_image = dict(
                bg1=torch.cat([
                    host[:1] * mask[:1], min_max_norm(residual[:1]), container[:1],
                    F.interpolate(aug_container[:1], size=tuple(image_size)),
                    F.interpolate(aug_container[:1] * aug_mask[:1], size=tuple(image_size)),
                    sync_object_patch[:1] * sync_mask_patch[:1],
                    sync_mask_patch2.sigmoid()[:1].repeat(1, 3, 1, 1)
                ], dim=0),
            )
            self.visualize2logger('val/ob2', show_image, self.current_epoch)

    def on_validation_epoch_end(self) -> None:
        psnr = self.val_psnr.compute()
        ssim = self.val_ssim.compute()
        bar = self.val_bar.compute()
        iou = self.val_iou.compute()
        iou2 = self.val_iou2.compute()

        self.val_psnr.reset()
        self.val_ssim.reset()
        self.val_bar.reset()
        self.val_iou.reset()
        self.val_iou2.reset()

        self.logger_instance.add_scalar('val/psnr', psnr, self.current_epoch)
        self.logger_instance.add_scalar('val/ssim', ssim, self.current_epoch)
        self.logger_instance.add_scalar('val/bar', bar, self.current_epoch)
        self.logger_instance.add_scalar('val/iou', iou, self.current_epoch)
        self.logger_instance.add_scalar('val/iou2', iou2, self.current_epoch)

        # Log to file
        log.info(f'Epoch {self.current_epoch}: psnr={psnr:.4f}, ssim={ssim:.4f}, bar={bar:.4f}, iou={iou:.4f}, iou2={iou2:.4f}')
        pass

    def on_test_start(self) -> None:
        seed_everything(42, workers=True)

    def test_step(self, batch: Any, batch_idx: int):
        (host, mask, msg, container, residual, aug_dict, seg_mask_logits_dict,
         sync_object_patch_dict, sync_mask_patch_dict, msg_hat_logit_dict) = self.shared_step(batch, phase='test')

        with torch.no_grad():
            self.test_psnr.update(container, host)
            self.test_ssim.update(container, host)

        for aug_name in self.augmenter.augment_types:
            aug_container, aug_mask = aug_dict[aug_name]
            seg_mask_logits = seg_mask_logits_dict[aug_name]
            msg_hat_logit = msg_hat_logit_dict[aug_name]

            # Update metrics
            with torch.no_grad():
                self.test_bars[aug_name].update(msg_hat_logit.sigmoid(), msg)
                self.test_ious[aug_name].update(seg_mask_logits.sigmoid(), aug_mask)

        # Visualize
        rnd_aug_name = random.choice(self.augmenter.augment_types)
        aug_container, aug_mask = aug_dict[rnd_aug_name]
        sync_object_patch = sync_object_patch_dict[rnd_aug_name]
        sync_mask_patch = sync_mask_patch_dict[rnd_aug_name]

        image_size = self.model_cfg.image_shape[-2:]
        if batch_idx % 100 == 0:
            show_image = dict(
                bg1=torch.cat([
                    host[:1] * mask[:1], min_max_norm(residual[:1]), container[:1],
                    F.interpolate(aug_container[:1], size=tuple(image_size)),
                    F.interpolate(aug_container[:1] * aug_mask[:1], size=tuple(image_size)),
                    sync_object_patch[:1] * sync_mask_patch[:1]
                ], dim=0),
            )
            self.visualize2logger('test', show_image, batch_idx)

    def on_test_epoch_end(self) -> None:
        psnr = self.test_psnr.compute()
        ssim = self.test_ssim.compute()
        self.test_psnr.reset()
        self.test_ssim.reset()
        self.logger_instance.add_scalar('test/psnr', psnr, self.current_epoch)
        self.logger_instance.add_scalar('test/ssim', ssim, self.current_epoch)

        log.info(f'Test: psnr={psnr:.4f}, ssim={ssim:.4f}')

        for aug_name in self.augmenter.augment_types:
            bar = self.test_bars[aug_name].compute()
            iou = self.test_ious[aug_name].compute()
            self.test_bars[aug_name].reset()
            self.test_ious[aug_name].reset()

            self.logger_instance.add_scalar(f'test/bar_{aug_name}', bar, self.current_epoch)
            self.logger_instance.add_scalar(f'test/iou_{aug_name}', iou, self.current_epoch)
            # Log to file
            log.info(f'{aug_name}: bar={bar:.4f}, iou={iou:.4f}')

    def configure_optimizers(self):
        enc_dec_optim = torch.optim.Adam([
            *self.encoder.parameters(),
            *self.decoder.parameters(),
        ], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)
        seg_sync_optim = torch.optim.Adam([
            *self.segmenter.parameters(),
            *self.syncor2.parameters(),
        ], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)
        return ({'optimizer': enc_dec_optim},
                {'optimizer': seg_sync_optim})

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
