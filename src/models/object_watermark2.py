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
from src.metrics import Accumulation
from omegaconf import DictConfig
from lpips import LPIPS
from src.models.discriminator import Discriminator
from src.models.syncor.crop_out import CropOut
from typing import Tuple, List, Any, Dict

from src.utils import get_pylogger

log = get_pylogger(__name__)


def min_max_norm(x: torch.Tensor):
    return (x - x.min()) / (x.max() - x.min())


def cal_mask_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor, threshold: float = 0.5):
    assert pred_mask.shape == target_mask.shape and target_mask.dim() == 2
    assert pred_mask.min() >= 0 and pred_mask.max() <= 1
    assert target_mask.min() >= 0 and target_mask.max() <= 1
    pred_mask = pred_mask.ge(threshold).int()
    target_mask = target_mask.ge(threshold).int()

    intersection = (pred_mask & target_mask).sum()
    union = (pred_mask | target_mask).sum()
    return intersection.float() / union.float() if union > 0 else 0.


LPIPS_FN = LPIPS(net='vgg')
DISCRIMINATOR = Discriminator(in_channels=3)
MASK_ACCUMULATOR = None
RANDOM_SCALE = K.augmentation.RandomAffine(degrees=0, scale=(0.5, 1), same_on_batch=False, p=1.0).to('cuda')
RANDOM_ROTATE = K.augmentation.RandomRotation(degrees=25, same_on_batch=False, p=1.0).to('cuda')


class ObjectWatermark2(LightningModule):
    def __init__(self, model_cfg: DictConfig,
                 encoder: nn.Module,
                 augmenter: Any,
                 segmenter: nn.Module,
                 syncor: nn.Module,
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
        # self.train_par = MultilabelAccuracy(num_labels=512*512, threshold=0.5)
        # self.train_par = MultilabelAccuracy(num_labels=256*256, threshold=0.5)

        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_bar = MultilabelAccuracy(num_labels=msg_len, threshold=0.5)
        self.val_iou = BinaryJaccardIndex(threshold=0.5)
        # TODO: measure the IoU of the synced object instead of the segmentation
        self.val_sync_iou = BinaryJaccardIndex(threshold=0.5)
        self.val_correct_ratio = Accumulation()     # measure the corrected object ratio
        # self.val_par = MultilabelAccuracy(num_labels=512*512, threshold=0.5)
        # self.val_par = MultilabelAccuracy(num_labels=256*256, threshold=0.5)

        self.test_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_bars = nn.ModuleDict({
            key: MultilabelAccuracy(num_labels=msg_len, threshold=0.5) for key in self.augmenter.augment_types})
        self.test_ious = nn.ModuleDict({
            key: BinaryJaccardIndex(threshold=0.5) for key in self.augmenter.augment_types})
        # For the combine distortion, which is random selected from these distortions
        self.test_bars['Combine'] = MultilabelAccuracy(num_labels=msg_len, threshold=0.5)
        self.test_ious['Combine'] = BinaryJaccardIndex(threshold=0.5)
        self.test_correct_ratio = Accumulation()
        # self.test_par = MultilabelAccuracy(num_labels=256*256, threshold=0.5)

        # Log model arch to file
        from torchinfo import summary
        log.info(f'Encoder Summary:\n'
                 f'{summary(self.encoder, input_size=((1, 3, 256, 256), (1, 1, 256, 256), (1, msg_len)), depth=5, verbose=0)}')
        log.info(f'Decoder Summary:\n'
                 f'{summary(self.decoder, input_size=((1, 3, 256, 256), (1, 1, 256, 256)), depth=5, verbose=0)}')

    def forward(self, x: torch.Tensor):
        raise NotImplementedError(f'Please use `encode` or `decode` methods instead.')

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
        self.val_correct_ratio.reset()

        # Using global variable to avoid saving model in pickle
        global LPIPS_FN, DISCRIMINATOR, MASK_ACCUMULATOR
        LPIPS_FN = LPIPS_FN.to(self.device)
        DISCRIMINATOR = DISCRIMINATOR.to(self.device)
        MASK_ACCUMULATOR = torch.zeros(self.model_cfg.image_shape[-2:]).to(self.device)  # Avoid zero division

    def shared_step(self, batch: Any, phase: str = 'train'):
        host, mask, msg, bg_img = batch
        # host: (batch_size, 3, H, W)
        # mask: (batch_size, 1, H, W)
        # msg: (batch_size, num_backgrounds, msg_len)
        # bg_img: (batch_size, num_backgrounds, 3, 512, 512)
        num_backgrounds = bg_img.shape[1]
        host = host.unsqueeze(dim=1).repeat(1, num_backgrounds, 1, 1, 1)
        mask = mask.unsqueeze(dim=1).repeat(1, num_backgrounds, 1, 1, 1)

        host = host.reshape(-1, *host.shape[2:])
        mask = mask.reshape(-1, *mask.shape[2:])
        msg = msg.reshape(-1, *msg.shape[2:])
        bg_img = bg_img.reshape(-1, *bg_img.shape[2:])

        if self.model_cfg.preprocess_bbox_crop_out:
            # Used for CropOut sync manner only.
            bbox_crop_out = CropOut(self.model_cfg.image_shape[-2:])
            host, mask = bbox_crop_out(host, mask)
        # if self.model_cfg.preprocess_random_translate:
        #     # Have implemented in the dataloader, so do nothing here
        #     pass
        if self.model_cfg.preprocess_random_scale:
            global RANDOM_SCALE
            host = RANDOM_SCALE(host)
            mask = RANDOM_SCALE(mask.float(), params=RANDOM_SCALE._params)
        if self.model_cfg.preprocess_random_rotate:
            global RANDOM_ROTATE
            host = RANDOM_ROTATE(host)
            mask = RANDOM_ROTATE(mask.float(), params=RANDOM_ROTATE._params)

        # Embed object or the whole image
        if not self.model_cfg.mask_object_on_train and phase == 'train':
            mask = torch.ones_like(mask)

        # Encode
        container, residual = self.encoder(host, mask, msg, normalize=True)

        def seg_sync_decode(_aug_container, _aug_mask):
            # Segment
            # Att: aug_container is detached from the compute graph
            seg_mask_logits = self.segmenter(_aug_container.detach(), normalize=True)

            # Sync
            gt_masks = _aug_mask.ge(0.5).int()
            pred_masks = seg_mask_logits.sigmoid().ge(0.5).int()
            if self.total_steps >= self.model_cfg.adopt_seg_delay_step:
                object_mask = []
                for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                    iou = cal_mask_iou(pred_mask[0], gt_mask[0])
                    if iou > self.model_cfg.iou_thresh_for_segment_pred:
                        object_mask.append(pred_mask)
                    else:
                        object_mask.append(gt_mask)
                object_mask = torch.stack(object_mask, dim=0)
            else:
                object_mask = gt_masks

            sync_object_patch, sync_mask_patch = self.syncor(_aug_container, object_mask)
            # Decode
            msg_hat_logit = self.decoder(sync_object_patch, sync_mask_patch, normalize=True)
            return seg_mask_logits, sync_object_patch, sync_mask_patch, msg_hat_logit

        if phase == 'train' or phase == 'val':
            aug_dict = self.augmenter(container, mask, bg_img, self.total_steps)
            aug_container, aug_mask = list(aug_dict.values())[0]
            return (host, mask, msg, container, residual,
                    aug_container, aug_mask, *seg_sync_decode(aug_container, aug_mask))
        elif phase == 'test':
            aug_dict = self.augmenter(container, mask, bg_img, self.total_steps, return_all=True)
            seg_mask_logits_dict, sync_object_patch_dict, sync_mask_patch_dict, msg_hat_logit_dict = {}, {}, {}, {}
            for aug_name, (aug_container, aug_mask) in aug_dict.items():
                (seg_mask_logits_dict[aug_name], sync_object_patch_dict[aug_name],
                 sync_mask_patch_dict[aug_name], msg_hat_logit_dict[aug_name]) = seg_sync_decode(aug_container,
                                                                                                 aug_mask)
            return (host, mask, msg, container, residual, aug_dict, seg_mask_logits_dict,
                    sync_object_patch_dict, sync_mask_patch_dict, msg_hat_logit_dict)
        else:
            raise ValueError(f'Invalid phase: {phase}')

    def training_step(self, batch: Any, batch_idx: int):
        # print(self.encoder.residual.conv.bias.grad)
        (host, mask, msg, container, residual, aug_container, aug_mask, seg_mask_logits,
         sync_object_patch, sync_mask_patch, msg_hat_logit) = self.shared_step(batch, phase='train')
        ob, wm_ob = host * mask, container * mask

        # Calculate loss and optimize
        # Optimize discriminator
        if batch_idx % 3 == 0 and self.loss_cfg.vis_weight3:
            dis_loss = DISCRIMINATOR.optimize_parameters(wm_ob, ob)
            self.log('train/dis_loss', dis_loss)

        enc_dec_optim, seg_sync_optim = self.optimizers()

        # erosion_kernel = torch.ones(5, 5, device=self.device)  # erase the border effect
        # mask_weight = (1 - K.morphology.erosion(mask, erosion_kernel)) * 2 + torch.ones_like(mask)
        # print(residual.min().item(), residual.max().item(), residual.mean().item())
        # import matplotlib.pyplot as plt
        # plt.subplot(131)
        # plt.imshow(ob[0].permute(1, 2, 0).cpu().detach().numpy())
        # plt.subplot(132)
        # print(edge_weight[0].min(), edge_weight[0].max())
        # plt.imshow(edge_weight[0, 0].cpu().detach().numpy(), cmap='gray')
        # plt.subplot(133)
        # plt.imshow(residual[0].permute(1, 2, 0).cpu().detach().numpy())
        # plt.show()
        # plt.close()
        if self.loss_cfg.accu_mask_weight:
            global MASK_ACCUMULATOR
            cur_mask_accumulator = mask.sum(dim=[0, 1]) / (mask.shape[0] * mask.shape[1])
            MASK_ACCUMULATOR = MASK_ACCUMULATOR * 0.9 + cur_mask_accumulator * 0.1
            accu_mask_weight = MASK_ACCUMULATOR.unsqueeze(dim=0).unsqueeze(dim=0)
            # Ensure the background will not be decreased to zero
            accu_mask_weight = accu_mask_weight + torch.ones_like(accu_mask_weight)
        else:
            accu_mask_weight = torch.ones_like(residual)
        vis_loss1 = self.loss_fn.pix_loss(residual * accu_mask_weight,
                                          torch.zeros_like(residual)) * self.loss_cfg.vis_weight1

        if self.loss_cfg.vis_weight2:
            if self.loss_cfg.lpips_mask:
                vis_loss2 = LPIPS_FN(wm_ob, ob, normalize=True).mean() * self.loss_cfg.vis_weight2
            else:
                vis_loss2 = LPIPS_FN(container, host, normalize=True).mean() * self.loss_cfg.vis_weight2
            vis_loss2 = 0. if torch.isnan(vis_loss2) else vis_loss2
        else:
            vis_loss2 = 0.

        if self.loss_cfg.vis_weight3:
            vis_loss3 = -DISCRIMINATOR(wm_ob).mean() * self.loss_cfg.vis_weight3
            vis_loss3 = 0. if torch.isnan(vis_loss3) else vis_loss3
        else:
            vis_loss3 = 0.
        vis_loss = (vis_loss1 + vis_loss2 + vis_loss3) * self.loss_cfg.vis_weight
        if self.total_steps <= self.loss_cfg.vis_delay_step:
            vis_loss = 0.

        # Decode loss
        msg_loss = F.binary_cross_entropy_with_logits(msg_hat_logit, msg.float()) * self.loss_cfg.msg_weight

        enc_dec_optim.zero_grad()
        self.manual_backward(vis_loss + msg_loss)
        nn.utils.clip_grad_value_([*self.encoder.parameters(),
                                   *self.decoder.parameters()], clip_value=10)
        enc_dec_optim.step()

        # Segment loss
        # seg_loss = F.binary_cross_entropy_with_logits(seg_mask_logits, aug_mask.float()) * self.loss_cfg.seg_weight
        seg_loss = K.losses.lovasz_hinge_loss(seg_mask_logits, aug_mask[:, 0]) * self.loss_cfg.seg_weight
        seg_loss = torch.tensor(0., requires_grad=True, device=self.device) if torch.isnan(seg_loss) else seg_loss
        seg_sync_optim.zero_grad()
        self.manual_backward(seg_loss)
        nn.utils.clip_grad_value_([*self.segmenter.parameters()], clip_value=10)
        seg_sync_optim.step()

        self.log('train/vis_loss1', vis_loss1)
        self.log('train/vis_loss2', vis_loss2)
        self.log('train/vis_loss3', vis_loss3)
        self.log('train/vis_loss', vis_loss)
        self.log('train/msg_loss', msg_loss)
        self.log('train/seg_loss', seg_loss)

        # Calculate metrics
        with torch.no_grad():
            psnr = self.train_psnr(wm_ob, ob)
            bar = self.train_bar(msg_hat_logit.sigmoid(), msg)
            iou = self.train_iou(seg_mask_logits.sigmoid(), aug_mask)
        self.log('train/psnr', psnr)
        self.log('train/bar', bar)
        self.log('train/iou', iou)

        # Visualize
        image_size = self.model_cfg.image_shape[-2:]
        if self.total_steps % 300 == 0:
            show_image = dict(
                ob1=torch.cat([
                    host[:1] * mask[:1] + torch.ones_like(mask[:1]) * (1 - mask[:1]),
                    min_max_norm(residual[:1] if residual.shape[1] == 3 else residual[:1].repeat(1, 3, 1, 1)),
                    container[:1],
                    F.interpolate(aug_container[:1], size=tuple(image_size)),
                    F.interpolate(aug_container[:1] * aug_mask[:1], size=tuple(image_size)),
                    sync_object_patch[:1] * sync_mask_patch[:1] + torch.ones_like(sync_mask_patch[:1]) * (1 - sync_mask_patch[:1]),
                ], dim=0),
                ob2=torch.cat([
                    host[1:2] * mask[1:2] + torch.ones_like(mask[1:2]) * (1 - mask[1:2]),
                    min_max_norm(residual[1:2] if residual.shape[1] == 3 else residual[1:2].repeat(1, 3, 1, 1)),
                    container[1:2],
                    F.interpolate(aug_container[1:2], size=tuple(image_size)),
                    F.interpolate(aug_container[1:2] * aug_mask[1:2], size=tuple(image_size)),
                    sync_object_patch[1:2] * sync_mask_patch[1:2] + torch.ones_like(sync_mask_patch[1:2]) * (1 - sync_mask_patch[1:2]),
                ], dim=0),
            )
            self.visualize2logger('train', show_image, self.total_steps)

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
         sync_object_patch, sync_mask_patch, msg_hat_logit) = self.shared_step(batch, phase='val')
        ob, wm_ob = host * mask, container * mask

        # Update metrics
        with torch.no_grad():
            self.val_psnr.update(wm_ob, ob)
            self.val_ssim.update(wm_ob, ob)
            self.val_bar.update(msg_hat_logit.sigmoid(), msg)
            self.val_iou.update(seg_mask_logits.sigmoid(), aug_mask)

            correct_num = 0
            for pred_mask, gt_mask in zip(sync_mask_patch, mask):
                iou = cal_mask_iou(pred_mask[0], gt_mask[0])
                if iou > self.model_cfg.iou_thresh_for_correct_ratio:
                    correct_num += 1
            self.val_correct_ratio.update(correct_num, mask.shape[0])

        # Visualize
        image_size = self.model_cfg.image_shape[-2:]
        if batch_idx == 0:
            # import pdb; pdb.set_trace()
            show_image = dict(
                ob1=torch.cat([
                    host[:1] * mask[:1] + torch.ones_like(mask[:1]) * (1 - mask[:1]),
                    min_max_norm(residual[:1] if residual.shape[1] == 3 else residual[:1].repeat(1, 3, 1, 1)),
                    container[:1],
                    F.interpolate(aug_container[:1], size=tuple(image_size)),
                    F.interpolate(aug_container[:1] * aug_mask[:1], size=tuple(image_size)),
                    sync_object_patch[:1] * sync_mask_patch[:1] + torch.ones_like(sync_mask_patch[:1]) * (1 - sync_mask_patch[:1]),
                ], dim=0),
                ob2=torch.cat([
                    host[1:2] * mask[1:2] + torch.ones_like(mask[1:2]) * (1 - mask[1:2]),
                    min_max_norm(residual[1:2] if residual.shape[1] == 3 else residual[1:2].repeat(1, 3, 1, 1)),
                    container[1:2],
                    F.interpolate(aug_container[1:2], size=tuple(image_size)),
                    F.interpolate(aug_container[1:2] * aug_mask[1:2], size=tuple(image_size)),
                    sync_object_patch[1:2] * sync_mask_patch[1:2] + torch.ones_like(sync_mask_patch[1:2]) * (1 - sync_mask_patch[1:2]),
                ], dim=0)
            )
            self.visualize2logger('val', show_image, self.total_steps)

    def on_validation_epoch_end(self) -> None:
        psnr = self.val_psnr.compute()
        ssim = self.val_ssim.compute()
        bar = self.val_bar.compute()
        iou = self.val_iou.compute()
        correct_ratio = self.val_correct_ratio.compute()

        self.val_psnr.reset()
        self.val_ssim.reset()
        self.val_bar.reset()
        self.val_iou.reset()
        self.val_correct_ratio.reset()

        self.logger_instance.add_scalar('val/psnr', psnr, self.total_steps)
        self.logger_instance.add_scalar('val/ssim', ssim, self.total_steps)
        self.logger_instance.add_scalar('val/bar', bar, self.total_steps)
        self.logger_instance.add_scalar('val/iou', iou, self.total_steps)
        self.logger_instance.add_scalar('val/correct_ratio', correct_ratio, self.total_steps)

        # Log for checkpoint callback
        self.log('hp_metric', bar + (psnr / 50))
        self.log('step', self.total_steps, logger=False)
        self.log('psnr', psnr, logger=False)
        self.log('bar', bar, logger=False)

        # Log to file
        log.info(f'Step: {self.total_steps}: psnr={psnr:.4f}, ssim={ssim:.4f}, '
                 f'bar={bar:.4f}, iou={iou:.4f}, correct_ratio={correct_ratio:.4f}')
        pass

    def on_test_start(self) -> None:
        seed_everything(42, workers=True)

    def test_step(self, batch: Any, batch_idx: int):
        (host, mask, msg, container, residual, aug_dict, seg_mask_logits_dict,
         sync_object_patch_dict, sync_mask_patch_dict, msg_hat_logit_dict) = self.shared_step(batch, phase='test')
        ob, wm_ob = host * mask, container * mask

        with torch.no_grad():
            self.test_psnr.update(wm_ob, ob)
            self.test_ssim.update(wm_ob, ob)

        rnd_aug_name = random.choice(self.augmenter.augment_types)
        for aug_name in self.augmenter.augment_types:
            aug_container, aug_mask = aug_dict[aug_name]
            seg_mask_logits = seg_mask_logits_dict[aug_name]
            msg_hat_logit = msg_hat_logit_dict[aug_name]

            # Update metrics
            with torch.no_grad():
                self.test_bars[aug_name].update(msg_hat_logit.sigmoid(), msg)
                self.test_ious[aug_name].update(seg_mask_logits.sigmoid(), aug_mask)

                # Combine distortion
                if aug_name == rnd_aug_name:
                    self.test_bars['Combine'].update(msg_hat_logit.sigmoid(), msg)
                    self.test_ious['Combine'].update(seg_mask_logits.sigmoid(), aug_mask)

        # Visualize
        aug_container, aug_mask = aug_dict[rnd_aug_name]
        sync_object_patch = sync_object_patch_dict[rnd_aug_name]
        sync_mask_patch = sync_mask_patch_dict[rnd_aug_name]

        with torch.no_grad():
            correct_num = 0
            for pred_mask, gt_mask in zip(sync_mask_patch, mask):
                iou = cal_mask_iou(pred_mask[0], gt_mask[0])
                if iou > self.model_cfg.iou_thresh_for_correct_ratio:
                    correct_num += 1
            self.test_correct_ratio.update(correct_num, mask.shape[0])

        image_size = self.model_cfg.image_shape[-2:]
        show_image = dict(
            ob1=torch.cat([
                host[:1] * mask[:1] + torch.ones_like(mask[:1]) * (1 - mask[:1]),
                min_max_norm(residual[:1] if residual.shape[1] == 3 else residual[:1].repeat(1, 3, 1, 1)),
                container[:1],
                F.interpolate(aug_container[:1], size=tuple(image_size)),
                F.interpolate(aug_container[:1] * aug_mask[:1], size=tuple(image_size)),
                sync_object_patch[:1] * sync_mask_patch[:1] + torch.ones_like(sync_mask_patch[:1]) * (1 - sync_mask_patch[:1]),
            ], dim=0),
            ob2=torch.cat([
                host[1:2] * mask[1:2] + torch.ones_like(mask[1:2]) * (1 - mask[1:2]),
                min_max_norm(residual[1:2] if residual.shape[1] == 3 else residual[1:2].repeat(1, 3, 1, 1)),
                container[1:2],
                F.interpolate(aug_container[1:2], size=tuple(image_size)),
                F.interpolate(aug_container[1:2] * aug_mask[1:2], size=tuple(image_size)),
                sync_object_patch[1:2] * sync_mask_patch[1:2] + torch.ones_like(sync_mask_patch[1:2]) * (1 - sync_mask_patch[1:2]),
            ], dim=0),
            ob3=torch.cat([
                host[2:3] * mask[2:3] + torch.ones_like(mask[2:3]) * (1 - mask[2:3]),
                min_max_norm(residual[2:3] if residual.shape[1] == 3 else residual[2:3].repeat(1, 3, 1, 1)),
                container[2:3],
                F.interpolate(aug_container[2:3], size=tuple(image_size)),
                F.interpolate(aug_container[2:3] * aug_mask[2:3], size=tuple(image_size)),
                sync_object_patch[2:3] * sync_mask_patch[2:3] + torch.ones_like(sync_mask_patch[2:3]) * (1 - sync_mask_patch[2:3]),
            ], dim=0)
        )
        self.visualize2logger('test', show_image, batch_idx)

    def on_test_epoch_end(self) -> None:
        psnr = self.test_psnr.compute()
        ssim = self.test_ssim.compute()
        correct_ratio = self.test_correct_ratio.compute()
        self.test_psnr.reset()
        self.test_ssim.reset()
        self.test_correct_ratio.reset()
        self.logger_instance.add_scalar('test/psnr', psnr, self.total_steps)
        self.logger_instance.add_scalar('test/ssim', ssim, self.total_steps)
        self.logger_instance.add_scalar('test/correct_ratio', correct_ratio, self.total_steps)

        log.info(f'Test: psnr={psnr:.4f}, ssim={ssim:.4f}, correct_ratio={correct_ratio:.4f}')

        for aug_name in ['Combine'] + list(self.augmenter.augment_types):
            bar = self.test_bars[aug_name].compute()
            iou = self.test_ious[aug_name].compute()
            self.test_bars[aug_name].reset()
            self.test_ious[aug_name].reset()

            self.logger_instance.add_scalar(f'test/bar_{aug_name}', bar, self.total_steps)
            self.logger_instance.add_scalar(f'test/iou_{aug_name}', iou, self.total_steps)
            # Log to file
            log.info(f'{aug_name}: bar={bar:.4f}, iou={iou:.4f}')

    def configure_optimizers(self):
        enc_dec_optim = torch.optim.Adam([
            *self.encoder.parameters(),
            *self.decoder.parameters(),
        ], lr=self.model_cfg.lr, betas=(0.9, 0.999), weight_decay=1e-5)

        seg_sync_optim = torch.optim.Adam([
            *self.segmenter.parameters(),
        ], lr=self.model_cfg.lr, betas=(0.9, 0.999), weight_decay=1e-5)
        return ({'optimizer': enc_dec_optim},
                {'optimizer': seg_sync_optim})

    @property
    def logger_instance(self):
        return self.logger.experiment

    @property
    def total_steps(self):
        return self.trainer.fit_loop.total_batch_idx

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
