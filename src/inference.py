#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ObjectWatermark
# @File         : inference.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/7/11 17:03

# Import lib here
import random
import time
import cv2
import hydra
import numpy as np
import pyrootutils
import torch
import torch.nn as nn
import lightning as L
import albumentations as A

from tqdm import tqdm
from collections import OrderedDict
from torchvision.transforms import functional as tvf
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from typing import List, Tuple
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

# from src.models.object_watermark import ObjectWatermark
from src.models.object_watermark2 import ObjectWatermark2
from src.datamodules import standard_object
from src import utils

log = utils.get_pylogger(__name__)


def plt_image_mask(image: np.ndarray, mask: np.ndarray):
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.show()
    plt.close()


def encode_objects(images: List[np.ndarray], masks: List[np.ndarray], msgs: List[np.ndarray]):
    pass


def get_transform_matrix_by_orb(source: np.ndarray, target: np.ndarray, verbose: bool = False):
    assert len(source.shape) == 2 and len(target.shape) == 2

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(source, None)
    kp2, des2 = orb.detectAndCompute(target, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    # Use the top N matches (e.g., 100) for alignment
    N = 100
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:N]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:N]]).reshape(-1, 1, 2)

    # Find the perspective transformation matrix (homography)
    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the source image to align with the target image
    aligned_image = cv2.warpPerspective(source, retval, (target.shape[1], target.shape[0]))

    # Display the aligned image
    if verbose:
        plt.subplot(1, 3, 1)
        plt.imshow(source, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(aligned_image, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow(target, cmap='gray')
        plt.show()
        plt.close()
    return retval


def encode_process(model):
    img_dir = '/sda1/Datasets/DUTS/DUTS-TE/DUTS-TE-Image'
    mask_dir = '/sda1/Datasets/DUTS/DUTS-TE/DUTS-TE-Mask'
    output_dir = '/sda1/Datasets/ObjectWatermark/DUTS-TE/'

    img_paths = sorted(Path(img_dir).glob('*.jpg'))
    mask_paths = sorted(Path(mask_dir).glob('*.png'))

    for img_path, mask_path in zip(img_paths[:10], mask_paths[:10]):
        img = cv2.imread(str(img_path))[..., ::-1]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        # plt_image_mask(img, mask)

        # standardize object
        std_img, std_mask = standard_object(img, mask)
        std_img_mask = np.concatenate([std_img, std_mask[..., None].repeat(3, axis=-1)], axis=1)
        cv2.imwrite(f'{output_dir}/std_img_mask/{img_path.name[11:]}',
                    std_img_mask[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])
        # plt_image_mask(img, mask)

        # encode watermark
        msg = torch.randint(0, 2, (32, ))
        wm_img_tsr, _ = model.encode(tvf.to_tensor(std_img), tvf.to_tensor(std_mask), msg)
        wm_img = wm_img_tsr.detach().cpu().numpy().transpose(1, 2, 0)
        wm_img = np.clip(wm_img*255, 0, 255).astype(np.uint8)

        # Get the transform matrix to inverse the standardization by keypoints matching
        matrix = get_transform_matrix_by_orb(std_mask, mask)
        # aligned_mask = cv2.warpPerspective(std_mask, matrix, (mask.shape[1], mask.shape[0]))

        # Warp back to the original shape
        aligned_wm_img = cv2.warpPerspective(wm_img, matrix, (img.shape[1], img.shape[0]))

        binary_mask = np.expand_dims(mask//255, axis=-1)
        aligned_wm_img = aligned_wm_img * binary_mask + img * (1 - binary_mask)
        msg_str = ''.join([str(i) for i in msg.tolist()])
        cv2.imwrite(f'{output_dir}/wm_object/{img_path.stem[11:]}_{msg_str}.jpg',
                    aligned_wm_img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])


def augment_process():
    img_dir = '/sda1/Datasets/ObjectWatermark/DUTS-TE/wm_object'
    mask_dir = '/sda1/Datasets/DUTS/DUTS-TE/DUTS-TE-Mask'
    output_dir = '/sda1/Datasets/ObjectWatermark/DUTS-TE/'

    img_paths = sorted(Path(img_dir).glob('*.jpg'))
    mask_paths = sorted(Path(mask_dir).glob('*.png'))

    affine_func = A.Affine(scale=[0.9, 1.1], translate_px=[10, 10], rotate=[-10, 10], fit_output=True, p=1.0)

    for img_path, mask_path in zip(img_paths[:10], mask_paths[:10]):
        img = cv2.imread(str(img_path))[..., ::-1]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        affine_outputs = affine_func(image=img, mask=mask)
        aff_img, aff_mask = affine_outputs['image'], affine_outputs['mask']
        cv2.imwrite(f'{output_dir}/aff_object/{img_path.name}', aff_img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(f'{output_dir}/aff_mask/{mask_path.name[11:]}', aff_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def decode_process(model):
    img_dir = '/sda1/Datasets/ObjectWatermark/DUTS-TE/aff_object'
    # mask_dir = '/sda1/Datasets/ObjectWatermark/DUTS-TE/aff_mask'
    mask_dir = '/sda1/Datasets/ObjectWatermark/DUTS-TE/selfreformer'
    output_dir = '/sda1/Datasets/ObjectWatermark/DUTS-TE/'

    img_paths = sorted(Path(img_dir).glob('*.jpg'))
    mask_paths = sorted(Path(mask_dir).glob('*.jpg'))

    acc_bits = []
    for img_path, mask_path in zip(img_paths[:10], mask_paths[:10]):
        img = cv2.imread(str(img_path))[..., ::-1]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

        # standardize object
        std_img, std_mask = standard_object(img, mask)
        std_img_mask = np.concatenate([std_img, std_mask[..., None].repeat(3, axis=-1)], axis=1)
        cv2.imwrite(f'{output_dir}/std_aff_img_mask_selfreformer/{mask_path.stem}.jpg',
                    std_img_mask[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])
        # plt_image_mask(std_img, std_mask)

        # decode watermark
        msg_hat_logits = model.decode(tvf.to_tensor(std_img), tvf.to_tensor(std_mask))
        msg_hat = torch.sigmoid(msg_hat_logits).gt(0.5).detach().cpu().numpy()

        msg_gt = img_path.stem.split('_')[-1]
        acc_bits.append(np.sum(msg_hat == np.array([int(i) for i in msg_gt])))
    print('bit acc per image: ', [acc_bits/32 for acc_bits in acc_bits])
    print('average bit acc: ', np.mean(acc_bits)/32)


# @utils.task_wrapper
def inference(cfg: DictConfig):
    object_watermark: ObjectWatermark2 = hydra.utils.instantiate(cfg.model)
    state_dict: OrderedDict = torch.load(cfg.ckpt_path)['state_dict']
    key_list = list(state_dict.keys())
    lpips_fn_keys = [key for key in key_list if key[:8] == 'lpips_fn']
    for key in lpips_fn_keys:
        del state_dict[key]
    object_watermark.load_state_dict(state_dict)

    if cfg.is_encode:
        encode_process(object_watermark)
    elif cfg.is_decode:
        decode_process(object_watermark)
    elif cfg.is_augment:
        augment_process()
    else:
        print(f'Nothing to do, exit...')


def encode_process2(model: ObjectWatermark2, cfg: DictConfig):
    img_dir = Path(cfg.input_dir) / 'Image'
    mask_dir = Path(cfg.input_dir) / 'Mask'

    time_stamp = str(int(time.time()))
    container_dir = Path(cfg.output_dir) / time_stamp / 'Container'
    container_dir.mkdir(parents=True, exist_ok=True)
    residual_dir = Path(cfg.output_dir) / time_stamp / 'Residual'
    residual_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir = Path(cfg.output_dir) / time_stamp / 'Mask'
    out_mask_dir.symlink_to(mask_dir)

    avg_psnr, avg_ssim = [], []
    mask_paths = sorted(mask_dir.glob('*.png'))[:200]
    for mask_path in mask_paths:
        img_path = img_dir / f'{mask_path.stem}.jpg'
        # import pdb; pdb.set_trace()
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

        # TODO: standardize object

        # Encode
        msg = torch.randint(0, 2, (cfg.model.model_cfg.msg_len, ))
        wm_img_tsr, residual_tsr = model.encode(tvf.to_tensor(img).to(model.device),
                                                tvf.to_tensor(mask).to(model.device),
                                                msg.to(model.device))
        wm_img = wm_img_tsr.detach().cpu().numpy().transpose(1, 2, 0)
        wm_img = np.clip(wm_img*255, 0, 255).astype(np.uint8)
        object_mask = np.expand_dims(mask//255, axis=-1).repeat(3, axis=-1)
        wm_img = wm_img * object_mask + img * (1 - object_mask)

        residual_tsr = (residual_tsr - residual_tsr.min()) / (residual_tsr.max() - residual_tsr.min())
        residual = residual_tsr.detach().cpu().numpy().transpose(1, 2, 0)
        residual = np.clip(residual*255, 0, 255).astype(np.uint8)

        psnr = peak_signal_noise_ratio(img, wm_img)
        ssim = structural_similarity(img, wm_img, channel_axis=2)
        log.info(f"{img_path.stem}'s psnr: {psnr}, ssim: {ssim}")
        avg_psnr.append(psnr)
        avg_ssim.append(ssim)

        msg_str = ''.join([str(i) for i in msg.tolist()])
        cv2.imwrite(f'{container_dir}/{mask_path.stem}_{msg_str}.jpg', wm_img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(f'{residual_dir}/{mask_path.stem}_{msg_str}.png', residual[..., ::-1])
    log.info(f"Average psnr: {np.mean(avg_psnr)}, ssim: {np.mean(avg_ssim)}, total: {len(avg_psnr)}")


def decode_process2(model: ObjectWatermark2, cfg: DictConfig):
    img_dir = Path(cfg.input_dir) / 'Container'
    mask_dir = Path(cfg.input_dir) / 'Mask'

    avg_acc = []
    for img_path in img_dir.glob('*.jpg'):
        true_img_stem = img_path.stem[:img_path.stem.rfind('_')]
        mask_path = mask_dir / f'{true_img_stem}.png'
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        msg_str = img_path.stem.split('_')[-1]
        msg = np.array([int(i) for i in msg_str])

        pred_msg = model.decode(tvf.to_tensor(img).to(model.device),
                                tvf.to_tensor(mask).to(model.device))
        pred_msg = pred_msg.detach().cpu().numpy()
        acc = np.mean(pred_msg == msg)
        log.info(f"{img_path.stem}'s acc: {acc}")
        avg_acc.append(acc)
    log.info(f"Average acc: {np.mean(avg_acc)}")


def copy_paste_attack2(model: ObjectWatermark2, cfg: DictConfig):
    container_dir = Path(cfg.input_dir) / 'Container'
    mask_dir = Path(cfg.input_dir) / 'Mask'
    bg_dir = Path(cfg.input_dir) / 'Background'

    time_stamp = str(int(time.time()))
    output_dir = Path(cfg.output_dir) / time_stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir = output_dir / 'Mask'
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    attacked_dir = output_dir / 'Attacked'
    attacked_dir.mkdir(parents=True, exist_ok=True)

    container_paths = sorted(container_dir.glob('*.jpg'))
    bg_paths = sorted(bg_dir.glob('*.jpg'))
    for container_path in container_paths:
        true_container_stem = container_path.stem[:container_path.stem.rfind('_')]
        mask_path = mask_dir / f'{true_container_stem}.png'
        bg_path = random.choice(bg_paths)

        container = cv2.cvtColor(cv2.imread(str(container_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        bg = cv2.cvtColor(cv2.imread(str(bg_path)), cv2.COLOR_BGR2RGB)
        pass

def inference2(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model: ObjectWatermark2 = hydra.utils.instantiate(cfg.model).to(device)
    state_dict: OrderedDict = torch.load(cfg.ckpt_path)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    if cfg.is_encode:
        encode_process2(model, cfg)
    elif cfg.is_decode:
        decode_process2(model, cfg)
    elif cfg.is_augment:
        raise NotImplementedError
    else:
        print(f'Nothing to do, exit...')


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)
    inference2(cfg)


if __name__ == '__main__':
    main()
