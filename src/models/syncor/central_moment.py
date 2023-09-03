#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ObjectWatermark
# @File         : central_moment.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/7/29 17:46
#
# Import lib here
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry import rotate
from kornia.augmentation import RandomRotation, RandomTranslate, RandomAffine
from .crop_out import CropOut
from PIL import Image

from typing import Tuple


class CentralMoment(nn.Module):
    def __init__(self, output_size: Tuple[int, int],
                 random_translate: bool = False,
                 random_scale: bool = False,
                 random_rotate: bool = False,
                 bbox_crop_out: bool = False):
        super().__init__()
        self.output_size = tuple(output_size)
        self.random_translate = random_translate
        self.random_scale = RandomAffine(degrees=0., scale=(0.5, 1.0), same_on_batch=False, p=1.) if random_scale else False
        self.random_rotate = RandomRotation(degrees=25., same_on_batch=False, p=1.) if random_rotate else False

        if bbox_crop_out:
            self.bbox_crop_out = CropOut(output_size)
        else:
            self.bbox_crop_out = None

    def standardize(self, image: torch.Tensor,
                    mask: np.ndarray):
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        if len(contours) == 0:
            std_mask = cv2.resize(mask, self.output_size, interpolation=cv2.INTER_LINEAR)
            std_mask = cv2.threshold(std_mask, 127, 255, cv2.THRESH_BINARY)[1]
            std_img = F.interpolate(image.unsqueeze(0), self.output_size, mode='bilinear', align_corners=False)[0]
            return std_img, std_mask
        moments = cv2.moments(contours[0])

        # Calculate the center of the object
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # Pad so that the object is in the center of the image for rotation
        height, width = image.shape[-2:]
        right_pad, left_pad, top_pad, bottom_pad = 0, 0, 0, 0
        if cx > width - cx:
            right_pad = (cx - width + cx)
        else:
            left_pad = (width - cx - cx)
        if cy > height - cy:
            bottom_pad = (cy - height + cy)
        else:
            top_pad = (height - cy - cy)

        image = F.pad(image, (left_pad, right_pad, top_pad, bottom_pad), mode='constant', value=0)
        mask = np.pad(mask, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')

        # import matplotlib.pyplot as plt
        # plt.subplot(1, 2, 1)
        # plt.imshow(mask, cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(image.permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        # Calculate the angle of the object
        theta = np.degrees(np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02']) / 2)

        # Rotate the image and mask
        rotated_mask = np.array(Image.fromarray(mask).rotate(theta, resample=2))
        rotated_image = rotate(image.unsqueeze(0), torch.tensor([theta], dtype=torch.float32, device=image.device))[0]

        # plt.subplot(1, 2, 1)
        # plt.imshow(rotated_mask, cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(rotated_image.permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        moments = cv2.moments(contours[0])
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # Get the max distance from the centroid to the boundary
        x, y, w, h = cv2.boundingRect(contours[0])
        left, right, top, bottom = x, x + w, y, y + h
        max_dist = max(abs(cx - left), abs(cx - right), abs(cy - top), abs(cy - bottom))

        rot_height, rot_width = rotated_mask.shape
        top_pad, left_pad, bottom_pad, right_pad = 0, 0, 0, 0
        if cy < max_dist:
            top_pad = max_dist - cy
        if cx < max_dist:
            left_pad = max_dist - cx
        if rot_height - cy < max_dist:
            bottom_pad = max_dist - (rot_height - cy)
        if rot_width - cx < max_dist:
            right_pad = max_dist - (rot_width - cx)
        pad_mask = np.pad(rotated_mask, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')
        pad_img = F.pad(rotated_image, (left_pad, right_pad, top_pad, bottom_pad), mode='constant', value=0)

        # plt.subplot(1, 2, 1)
        # plt.imshow(pad_mask, cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(pad_img.permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        crop_mask = pad_mask[cy + top_pad - max_dist:cy + top_pad + max_dist,
                    cx + left_pad - max_dist:cx + left_pad + max_dist]
        crop_img = pad_img[:, cy + top_pad - max_dist:cy + top_pad + max_dist,
                   cx + left_pad - max_dist:cx + left_pad + max_dist]

        # plt.subplot(1, 2, 1)
        # plt.imshow(crop_mask, cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(crop_img.permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        std_mask = cv2.resize(crop_mask, self.output_size, interpolation=cv2.INTER_LINEAR)
        std_mask = cv2.threshold(std_mask, 127, 255, cv2.THRESH_BINARY)[1]
        std_img = F.interpolate(crop_img.unsqueeze(0), self.output_size, mode='bilinear')[0]

        if self.random_translate:
            # Here, we manually translate the object while ensuring it is not out of the image
            ind_h, ind_w = np.where(std_mask == 255)
            margin_u, margin_d = np.min(ind_h), std_mask.shape[0] - 1 - np.max(ind_h)
            margin_l, margin_r = np.min(ind_w), std_mask.shape[1] - 1 - np.max(ind_w)
            if margin_u + margin_d > 0:
                shift_h = np.random.randint(-margin_u, margin_d)
            else:
                shift_h = 0
            if margin_l + margin_r > 0:
                shift_w = np.random.randint(-margin_l, margin_r)
            else:
                shift_w = 0
            # print(shift_w, shift_h, np.mean(mask))
            std_mask = np.roll(std_mask, shift=(shift_h, shift_w), axis=(0, 1))
            std_img = torch.roll(std_img, shifts=(shift_h, shift_w), dims=(-2, -1))
        return std_img, std_mask

    def forward(self, images: torch.Tensor,
                masks: torch.Tensor):
        batch_size = images.shape[0]
        masks = masks.detach().ge(0.5).int().cpu().numpy()
        std_imgs, std_masks = [], []
        for idx in range(batch_size):
            img = images[idx]
            mask = np.clip(masks[idx, 0] * 255, 0, 255).astype(np.uint8)
            std_img, std_mask = self.standardize(img, mask)
            std_mask = torch.tensor(std_mask/255., dtype=torch.int, device=images.device).unsqueeze(0)
            std_imgs.append(std_img)
            std_masks.append(std_mask)
        std_imgs = torch.stack(std_imgs, dim=0)
        std_masks = torch.stack(std_masks, dim=0)

        # Random translate, scale and rotate for each image, please do not do it in batch
        if self.random_translate:
            # Have implemented in standardize, so do nothing here
            pass
        if self.random_scale:
            std_imgs = self.random_scale(std_imgs)
            std_masks = self.random_scale(std_masks.float(), params=self.random_scale._params)
        if self.random_rotate:
            std_imgs = self.random_rotate(std_imgs)
            std_masks = self.random_rotate(std_masks.float(), params=self.random_rotate._params)

        if self.bbox_crop_out:
            std_imgs, std_masks = self.bbox_crop_out(std_imgs, std_masks)
        return std_imgs, std_masks


def run():
    central_moment = CentralMoment((256, 256))
    image = torch.randn(2, 3, 12, 12)
    mask = torch.randn(2, 1, 12, 12)
    std_img, std_mask = central_moment(image, mask)
    import pdb; pdb.set_trace()
    print('pause')
    pass


if __name__ == '__main__':
    run()
