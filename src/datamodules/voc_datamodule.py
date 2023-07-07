#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : voc_datamodule.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/5/29 22:15

# Import lib here
import numpy as np
import random
import cv2
import torch
import albumentations as A
import matplotlib.pyplot as plt
from tqdm import tqdm
from albumentations.augmentations.geometric.functional import smallest_max_size
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from typing import Tuple, Optional, List, Dict, Any


def save_max_mask():
    # Find and save the max area mask of segmentation object
    save_dir = Path('/sda1/Datasets/VOC2012/SegmentationMaxObject')
    mask_dir = Path('/sda1/Datasets/VOC2012/SegmentationObject')
    mask_paths = sorted(mask_dir.glob('*.png'))
    for path in tqdm(mask_paths):
        mask = Image.open(path).convert('L')
        mask = np.array(mask).squeeze()
        bin_counts = np.bincount(mask.flatten())
        sorted_bin_counts = np.sort(bin_counts)
        # Get the frequency value that it not belong to background(0) and border(220)
        idx = 0
        freq_value = 0
        while freq_value == 0 or freq_value == 220:
            idx += 1
            freq_value = np.where(bin_counts == sorted_bin_counts[-idx])[0][0]
        mask = (mask == freq_value).astype(np.uint8) * 255
        mask = Image.fromarray(mask)
        mask.save(save_dir/path.name)
    pass


def filter_mask():
    # Random crop so that the area of the object(mask) is greater than 0.25 of the total area
    img_dir = Path('/sda1/Datasets/VOC2012/JPEGImages')
    mask_dir = Path('/sda1/Datasets/VOC2012/SegmentationMaxObject')
    output_dir = Path('/sda1/Datasets/VOC2012/Object/')
    transform = A.Compose([
        A.RandomCrop(256, 256),
        A.HorizontalFlip(),
    ])
    mask_paths = sorted(mask_dir.glob('*.png'))
    for path in tqdm(mask_paths):
        img_path = img_dir / path.name.replace('.png', '.jpg')
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(path).convert('L'))
        img = smallest_max_size(img, 320, cv2.INTER_NEAREST)
        mask = smallest_max_size(mask, 320, cv2.INTER_NEAREST)

        for idx in range(4):
            transformed = transform(image=img, mask=mask)
            transformed_mask = transformed['mask']
            if np.sum(transformed_mask == 255) >= 256 * 256 * 0.25:
                transformed_mask = Image.fromarray(transformed_mask)
                transformed_mask.save(output_dir/'masks'/f'{path.stem}_{idx}.png')

                transformed_img = transformed['image']
                transformed_img = Image.fromarray(transformed_img)
                transformed_img.save(output_dir/'images'/f'{path.stem}_{idx}.png')


def standardize_image_mask_v0():
    # standardize the image and mask by moving the object to the center of the image, and rotate the image by PCA, then
    # pad and resize the image to 256*256
    ob_img_dir = Path('/sda1/Datasets/VOC2012/Object/images')
    ob_mask_dir = Path('/sda1/Datasets/VOC2012/Object/masks')
    output_dir = Path('/sda1/Datasets/VOC2012/Object/standardized')

    img_paths = sorted(ob_img_dir.glob('*.png'))
    for path in img_paths[3:]:
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(str(ob_mask_dir/path.name)), cv2.COLOR_BGR2GRAY)

        # Get the contour of the object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mean, eigen_vectors, eigen_values = cv2.PCACompute2(contours[0].squeeze().astype(float), np.zeros(0))
        cntr = (int(mean[0][0]), int(mean[0][1]))
        # cv2.drawMarker(img, cntr, (0, 0, 255), cv2.MARKER_STAR, 20, 2)
        # cv2.drawMarker(mask, cntr, 0, cv2.MARKER_STAR, 20, 2)

        # draw contour and center
        # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        # cv2.drawMarker(img, cntr, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        # plt.subplot(1, 2, 1)
        # plt.imshow(img)
        # cv2.drawMarker(mask, cntr, 0, cv2.MARKER_STAR, 20, 2)
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask, cmap='gray')
        # plt.show()
        # plt.close()

        # Pad image so that cntr is the center of the image
        w1, w2 = cntr[0], img.shape[1] - cntr[0]
        h1, h2 = cntr[1], img.shape[0] - cntr[1]
        right_pad, left_pad, top_pad, bottom_pad = 0, 0, 0, 0
        if w1 > w2:
            right_pad = (w1-w2)
        else:
            left_pad = (w2-w1)
        if h1 > h2:
            bottom_pad = (h1-h2)
        else:
            top_pad = (h2-h1)

        img = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
        mask = np.pad(mask, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')

        # cv2.circle(img, (img.shape[1]//2, img.shape[0]//2), 10, (0, 0, 255), 1)
        # plt.subplot(1, 2, 1)
        # plt.imshow(img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask, cmap='gray')
        # plt.show()
        # plt.close()

        # No need for resize
        # Resize image and mask to 256*256
        # img = cv2.resize(img, (256, 256))
        # mask = cv2.resize(mask, (256, 256))

        # Check whether the object is in the center of the image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mean, eigen_vectors, eigen_values = cv2.PCACompute2(contours[0].squeeze().astype(float), None)

        cv2.drawMarker(img, (int(mean[0][0]), int(mean[0][1])), (0, 0, 255), cv2.MARKER_STAR, 20, 2)
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)

        draw_mask = mask.copy()
        cv2.drawMarker(draw_mask, (int(mean[0][0]), int(mean[0][1])), 0, cv2.MARKER_CROSS, 20, 2)
        M = cv2.moments(contours[0])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.drawMarker(draw_mask, (cx, cy), 0, cv2.MARKER_STAR, 20, 2)
        plt.imshow(draw_mask, cmap='gray')
        plt.show()
        plt.close()

        # Rotate image by PCA
        angle = np.arctan2(eigen_vectors[0][1], eigen_vectors[0][0]) * 180 / np.pi
        # angle = 45
        rotated_img = np.array(Image.fromarray(img).rotate(angle, expand=True))
        rotated_mask = np.array(Image.fromarray(mask).rotate(angle, expand=True))

        # plt.subplot(1, 2, 1)
        # plt.imshow(rotated_img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(rotated_mask)
        # plt.show()
        # plt.close()

        # Get the bounding box of the object and draw
        contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mean, eigen_vectors, eigen_values = cv2.PCACompute2(contours[0].squeeze().astype(float), np.zeros(0))
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(rotated_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.rectangle(rotated_mask, (x, y), (x+w, y+h), 255, 3)
        cntr = (int(mean[0][0]), int(mean[0][1]))
        cv2.drawMarker(rotated_img, cntr, color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
        cv2.drawMarker(rotated_mask, cntr, color=0, markerType=cv2.MARKER_STAR, markerSize=20, thickness=3)
        cv2.circle(rotated_mask, (rotated_mask.shape[1]//2, rotated_mask.shape[0]//2), 10, 0, 1)

        plt.subplot(1, 2, 1)
        plt.imshow(rotated_img)
        plt.subplot(1, 2, 2)
        plt.imshow(rotated_mask, cmap='gray')
        plt.show()
        plt.close()
        input()
    pass

def standardize_image_mask():
    # Standardize the image and mask by moments

    ob_img_dir = Path('/sda1/Datasets/VOC2012/Object/images')
    ob_mask_dir = Path('/sda1/Datasets/VOC2012/Object/masks')
    output_dir = Path('/sda1/Datasets/VOC2012/StandardObjecct/')

    img_paths = sorted(ob_img_dir.glob('*.png'))
    for path in tqdm(img_paths):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(str(ob_mask_dir/path.name)), cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        moments = cv2.moments(contours[0])

        # Calculate the center of the object
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # Pad so that the object is in the center of the image for rotation
        right_pad, left_pad, top_pad, bottom_pad = 0, 0, 0, 0
        if cx > img.shape[1]-cx:
            right_pad = (cx - img.shape[1] + cx)
        else:
            left_pad = (img.shape[1] - cx - cx)
        if cy > img.shape[0]-cy:
            bottom_pad = (cy - img.shape[0] + cy)
        else:
            top_pad = (img.shape[0] - cy - cy)

        img = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
        mask = np.pad(mask, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')

        # Calculate the angle of the object
        theta = np.degrees(np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02']) / 2)
        if theta < 0:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
            theta = -theta
        # Rotate the image and mask
        rotated_img = np.array(Image.fromarray(img).rotate(theta, resample=2, expand=True))
        rotated_mask = np.array(Image.fromarray(mask).rotate(theta, resample=2, expand=True))

        # Check if the object is in the center
        # contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        # moments = cv2.moments(contours[0])
        # cx = int(moments['m10'] / moments['m00'])
        # cy = int(moments['m01'] / moments['m00'])
        # cv2.drawMarker(rotated_img, (cx, cy), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
        # cv2.drawMarker(rotated_mask, (cx, cy), color=0, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
        # cv2.circle(rotated_img, (rotated_img.shape[1]//2, rotated_img.shape[0]//2), 10, 0, 1)
        # cv2.circle(rotated_mask, (rotated_mask.shape[1]//2, rotated_mask.shape[0]//2), 10, 0, 1)
        # plt.subplot(1, 2, 1)
        # plt.imshow(rotated_img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(rotated_mask, cmap='gray')
        # plt.show()
        # plt.close()
        # input()

        # Get the centroid of the object
        contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        moments = cv2.moments(contours[0])
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # Get the max distance from the centroid to the boundary
        x, y, w, h = cv2.boundingRect(contours[0])
        left, right, top, bottom = x, x+w, y, y+h
        max_dist = max(abs(cx-left), abs(cx-right), abs(cy-top), abs(cy-bottom))

        top_pad, left_pad, bottom_pad, right_pad = 0, 0, 0, 0
        if cy < max_dist:
            top_pad = max_dist - cy
        if cx < max_dist:
            left_pad = max_dist - cx
        if rotated_img.shape[0]-cy < max_dist:
            bottom_pad = max_dist - (rotated_img.shape[0]-cy)
        if rotated_img.shape[1]-cx < max_dist:
            right_pad = max_dist - (rotated_img.shape[1]-cx)
        pad_img = np.pad(rotated_img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
        pad_mask = np.pad(rotated_mask, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')

        # plt.subplot(1, 2, 1)
        # plt.imshow(pad_img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(pad_mask, cmap='gray')
        # plt.show()
        # plt.close()

        crop_img = pad_img[cy+top_pad-max_dist:cy+top_pad+max_dist, cx+left_pad-max_dist:cx+left_pad+max_dist]
        crop_mask = pad_mask[cy+top_pad-max_dist:cy+top_pad+max_dist, cx+left_pad-max_dist:cx+left_pad+max_dist]

        std_img = cv2.resize(crop_img, (256, 256), interpolation=cv2.INTER_LINEAR)
        std_mask = cv2.resize(crop_mask, (256, 256), interpolation=cv2.INTER_LINEAR)
        std_mask = cv2.threshold(std_mask, 127, 255, cv2.THRESH_BINARY)[1]

        Image.fromarray(std_img).save(output_dir / 'images' / f'{path.name}')
        Image.fromarray(std_mask).save(output_dir / 'masks' / f'{path.name}')

        # Crop the image and mask
        # cropped_img = rotated_img[y:y+h, x:x+w]
        # cropped_mask = rotated_mask[y:y+h, x:x+w]

        # Rescale the image and mask so that the longer side is 256
        # longer_side = max(cropped_img.shape[:2])
        # scale = 256 / longer_side
        # cropped_img = cv2.resize(cropped_img, (round(cropped_img.shape[1]*scale), round(cropped_img.shape[0]*scale)))
        # cropped_mask = cv2.resize(cropped_mask, (round(cropped_mask.shape[1]*scale), round(cropped_mask.shape[0]*scale)))

        # Pad the image and mask so that it is a square
        # height, width = cropped_img.shape[:2]
        # if height > width and height == 256:
        #     pad = (256 - width) // 2
        #     cropped_img = np.pad(cropped_img, ((0, 0), (pad, 256-width-pad), (0, 0)), mode='constant')
        #     cropped_mask = np.pad(cropped_mask, ((0, 0), (pad, 256-width-pad)), mode='constant')
        # elif width > height and width == 256:
        #     pad = (256 - height) // 2
        #     cropped_img = np.pad(cropped_img, ((pad, 256-height-pad), (0, 0), (0, 0)), mode='constant')
        #     cropped_mask = np.pad(cropped_mask, ((pad, 256-height-pad), (0, 0)), mode='constant')
        # else:
        #     raise ValueError('The image is not square after rescaling')

        # Pad the image and mask so that it is a square
        # height, width = rotated_img.shape[:2]
        # if height > width:
        #     pad = (height - width) // 2
        #     pad_img = np.pad(rotated_img, ((0, 0), (pad, height-width-pad), (0, 0)), mode='constant')
        #     pad_mask = np.pad(rotated_mask, ((0, 0), (pad, height-width-pad)), mode='constant')
        # else:
        #     pad = (width - height) // 2
        #     pad_img = np.pad(rotated_img, ((pad, width-height-pad), (0, 0), (0, 0)), mode='constant')
        #     pad_mask = np.pad(rotated_mask, ((pad, width-height-pad), (0, 0)), mode='constant')

        # Rescale the image and mask to 256
        # std_img = cv2.resize(pad_img, (256, 256))
        # std_mask = cv2.resize(pad_mask, (256, 256))

        # plt.subplot(1, 2, 1)
        # plt.imshow(crop_img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(crop_mask, cmap='gray')
        # plt.show()
        # plt.close()
        # input()


class VOCDataset0(Dataset):
    def __init__(self, data_dir: str,
                 img_size: Tuple[int, int] = (400, 400),
                 msg_len: int = 10,
                 stage: str = 'train'):
        super().__init__()
        assert stage.lower() in ['train', 'val', 'test']
        data_dir = Path(data_dir)
        self.img_size = img_size
        self.msg_len = msg_len
        self.img_dir = data_dir / 'JPEGImages'
        max_obj_dir = data_dir / 'SegmentationMaxObject'
        max_obj_paths = sorted(max_obj_dir.glob('*.png'))
        num = len(max_obj_paths)
        if stage.lower() == 'test':
            max_obj_paths = max_obj_paths[:100]
        elif stage.lower() == 'val':
            max_obj_paths = max_obj_paths[100:200]
        else:
            max_obj_paths = max_obj_paths[200:]
        self.max_obj_paths = max_obj_paths

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(self.img_size, pad_if_needed=True),
        ])

    def __len__(self):
        return len(self.max_obj_paths)

    def __getitem__(self, idx):
        img_path = self.img_dir / self.max_obj_paths[idx].name.replace('.png', '.jpg')
        max_obj_path = self.max_obj_paths[idx]
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(max_obj_path).convert('L'))
        img_mask = np.concatenate([img, mask[..., None]], axis=-1)
        img_mask = self.transform(img_mask)
        img, mask = img_mask[:3], img_mask[3:]
        if torch.sum(mask) <= self.img_size[0] * self.img_size[1] * 0.25:
            return self.__getitem__((idx+1) % len(self.max_obj_paths))
        msg = torch.randint(0, 2, (self.msg_len,))
        return img, mask, msg


class VOCDataset(Dataset):
    def __init__(self, data_dir: str,
                 msg_len: int = 32,
                 stage: str = 'train'):
        super().__init__()
        assert stage.lower() in ['train', 'val', 'test']
        data_dir = Path(data_dir)
        self.msg_len = msg_len
        self.img_dir = data_dir / 'StandardObject' / 'images'
        mask_dir = data_dir / 'StandardObject' / 'masks'
        mask_paths = list(mask_dir.glob('*.png'))
        random.shuffle(mask_paths)
        if stage.lower() == 'test':
            mask_paths = mask_paths[:300]
        elif stage.lower() == 'val':
            mask_paths = mask_paths[300:600]
        else:
            mask_paths = mask_paths[600:]
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        img_path = self.img_dir / self.mask_paths[idx].name
        mask_path = self.mask_paths[idx]
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        mask = transforms.ToTensor()(Image.open(mask_path).convert('L'))
        msg = torch.randint(0, 2, (self.msg_len,))
        return img, mask, msg


class VOCDataModule(LightningDataModule):
    def __init__(self, dataset_cfg: DictConfig, dataloader_cfg: DictConfig):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg

        self.train_data: Optional[VOCDataset] = None
        self.valid_data: Optional[VOCDataset] = None
        self.test_data: Optional[VOCDataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = None) -> None:
        if not self.train_data:
            self.train_data = VOCDataset(**self.dataset_cfg, stage='train')
        if not self.valid_data:
            self.valid_data = VOCDataset(**self.dataset_cfg, stage='val')
        if not self.test_data:
            self.test_data = VOCDataset(**self.dataset_cfg, stage='test')

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, **self.dataloader_cfg)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, **self.dataloader_cfg)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, **self.dataloader_cfg)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


def run():
    from src.utils.image_tools import image_show
    import pdb
    # save_max_mask()
    # filter_mask()
    standardize_image_mask()
    # voc_dataset = VOCDataset('/sda1/Datasets/VOC2012')
    # for img, mask, msg in voc_dataset:
    #     image_show(img)
    #     image_show(mask)
    #     print(torch.sum(mask))
    #     print(msg)
    #     pdb.set_trace()
    #     print('pause')


if __name__ == '__main__':
    run()
