#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Project      : ObjectWatermark
# @File         : coco_datamodule.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/5/22 15:51

# Import lib here
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from pycocotools.coco import COCO
from omegaconf import DictConfig
from typing import Tuple, Optional, List, Dict, Any


# CoCo dataset
class CoCoDataset(Dataset):
    def __init__(self, data_dir: str, cat_ids: List[int], stage: str = 'train'):
        super().__init__()
        assert stage.lower() in ['train', 'val']

        img_dir = Path(data_dir) / f'images/{stage}2017'
        ann_file = Path(data_dir) / f'annotations/instances_{stage}2017.json'
        self.coco = COCO(ann_file)

        self.cat_ids = cat_ids
        self.img_ids = self.coco.getImgIds(catIds=cat_ids)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, areaRng=[], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        sorted(anns, key=lambda x: x['area'], reverse=True)
        if anns[0]['area'] < 1000:
            return self.__getitem__(idx + 1)
        else:
            ann = anns[0]
            img_path = self.img_dir / self.coco.loadImgs(img_id)[0]['file_name']
            img = Image.open(img_path).convert('RGB')
            img = transforms.ToTensor()(img)
        return img, ann


def run():
    from src.utils.image_tools import image_show
    import pdb
    coco_dataset = CoCoDataset('/sda1/Datasets/CoCo/', cat_ids=[3])
    for img, anns in coco_dataset:
        plt.imshow(img.numpy())
        anns = [anns]
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            import matplotlib.pyplot as plt
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Polygon

            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        plt.show()
        print('pause')
    pass


if __name__ == '__main__':
    run()
