import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np

from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

import random

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask


class TusimpleMAE(Dataset):
    def __init__(self, dataset, transform=None):
        self._gt_img_list = []

        self.transform = transform
        
        self.generate_mask = RandomMaskingGenerator(24, 0.75)

        with open(dataset, 'r') as file:
            for _info in file:
                _info = _info.strip('\n')
                _info = _info.split(' ')[0]
                info_tmp = os.path.join(dataset, _info)
                self._gt_img_list.append(info_tmp)

        # self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self._gt_img_list))
        random.shuffle(c)
        self._gt_img_list = zip(*c)

    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        # load all

        img = Image.open(self._gt_img_list[idx])
        # optional transformations
        if self.transform:
            img = self.transform(img)

        mask = self.generate_mask()

        # we could split the instance label here, each instance in one channel (basically a binary mask for each)
        return img, mask