import time
import os
import sys

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms

import numpy as np
import pandas as pd
import cv2

import math
from timm.models.vision_transformer import vit_base_patch16_384
from timm.models.MAE import MAE
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import TusimpleMAE
from train_mae import train_mae

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    save_path = "./logs"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_dataset_file = os.path.join(r"D:\Tusimple\clips", 'train.txt')
    val_dataset_file = os.path.join(r"D:\Tusimple\test_set\clips", 'val.txt')

    resize_height = 384
    resize_width = 384

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    train_dataset = TusimpleMAE(train_dataset_file, transform=data_transforms['train'])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TusimpleMAE(val_dataset_file, transform=data_transforms['val'])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    dataloaders = {
        'train' : train_loader,
        'val' : val_loader
    }
    dataset_sizes = {'train': len(train_loader.dataset), 'val' : len(val_loader.dataset)}

    model = MAE()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model, log = train_mae(model, optimizer, scheduler=None, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=DEVICE, num_epochs=10)
    df=pd.DataFrame({'epoch':[],'training_loss':[],'val_loss':[]})
    df['epoch'] = log['epoch']
    df['training_loss'] = log['training_loss']
    df['val_loss'] = log['val_loss']

    train_log_save_filename = os.path.join(save_path, 'training_log.csv')
    df.to_csv(train_log_save_filename, columns=['epoch','training_loss','val_loss'], header=True,index=False,encoding='utf-8')
    print("training log is saved: {}".format(train_log_save_filename))
    
    model_save_filename = os.path.join(save_path, 'best_model.pth')
    torch.save(model.state_dict(), model_save_filename)
    print("model is saved: {}".format(model_save_filename))


if __name__ == '__main__':
    train()