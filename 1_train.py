import os
import cv2
import torch
import random
import argparse
import numpy as np
from PIL import Image

# data_manager
from torch import optim
import torchvision 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# model load
from model.deepcrack import DeepCrack
from torch.utils.data import Dataset

class CrackDataset(Dataset): 
    def __init__(self, folder, transform, img_size=[512,512]): 

        self.folder = folder 
        image_folder = '%s/image' % self.folder
        self.transform = transform

        self.img_size = img_size 
        self.filename = [f for f in os.listdir(image_folder)] 

    def __len__(self): 
        return 2
#        return len(self.filename)

    def __getitem__(self, idx): 

        filename = self.filename[idx]

        image_file = '%s/image/%s' % (self.folder, filename) 
        gt_file = '%s/gt/%s.bmp' % (self.folder, filename.split('.')[0]) 
       
        img = cv2.imread(image_file) 
        gt = cv2.imread(gt_file, 0)

        img = Image.fromarray(img)
        gt = Image.fromarray(gt) 

        img = self.transform(img) 
        gt = self.transform(gt) 

        return img, gt 


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='DeepCrack') 
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=100)
    args = parser.parse_args()

    model = DeepCrack()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) 

    loss = torch.nn.BCEWithLogitsLoss() 

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(), 
        transforms.RandomCrop((512,512)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])
   
    root_folder = 'datasets/CrackTree260' 

    dataset = CrackDataset(root_folder, transform) 
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch, shuffle=True) 

    for epoch in range(args.epoch): 

        err = [] 

        for inputs, targets in data_loader: 

            inputs_ = torchvision.utils.make_grid(inputs, 10, 1) 
            targets_ = torchvision.utils.make_grid(targets, 10, 1) 

            optimizer.zero_grad() 
            pred_output, pred_fuse5, pred_fuse4, pred_fuse3, pred_fuse2, pred_fuse1 = model(inputs)

            output_loss = loss(pred_output.view(-1,1), targets.view(-1,1)) 

            fuse5_loss = loss(pred_fuse5.view(-1,1), targets.view(-1,1)) 
            fuse4_loss = loss(pred_fuse4.view(-1,1), targets.view(-1,1)) 
            fuse3_loss = loss(pred_fuse3.view(-1,1), targets.view(-1,1)) 
            fuse2_loss = loss(pred_fuse2.view(-1,1), targets.view(-1,1)) 
            fuse1_loss = loss(pred_fuse1.view(-1,1), targets.view(-1,1)) 

            total_loss = output_loss + fuse5_loss + fuse4_loss + fuse3_loss + fuse2_loss + fuse1_loss 

            err.append(total_loss.item()) 
            total_loss.backward() 
            optimizer.step() 

        print('Epoch %2d : %f' % (epoch, sum(err)/len(err)))
