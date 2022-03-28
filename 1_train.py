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
        return len(self.filename)

    def __getitem__(self, idx): 

        filename = self.filename[idx]

        image_file = '%s/image/%s' % (self.folder, filename) 
        gt_file = '%s/gt/%s.bmp' % (self.folder, filename.split('.')[0]) 
       
        img = cv2.imread(image_file) 
        gt = cv2.imread(gt_file, 0)[:,:,None]

        data = np.concatenate([img, gt], 2) 
        data = Image.fromarray(data)
        data = self.transform(data)
       
        img, gt = data[:3], data[3:]
        img = (img - 0.5) / 0.5  # normalization 

        return img, gt 


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='DeepCrack') 
    parser.add_argument('--batch', type=int, default=4) 
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None)
    args = parser.parse_args()

    model_file = 'model.pt' 

    model = DeepCrack()

    if torch.cuda.device_count() > 1:
        if args.gpu_ids == None:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            device = torch.device('cuda:0')
        else:
            print("Let's use", len(args.gpu_ids), "GPUs!")
            device = torch.device("cuda:" + str(args.gpu_ids[0]))
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) 

    # load model
    if os.path.isfile(model_file):
        print('Load saved model')
        checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])  # , strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        init_epoch = -1

    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


    loss = torch.nn.BCEWithLogitsLoss() 

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(), 
        transforms.RandomCrop((512,512)),
        transforms.ToTensor(), 
    ])
   
    root_folder = 'datasets/CrackTree260' 

    dataset = CrackDataset(root_folder, transform) 
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch, shuffle=True) 

    for epoch in range(init_epoch+1, args.epoch): 

        err = [] 

        for inputs, targets in data_loader: 

            inputs = inputs.to(device) 
            targets = targets.to(device) 

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

        model_dictionary = {'epoch': epoch,
            'state_dict': list(model.children())[0].state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        torch.save(model_dictionary, model_file)
