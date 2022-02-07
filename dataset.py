import glob
import random
import os
import json
import math
import time
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
import torchvision.transforms as transforms



class ImageDataset(Dataset):
    def __init__(self, root="../scratch/CVUSA/dataset/splits/", transforms_street=[transforms.ToTensor(),],transforms_sat=[transforms.ToTensor(),], mode='train', zooms=[20]):
        self.zooms = zooms
        self.transforms_street = transforms.Compose(transforms_street)
        self.transforms_sat = transforms.Compose(transforms_sat)

        if mode == "val":
            self.file = root + "val-19zl.csv"
        elif mode == "train":
            self.file = root + "train-19zl.csv"
        else:
            raise RuntimeError("no such mode")

        self.data_list = []
        
        csv_file = open(self.file)
        for l in csv_file.readlines():
            data = l.strip().split(",")
            data.pop(2)
            self.data_list.append(data)

        csv_file.close()

    def __getitem__(self, index):
        satellite_file, ground_file = self.data_list[index]

        satellite = self.transforms_sat(Image.open(os.path.join("../scratch/CVUSA/dataset", satellite_file)))
        ground = self.transforms_street(Image.open(os.path.join("../scratch/CVUSA/dataset", ground_file)))

        return {'satellite':satellite, 'ground':ground}


    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":

    STREET_IMG_WIDTH = 616
    STREET_IMG_HEIGHT = 112
    SATELLITE_IMG_WIDTH = 256
    SATELLITE_IMG_HEIGHT = 256

    transforms_sate = [transforms.Resize((SATELLITE_IMG_WIDTH, SATELLITE_IMG_HEIGHT)),
                    transforms.ToTensor()
                    ]
    transforms_street = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
                    transforms.ToTensor()
                    ]

    dataloader = DataLoader(ImageDataset(transforms_street=transforms_street,transforms_sat=transforms_sate,mode="train"),\
         batch_size=4, shuffle=True, num_workers=8)
    
    # print(len(dataloader))
    total_time = 0
    start = time.time()
    for i,b in enumerate(dataloader):
        end = time.time()
        elapse = end - start
        total_time += elapse
        # print(elapse)
        start = end
        # print("===========================")
        # print(b["ground"].shape)
        # print(b["satellite"].shape)
        # print("===========================")
        time.sleep(2)

    print(total_time / i)
