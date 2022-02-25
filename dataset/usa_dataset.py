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
    def __init__(self, data_dir="../scratch/CVUSA/dataset/", transforms_street=[transforms.ToTensor(),],transforms_sat=[transforms.ToTensor(),], mode='train', is_polar=True):
        self.data_dir = data_dir
        self.transforms_street = transforms.Compose(transforms_street)
        self.transforms_sat = transforms.Compose(transforms_sat)

        if mode == "val" or mode == "dev":
            self.file = os.path.join(self.data_dir, "splits", "val-19zl.csv")
        elif mode == "train":
            self.file = os.path.join(self.data_dir, "splits", "train-19zl.csv")
        else:
            raise RuntimeError("no such mode")

        self.data_list = []
        
        csv_file = open(self.file)
        for l in csv_file.readlines():
            data = l.strip().split(",")
            data.pop(2)
            if is_polar:
                data[0] = data[0].replace("bingmap", "polarmap")
                data[0] = data[0].replace("jpg", "png")
            self.data_list.append(data)

        csv_file.close()

        if mode == "dev":
            self.data_list = self.data_list[0:200]

    def __getitem__(self, index):
        satellite_file, ground_file = self.data_list[index]
        # x = Image.open(os.path.join(self.data_dir, satellite_file))
        # y = Image.open(os.path.join(self.data_dir, ground_file))
        # print(x.size)
        # print(y.size)
        satellite = self.transforms_sat(Image.open(os.path.join(self.data_dir, satellite_file)))
        ground = self.transforms_street(Image.open(os.path.join(self.data_dir, ground_file)))

        return {'satellite':satellite, 'ground':ground}


    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":

    STREET_IMG_WIDTH = 671
    STREET_IMG_HEIGHT = 122
    # SATELLITE_IMG_WIDTH = 256
    # SATELLITE_IMG_HEIGHT = 256
    SATELLITE_IMG_WIDTH = 671
    SATELLITE_IMG_HEIGHT = 122

    transforms_sate = [transforms.Resize((SATELLITE_IMG_WIDTH, SATELLITE_IMG_HEIGHT)),
                    transforms.ToTensor()
                    ]
    transforms_street = [transforms.Resize((STREET_IMG_WIDTH, STREET_IMG_HEIGHT)),
                    transforms.ToTensor()
                    ]

    dataloader = DataLoader(ImageDataset(transforms_street=transforms_street,transforms_sat=transforms_sate, data_dir='/mnt/CVUSA/dataset',mode="train", is_polar=True),\
         batch_size=4, shuffle=True, num_workers=8)
    
    # print(len(dataloader))
    total_time = 0
    start = time.time()
    for i,b in enumerate(dataloader):
        end = time.time()
        elapse = end - start
        print("===========================")
        print(b["ground"].shape)
        print(b["satellite"].shape)
        print("===========================")
        time.sleep(2)

    print(total_time / i)
