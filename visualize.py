import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm
import os
import numpy as np
import argparse
import json
import scipy.io as sio
from utils.utils import ReadConfig, ValidateAll, validatenp
import PIL
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from models.SAFA_TR_VIS import SAFA_TR_VIS



args_do_not_overide = ['data_dir', 'verbose', 'dataset']

def ValidateOne(distArray, topK):
    acc = 0.0
    dataAmount = 0.0
    for i in range(distArray.shape[0]):
        groundTruths = distArray[i,i]
        pred = torch.sum(distArray[:,i] < groundTruths)
        if pred < topK:
            acc += 1.0
        dataAmount += 1.0
    return acc / dataAmount

def ValidateAll(streetFeatures, satelliteFeatures):
    distArray = 2 - 2 * torch.matmul(satelliteFeatures, torch.transpose(streetFeatures, 0, 1))
    topOnePercent = int(distArray.shape[0] * 0.01) + 1
    valAcc = torch.zeros((1, topOnePercent))
    for i in range(topOnePercent):
        valAcc[0,i] = ValidateOne(distArray, i)
    
    return valAcc

def GetBestModel(path):
    all_files = os.listdir(path)
    if "epoch_last" in all_files:
        all_files.remove("epoch_last")
    config_files =  list(filter(lambda x: x.startswith('epoch_'), all_files))
    config_files = sorted(list(map(lambda x: int(x.split("_")[1]), config_files)), reverse=True)
    best_epoch = config_files[0]
    return os.path.join('epoch_'+str(best_epoch), 'epoch_'+str(best_epoch)+'.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to model weights')
    parser.add_argument('--test_img_path', type=str, help='path to test images')

    opt = parser.parse_args()

    config = ReadConfig(opt.model_path)
    for k,v in config.items():
        if k in args_do_not_overide:
            continue
        setattr(opt, k, v)
    
    print(opt)

    batch_size = opt.batch_size
    number_SAFA_heads = opt.SAFA_heads

    if opt.no_polar:
        SATELLITE_IMG_WIDTH = 256
        SATELLITE_IMG_HEIGHT = 256
        polar_transformation = False
    else:
        SATELLITE_IMG_WIDTH = 671
        SATELLITE_IMG_HEIGHT = 122
        polar_transformation = True
    print("SATELLITE_IMG_WIDTH:",SATELLITE_IMG_WIDTH)
    print("SATELLITE_IMG_HEIGHT:",SATELLITE_IMG_HEIGHT)

    STREET_IMG_WIDTH = 671
    STREET_IMG_HEIGHT = 122

    if opt.pos == "learn_pos":
        pos = "learn_pos"
    else:
        pos = None


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SAFA_TR_VIS(safa_heads=number_SAFA_heads, tr_heads=opt.TR_heads, tr_layers=opt.TR_layers, dropout = opt.dropout, d_hid=opt.TR_dim, is_polar=polar_transformation, pos=pos)

    model = nn.DataParallel(model)
    model.to(device)

    best_model = GetBestModel(opt.model_path)
    # best_epoch = 'last'
    # best_model = os.path.join('epoch_'+str(best_epoch), 'epoch_'+str(best_epoch)+'.pth')
    best_model = os.path.join(opt.model_path, best_model)
    print("loading model : ", best_model)
    model.load_state_dict(torch.load(best_model)['model_state_dict'])

    transforms_ = [
        transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ]

    transforms_ = transforms.Compose(transforms_)

    csv_file = '/mnt/CVUSA/dataset/splits/val-19zl.csv'

    img_pairs = []
    csv_file = open(csv_file)
    for l in csv_file.readlines():
        data = l.strip().split(",")
        data.pop(2)
        if opt.no_polar == False:
            data[0] = data[0].replace("bingmap", "polarmap")
            data[0] = data[0].replace("jpg", "png")
        img_pairs.append(data)

    csv_file.close()

    img_pairs = img_pairs[0:12]

    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(img_pairs)):
            sat = batch[0]
            sat = transforms_(Image.open(os.path.join('/mnt/CVUSA/dataset/', sat))).unsqueeze(0)
            
            grd = batch[1]
            grd = transforms_(Image.open(os.path.join('/mnt/CVUSA/dataset/', grd))).unsqueeze(0)


            sat_global, grd_global, sat_discriptors, grd_discriptors, sat_raw, grd_raw = model(sat, grd, is_cf=False)

            # sat_raw_mean = torch.mean(sat_raw, dim=1)
            # grd_raw_mean = torch.mean(grd_raw, dim=1)

            # sat_raw_max, _ = torch.max(sat_raw, dim=1)
            # grd_raw_max, _ = torch.max(grd_raw, dim=1)

            # sat_raw_mean = sat_raw_mean.view(1, 8, 42).unsqueeze(0) / sat_raw_mean.max()
            # grd_raw_mean = grd_raw_mean.view(1, 8, 42).unsqueeze(0) / grd_raw_mean.max()
            # print(sat_raw.shape)

            # sat_raw_max = sat_raw_max.view(1, 8, 42).unsqueeze(0) / sat_raw_max.max()
            # grd_raw_max = grd_raw_max.view(1, 8, 42).unsqueeze(0) / grd_raw_max.max()

            # sat_raw_mean = F.interpolate(sat_raw_mean, size = (64, 336))
            # grd_raw_mean = F.interpolate(grd_raw_mean, size = (64, 336))

            # sat_raw_max = F.interpolate(sat_raw_max, size = (64, 336))
            # grd_raw_max = F.interpolate(grd_raw_max, size = (64, 336))


            sat_discriptors = sat_discriptors.view(1, 8, 42, 8)
            grd_discriptors = grd_discriptors.view(1, 8, 42, 8)

            sat_discriptors = sat_discriptors.permute(3,0,1,2)
            grd_discriptors = grd_discriptors.permute(3,0,1,2)

            sat_discriptors = (sat_discriptors + 1.0) / 2.0
            grd_discriptors = (grd_discriptors + 1.0) / 2.0

            # discriptor_diff = sat_discriptors - grd_discriptors

            sat_discriptors = F.interpolate(sat_discriptors, size = (64, 336))
            grd_discriptors = F.interpolate(grd_discriptors, size = (64, 336))
            # discriptor_diff = F.interpolate(discriptor_diff, size = (64, 336))


            # save_image(sat_discriptors, f'/mnt/visualize/{i}_sat.png', nrow = 1)
            # save_image(grd_discriptors, f'/mnt/visualize/{i}_grd.png', nrow = 1)
            # save_image(discriptor_diff, f'/mnt/visualize/{i}_diff.png', nrow = 1)
            # save_image(sat_raw_mean, f'/mnt/visualize/{i}_raw_sat_mean.png', nrow = 1)
            # save_image(grd_raw_mean, f'/mnt/visualize/{i}_raw_grd_mean.png', nrow = 1)
            # save_image(sat_raw_max, f'/mnt/visualize/{i}_raw_sat_max.png', nrow = 1)
            # save_image(grd_raw_max, f'/mnt/visualize/{i}_raw_grd_max.png', nrow = 1)

            grd_discriptors = grd_discriptors.detach().cpu().numpy()
            sat_discriptors = sat_discriptors.detach().cpu().numpy()
            discriptor_diff = np.absolute(sat_discriptors - grd_discriptors)

            # print(type(grd_discriptors))
            # print(type(sat_discriptors))

            # print(grd_discriptors.shape)
            # print(sat_discriptors.shape)
            fig, axs = plt.subplots(8, 3, sharey=True, sharex=True)
            for p in range(grd_discriptors.shape[0]):
                # axs[p,0].imshow(grd_discriptors[p,0,:,:], cmap='hot', interpolation='nearest')
                # axs[p,1].imshow(sat_discriptors[p,0,:,:], cmap='hot', interpolation='nearest')
                # axs[p,0].axis('off')
                # axs[p,1].axis('off')


                pcm = axs[p,0].pcolormesh(grd_discriptors[p,0,:,:], cmap='plasma')
                _ = axs[p,1].pcolormesh(sat_discriptors[p,0,:,:], cmap='plasma')
                _ = axs[p,2].pcolormesh(discriptor_diff[p,0,:,:], cmap='plasma')
                
                descriptor_index = p+1
                ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
                axs[p,0].set_ylabel(f'{ordinal(descriptor_index)}', rotation=45)

                if p == 7:
                    axs[p,0].set_xlabel('Ground descriptors')
                    axs[p,1].set_xlabel('Aerial descriptors')
                    axs[p,2].set_xlabel('Difference')
                axs[p,0].set_yticklabels([])
                axs[p,0].set_xticklabels([])
                axs[p,0].tick_params(direction='out', length=1, width=1)
                axs[p,1].tick_params(direction='out', length=1, width=1)
                axs[p,2].tick_params(direction='out', length=1, width=1)

            fig.colorbar(pcm, ax=axs[:, :], shrink=1.0)
            fig.savefig(f'/mnt/visualize/{i}_descriptors.png')