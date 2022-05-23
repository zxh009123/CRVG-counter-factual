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

def sample_within_bounds(signal, x, y, bounds):
    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    sample = np.zeros((x.shape[0], x.shape[1], signal.shape[-1]))

    sample[idxs, :] = signal[x[idxs], y[idxs], :]

    return sample


def sample_bilinear(signal, rx, ry):

    signal_dim_x = signal.shape[0]
    signal_dim_y = signal.shape[1]

    # obtain four sample coordinates
    ix0 = rx.astype(int)
    iy0 = ry.astype(int)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    na = np.newaxis
    # linear interpolation in x-direction
    fx1 = (ix1-rx)[...,na] * signal_00 + (rx-ix0)[...,na] * signal_10
    fx2 = (ix1-rx)[...,na] * signal_01 + (rx-ix0)[...,na] * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry)[...,na] * fx1 + (ry - iy0)[...,na] * fx2



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
    # best_epoch = '1'
    # best_model = os.path.join(opt.model_path, 'epoch_'+str(best_epoch), 'epoch_'+str(best_epoch)+'.pth')
    best_model = os.path.join(opt.model_path, best_model)
    print("loading model : ", best_model)
    model.load_state_dict(torch.load(best_model)['model_state_dict'])
    # for name, w in model.named_parameters():
    #     if "spatial" in name:
    #         print(name)
    #         print(w)
    # exit(0)
    # print("not load")
    transforms_street = [
        transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
        # transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ]

    transforms_sat = [
        transforms.Resize((SATELLITE_IMG_HEIGHT, SATELLITE_IMG_WIDTH)),
        # transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ]
    
    transforms_sat = transforms.Compose(transforms_sat)
    transforms_street = transforms.Compose(transforms_street)

    csv_file = '/mnt/CVUSA/dataset/splits/train-19zl.csv'

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
            sat = transforms_sat(Image.open(os.path.join('/mnt/CVUSA/dataset/', sat))).unsqueeze(0)
            # sat = (torch.randn_like(sat) / 0.5 + 0.5)
            # sat = torch.zeros_like(sat)
            
            grd = batch[1]
            grd = transforms_street(Image.open(os.path.join('/mnt/CVUSA/dataset/', grd))).unsqueeze(0)
            # grd = (torch.randn_like(grd) / 0.5 + 0.5)
            # grd = torch.zeros_like(grd)

            # random_in = torch.randn(1, 512, 336).cuda() + 1.0
            # m = model.module.cuda()
            # out = m.forward_TR(random_in)
            # print(out[0,1])
            # continue

            # sa = model(sat, grd, is_cf=False)
            sat_global, grd_global, sat_discriptors, grd_discriptors, sat_raw, grd_raw = model(sat, grd, is_cf=False)
            # print(sa[0,1])
            # continue

            if opt.no_polar:
                sat_discriptors = sat_discriptors.view(1, 16, 16, 8)
            else:
                sat_discriptors = sat_discriptors.view(1, 8, 42, 8)
            grd_discriptors = grd_discriptors.view(1, 8, 42, 8)

            sat_discriptors = sat_discriptors.permute(3,0,1,2)
            grd_discriptors = grd_discriptors.permute(3,0,1,2)

            sat_discriptors = (sat_discriptors + 1.0) / 2.0
            grd_discriptors = (grd_discriptors + 1.0) / 2.0

            # discriptor_diff = sat_discriptors - grd_discriptors
            if opt.no_polar:
                sat_discriptors = F.interpolate(sat_discriptors, size = (64, 64), mode="nearest")
            else:
                sat_discriptors = F.interpolate(sat_discriptors, size = (64, 336), mode="nearest")
            grd_discriptors = F.interpolate(grd_discriptors, size = (64, 336))
            # discriptor_diff = F.interpolate(discriptor_diff, size = (64, 336))

            

            grd_discriptors = grd_discriptors.detach().cpu().numpy()
            sat_discriptors = sat_discriptors.detach().cpu().numpy()

            if opt.no_polar == False:
                discriptor_diff = np.absolute(sat_discriptors - grd_discriptors)
                third_col = discriptor_diff
            else:
                # polar trans for non polar one 
                S = 64  # Original size of the aerial image
                height = 64  # Height of polar transformed aerial image
                width = 336   # Width of polar transformed aerial image

                n = np.arange(0, height)
                j = np.arange(0, width)
                jj, ii = np.meshgrid(j, n)

                y = S/2. - S/2./height*(height-1-ii)*np.sin(2*np.pi*jj/width)
                x = S/2. + S/2./height*(height-1-ii)*np.cos(2*np.pi*jj/width)

                sat_discriptors_PL = np.zeros((8, 1, height, width))
                for sd in range(sat_discriptors.shape[0]):
                    dis = sat_discriptors[sd,:,:,:]
                    dis = np.transpose(dis, (1, 2, 0))
                    sat_discriptors_PL[sd,:,:,:] = np.transpose(sample_bilinear(dis, x, y), (2,0,1))

                third_col = sat_discriptors_PL



            # print(sat_discriptors.shape)
            # print(type(grd_discriptors))
            # print(type(sat_discriptors))

            # print(grd_discriptors.shape)
            # print(sat_discriptors.shape)

            if opt.no_polar == False:
                fig, axs = plt.subplots(8, 3)
            else:
                fig, axs = plt.subplots(8, 4, gridspec_kw={'width_ratios': [3.5,3.5, 1, 1]})
                empty = np.ones((8, 1, S, S)) * 255.0
                # empty = np.random.rand(8, 1, S, S)
            
            for p in range(grd_discriptors.shape[0]):
                gd = grd_discriptors[p,0,::-1,:]
                sd = sat_discriptors[p,0,::-1,:]
                tc = third_col[p,0,::-1,:]

                # gd = gd / np.amax(gd)
                # sd = sd / np.amax(gd)
                # tc = tc / np.amax(gd)
                if opt.no_polar == False:
                    pcm = axs[p,0].pcolormesh(gd, cmap='plasma')
                    _ = axs[p,1].pcolormesh(sd, cmap='plasma')
                    _ = axs[p,2].pcolormesh(tc, cmap='plasma')
                else:
                    pcm = axs[p,0].pcolormesh(gd, cmap='plasma')
                    _ = axs[p,1].pcolormesh(tc, cmap='plasma')
                    _ = axs[p,2].imshow(empty[p,0,:,:])
                    _ = axs[p,3].pcolormesh(sd, cmap='plasma')
                
                descriptor_index = p+1
                ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
                axs[p,0].set_ylabel(f'{ordinal(descriptor_index)}', rotation=45)

                if p == 7:
                    axs[p,0].set_xlabel('Ground descriptors')
                    if opt.no_polar == False:
                        axs[p,1].set_xlabel('Aerial descriptors')
                        axs[p,2].set_xlabel('Difference')
                    else:
                        axs[p,3].set_xlabel('Aerial descriptors')
                        axs[p,1].set_xlabel('Polar transformed \naerial descriptors')

                for m in range(len(axs[p])):
                    axs[p,m].set_yticklabels([])
                    axs[p,m].set_xticklabels([])
                    axs[p,m].tick_params(direction='out', length=0, width=0)
                    axs[p,m].spines['right'].set_visible(False)
                    axs[p,m].spines['left'].set_visible(False)
                    axs[p,m].spines['top'].set_visible(False)
                    axs[p,m].spines['bottom'].set_visible(False)
                
 
            fig.colorbar(pcm, ax=axs[:,:], shrink=1.0, aspect=50)
            fig.savefig(f'/mnt/visualize/{i}_descriptors.png', dpi=300)
            sat_save = sat * 0.5 + 0.5
            save_image(sat_save, f'/mnt/visualize/{i}_sat_input.png')
            grd_save = grd * 0.5 + 0.5
            save_image(grd_save, f'/mnt/visualize/{i}_grd_input.png')
            plt.clf()