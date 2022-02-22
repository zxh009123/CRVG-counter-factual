import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torchvision.models as models
from torch.utils.data import DataLoader
from dataset import ImageDataset
# from SMTL import softMarginTripletLoss
from tqdm import tqdm
import os
import numpy as np
import argparse
from act_dataloader import TestDataset

from SAFA_TR import SAFA_TR
from BAP import SCN_ResNet
from SAFA_vgg import SAFA_vgg

# STREET_IMG_WIDTH = 616
# STREET_IMG_HEIGHT = 112
SATELLITE_IMG_WIDTH = 256
SATELLITE_IMG_HEIGHT = 256

STREET_IMG_WIDTH = 671
STREET_IMG_HEIGHT = 122



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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--data_dir", type=str, default='../scratch/CVUSA/dataset/', help='dir to the dataset')
    parser.add_argument("--SAFA_heads", type=int, default=16, help='number of SAFA heads')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument("--model", type=str, help='model')
    parser.add_argument('--model_path', type=str, help='path to model weights')
    opt = parser.parse_args()
    print(opt)

    batch_size = opt.batch_size
    number_SAFA_heads = opt.SAFA_heads

    transforms_sat = [transforms.Resize((SATELLITE_IMG_WIDTH, SATELLITE_IMG_HEIGHT)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                    ]
    transforms_street = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    validateloader = DataLoader(TestDataset(data_dir = opt.data_dir, transforms_sat=transforms_sat,transforms_grd=transforms_street), batch_size=batch_size, shuffle=True, num_workers=8)
    # validateloader = DataLoader(ImageDataset(data_dir = opt.data_dir, transforms_street=transforms_street,transforms_sat=transforms_sat, mode="val", zooms=[20]), batch_size=batch_size, shuffle=True, num_workers=8)

    if opt.model == "SAFA_vgg":
        model = SAFA_vgg(n_heads = number_SAFA_heads)
    elif opt.model == "SAFA_TR":
        model = SAFA_TR(n_heads = number_SAFA_heads)
    elif opt.model == "SCN_ResNet":
        model = SCN_ResNet()
    else:
        raise RuntimeError(f"model {opt.model} is not implemented")
    model = nn.DataParallel(model)
    model.to(device)

    print("loading model")
    model.load_state_dict(torch.load(opt.model_path))

    print("start testing...")

    valSateFeatures = None
    valStreetFeature = None

    model.eval()
    with torch.no_grad():
        for batch in tqdm(validateloader, disable=opt.verbose):
            sat = batch['satellite'].to(device)
            # sat = sat.reshape(sat.shape[0], sat.shape[2], sat.shape[3], sat.shape[4])
            grd = batch['ground'].to(device)
            # grd = batch['street'][:,batch['street'].shape[1] // 2]

            sat_global, grd_global = model(sat, grd, is_cf=False)
            # sat_global = F.normalize(sat_global)
            # grd_global = F.normalize(grd_global)

            #stack features to the container
            if valSateFeatures == None:
                valSateFeatures = sat_global.detach()
            else:
                valSateFeatures = torch.cat((valSateFeatures, sat_global.detach()), dim=0)

            if valStreetFeature == None:
                valStreetFeature = grd_global.detach()
            else:
                valStreetFeature = torch.cat((valStreetFeature, grd_global.detach()), dim=0)

        valAcc = ValidateAll(valStreetFeature, valSateFeatures)
        print(f"-----------validation result---------------")
        try:
            #print epoch loss
            top1 = valAcc[0, 1]
            print('top1', ':', valAcc[0, 1] * 100.0)
            print('top5', ':', valAcc[0, 5] * 100.0)
            print('top10', ':', valAcc[0, 10] * 100.0)
            print('top1%', ':', valAcc[0, -1] * 100.0)
        except:
            print(valAcc)
        print(f"=================================================")
