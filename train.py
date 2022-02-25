import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset.usa_dataset import ImageDataset
from dataset.act_dataset import TestDataset, TrainDataset
from torch.utils.tensorboard import SummaryWriter


from tqdm import tqdm
import os
import numpy as np
import argparse
import logging
import calendar
import time
import json

from models.SAFA_TR import SAFA_TR
from models.BAP import SCN_ResNet
from models.SAFA_vgg import SAFA_vgg

from utils.utils import WarmUpGamma, LambdaLR, softMarginTripletLoss, CFLoss, save_model, ValidateAll

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--save_suffix", type=str, default='test_PE_dropout', help='name of the model at the end')
    parser.add_argument("--data_dir", type=str, default='../scratch/CVUSA/dataset/', help='dir to the dataset')
    parser.add_argument("--model", type=str, help='model')
    parser.add_argument("--SAFA_heads", type=int, default=16, help='number of SAFA heads')
    parser.add_argument("--TR_heads", type=int, default=8, help='number of heads in Transformer')
    parser.add_argument("--TR_layers", type=int, default=6, help='number of layers in Transformer')
    parser.add_argument("--TR_dim", type=int, default=2048, help='dim of FFD in Transformer')
    parser.add_argument("--dropout", type=float, default=0.3, help='dropout in Transformer')
    parser.add_argument("--gamma", type=float, default=10.0, help='value for gamma')
    parser.add_argument('--cf', default=False, action='store_true', help='counter factual loss')
    parser.add_argument('--verbose', default=True, action='store_false', help='turn on progress bar')
    parser.add_argument('--no_polar', default=False, action='store_true', help='turn off polar transformation')
    opt = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    batch_size = opt.batch_size
    number_of_epoch = opt.epochs
    learning_rate = opt.lr
    number_SAFA_heads = opt.SAFA_heads
    gamma = opt.gamma
    is_cf = opt.cf
    dataset_name = "CVUSA"

    hyper_parameter_dict = vars(opt)
    
    logger.info("Configuration:")
    for k, v in hyper_parameter_dict.items():
        print(f"{k} : {v}")
    
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

    # generate time stamp
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    save_name = f"{ts}_{opt.model}_{dataset_name}_{is_cf}_{polar_transformation}_{opt.save_suffix}"
    print("save_name : ", save_name)
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    else:
        logger.info("Note! Saving path existed !")

    with open(os.path.join(save_name,f"{ts}_parameter.json"), "w") as outfile:
        json.dump(hyper_parameter_dict, outfile, indent=4)

    writer = SummaryWriter(save_name)

    val_transforms_sate = [transforms.Resize((SATELLITE_IMG_WIDTH, SATELLITE_IMG_HEIGHT)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                    ]
    val_transforms_street = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                    ]

    if opt.model == "SAFA_TR": #TR model need strong augmentation
        train_transforms_sate = [transforms.Resize((SATELLITE_IMG_WIDTH, SATELLITE_IMG_HEIGHT)),
                        transforms.ColorJitter(0.2, 0.2, 0.2),
                        transforms.ToTensor(),
                        transforms.RandomErasing(),
                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                        ]
        train_transforms_street = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
                        transforms.ColorJitter(0.2, 0.2, 0.2),
                        transforms.ToTensor(),
                        transforms.RandomErasing(),
                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                        ]
    else:
        train_transforms_sate = [transforms.Resize((SATELLITE_IMG_WIDTH, SATELLITE_IMG_HEIGHT)),
                        transforms.ColorJitter(0.1, 0.1, 0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                        ]
        train_transforms_street = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
                        transforms.ColorJitter(0.1, 0.1, 0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                        ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #dataloader now only support CVUSA
    # TODO: add support to CVACT

    dataloader = DataLoader(ImageDataset(data_dir = opt.data_dir, transforms_street=train_transforms_street,transforms_sat=train_transforms_sate, mode="train", is_polar=polar_transformation),\
        batch_size=batch_size, shuffle=True, num_workers=8)

    validateloader = DataLoader(ImageDataset(data_dir = opt.data_dir, transforms_street=val_transforms_street,transforms_sat=val_transforms_sate, mode="val", is_polar=polar_transformation),\
        batch_size=batch_size, shuffle=True, num_workers=8)

    if opt.model == "SAFA_vgg":
        model = SAFA_vgg(safa_heads = number_SAFA_heads, is_polar=polar_transformation)
    elif opt.model == "SAFA_TR":
        model = SAFA_TR(safa_heads=opt.SAFA_heads, tr_heads=opt.TR_heads, tr_layers=opt.TR_layers, dropout = opt.dropout, d_hid=opt.TR_dim, is_polar=polar_transformation)
    elif opt.model == "SCN_ResNet":
        model = SCN_ResNet()
    else:
        raise RuntimeError(f"model {opt.model} is not implemented")
    model = nn.DataParallel(model)
    model.to(device)

    #set optimizer and lr scheduler
    if opt.model == "SAFA_vgg":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
        lrSchedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    elif opt.model == "SCN_ResNet":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
        lrSchedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(number_of_epoch, 0, 30).step)
    elif opt.model == "SAFA_TR":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.03, eps=1e-6)
        lrSchedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=WarmUpGamma(number_of_epoch, 5, 0.97).step)
    else:
        raise RuntimeError("configs not implemented")

    # Start training
    logger.info("start training...")
    best_epoch = {'acc':0, 'epoch':0}
    for epoch in range(number_of_epoch):
        if is_cf:
            epoch_triplet_loss = 0
            epoch_cf_loss = 0
        else:
            epoch_loss = 0
        model.train() # set model to train
        for batch in tqdm(dataloader, disable = opt.verbose):
            sat = batch['satellite'].to(device)
            grd = batch['ground'].to(device)

            if is_cf:
                sat_global, grd_global, fake_sat_global, fake_grd_global = model(sat, grd, is_cf)
            else:
                sat_global, grd_global = model(sat, grd, is_cf)
            # soft margin triplet loss
            triplet_loss = softMarginTripletLoss(sat_global, grd_global, gamma)

            if is_cf:# calculate CF loss
                CFLoss_sat= CFLoss(sat_global, fake_sat_global)
                CFLoss_grd = CFLoss(grd_global, fake_grd_global)
                CFLoss_total = (CFLoss_sat + CFLoss_grd) / 2.0
                loss = triplet_loss + CFLoss_total

                epoch_triplet_loss += triplet_loss.item()
                epoch_cf_loss += CFLoss_total.item()
            else:
                loss = triplet_loss
                epoch_loss += loss.item()


            optimizer.zero_grad()
            loss.backward()
            if opt.model == "SAFA_TR":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # adjust lr
        lrSchedule.step()

        logger.info(f"Summary of epoch {epoch}")
        print(f"===============================================")
        if is_cf:
            current_triplet_loss = float(epoch_triplet_loss) / float(len(dataloader))
            current_cf_loss = float(epoch_cf_loss) / float(len(dataloader))
            print("---------loss---------")
            print(f"Epoch {epoch} CF_Loss: {current_cf_loss}")
            print(f"Epoch {epoch} TRI_Loss: {current_triplet_loss}")
            writer.add_scalar('triplet_loss', current_triplet_loss, epoch)
            writer.add_scalar('cf_loss', current_cf_loss, epoch)
            print("----------------------")
        else:
            epoch_loss = float(epoch_loss) / float(len(dataloader))
            print("---------loss---------")
            print(f"Epoch {epoch} Loss {epoch_loss}")
            writer.add_scalar('loss', epoch_loss, epoch)
            print("----------------------")

        # Testing phase
        valSateFeatures = None
        valStreetFeature = None

        model.eval()
        with torch.no_grad():
            for batch in tqdm(validateloader, disable = opt.verbose):
                sat = batch['satellite'].to(device)
                grd = batch['ground'].to(device)

                sat_global, grd_global = model(sat, grd, is_cf=False)

                if valSateFeatures == None:
                    valSateFeatures = sat_global.detach()
                else:
                    valSateFeatures = torch.cat((valSateFeatures, sat_global.detach()), dim=0)

                if valStreetFeature == None:
                    valStreetFeature = grd_global.detach()
                else:
                    valStreetFeature = torch.cat((valStreetFeature, grd_global.detach()), dim=0)

            valAcc = ValidateAll(valStreetFeature, valSateFeatures)
            logger.info("validation result")
            print(f"------------------------------------")
            try:
                #print recall value
                top1 = valAcc[0, 1]
                print('top1', ':', valAcc[0, 1] * 100.0)
                print('top5', ':', valAcc[0, 5] * 100.0)
                print('top10', ':', valAcc[0, 10] * 100.0)
                print('top1%', ':', valAcc[0, -1] * 100.0)
                # write to tensorboard
                writer.add_scalars('validation recall@k',{
                    'top 1':valAcc[0, 1],
                    'top 5':valAcc[0, 5],
                    'top 10':valAcc[0, 10],
                    'top 1%':valAcc[0, -1]
                }, epoch)
            except:
                print(valAcc)

            if top1 > best_epoch['acc']:
                best_epoch['acc'] = top1
                best_epoch['epoch'] = epoch
                save_model(save_name, model, epoch)
            print(f"=================================================")

    # get the best model and recall
    print("best acc : ", best_epoch['acc'])
    print("best epoch : ", best_epoch['epoch'])
