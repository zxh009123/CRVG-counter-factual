import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torchvision.models as models
from torch.utils.data import DataLoader
from dataset import ImageDataset
# from SMTL import softMarginTripletLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
import argparse

import torch.nn as nn
from torchvision import models
from SAFA_TR import SAFA_TR

# STREET_IMG_WIDTH = 616
# STREET_IMG_HEIGHT = 112
SATELLITE_IMG_WIDTH = 256
SATELLITE_IMG_HEIGHT = 256

STREET_IMG_WIDTH = 671
STREET_IMG_HEIGHT = 122


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet34(pretrained=True)
        layers = list(net.children())[:3]
        layers_end = list(net.children())[4:-3]
        self.layers = nn.Sequential(*layers, *layers_end)

    def forward(self, x):
        return self.layers(x)

class SA(nn.Module):
    def __init__(self, in_dim, num=8):
        super().__init__()
        hid_dim = in_dim // 2
        self.w1, self.b1 = self.init_weights_(in_dim, hid_dim, num)
        self.w2, self.b2 = self.init_weights_(hid_dim, in_dim, num)

    def init_weights_(self, din, dout, dnum):
        weight = torch.empty(din, dout, dnum)
        nn.init.normal_(weight, mean=0.0, std=0.005)
        bias = torch.empty(1, dout, dnum)
        nn.init.constant_(bias, val=0.1)
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
        return weight, bias

    def forward(self, x):
        mask, _ = x.max(1)
        mask = torch.einsum('bi, ijd -> bjd', mask, self.w1) + self.b1
        mask = torch.einsum('bjd, jid -> bid', mask, self.w2) + self.b2
        return mask

class SAFA_vgg(nn.Module):
    def __init__(self, n_heads = 1):
        super().__init__()
        # self.backbone_grd = models.vgg16(pretrained=True)
        # self.backbone_sat = models.vgg16(pretrained=True)

        # feats_list = list(self.backbone_grd.features)
        # feats_list = feats_list[:-1]
        # new_feats_list = []
        # for i in range(len(feats_list)):
        #     new_feats_list.append(feats_list[i])
        #     if isinstance(feats_list[i], nn.Conv2d) and i > 14:
        #         new_feats_list.append(nn.Dropout(p=0.2, inplace=True))
        # self.backbone_grd.features = nn.Sequential(*new_feats_list)

        # modules=list(self.backbone_grd.children())
        # modules = modules[:len(modules) - 2]
        # self.backbone_grd = nn.Sequential(*modules)

        # feats_list = list(self.backbone_sat.features)
        # feats_list = feats_list[:-1]
        # new_feats_list = []
        # for i in range(len(feats_list)):
        #     new_feats_list.append(feats_list[i])
        #     if isinstance(feats_list[i], nn.Conv2d) and i > 14:
        #         new_feats_list.append(nn.Dropout(p=0.2, inplace=True))
        # self.backbone_sat.features = nn.Sequential(*new_feats_list)

        # modules=list(self.backbone_sat.children())
        # modules = modules[:len(modules) - 2]
        # self.backbone_sat = nn.Sequential(*modules)

        # self.spatial_aware_grd = SA(in_dim=266, num=n_heads)
        # self.spatial_aware_sat = SA(in_dim=256, num=n_heads)


        self.backbone_grd = ResNet34()
        self.backbone_sat = ResNet34()

        self.spatial_aware_grd = SA(in_dim=1344, num=n_heads)
        self.spatial_aware_sat = SA(in_dim=1024, num=n_heads)

        self.tanh = nn.Tanh()

    def forward(self, sat, grd, is_cf):
        sat_x = self.backbone_sat(sat)
        grd_x = self.backbone_grd(grd)
        # print("sat_x : ",sat_x.shape)
        # print("grd_x : ",grd_x.shape)
        sat_x = sat_x.view(sat_x.shape[0], sat_x.shape[1],-1)
        grd_x = grd_x.view(grd_x.shape[0], grd_x.shape[1],-1)
        sat_sa = self.spatial_aware_sat(sat_x)
        grd_sa = self.spatial_aware_grd(grd_x)
        sat_sa = self.tanh(sat_sa)
        grd_sa = self.tanh(grd_sa)
        if is_cf:
            fake_sat_sa = torch.zeros_like(sat_sa).uniform_(-1, 1)
            fake_grd_sa = torch.zeros_like(grd_sa).uniform_(-1, 1)

            sat_global = torch.matmul(sat_x, sat_sa).view(sat_x.shape[0], -1)
            grd_global = torch.matmul(grd_x, grd_sa).view(grd_x.shape[0], -1)

            sat_global = F.normalize(sat_global, p=2, dim=1)
            grd_global = F.normalize(grd_global, p=2, dim=1)

            # print("sat_global : ",sat_global.shape)
            # print("grd_global : ",grd_global.shape)

            fake_sat_global = torch.matmul(sat_x, fake_sat_sa).view(sat_x.shape[0], -1)
            fake_grd_global = torch.matmul(grd_x, fake_grd_sa).view(grd_x.shape[0], -1)

            fake_sat_global = F.normalize(fake_sat_global, p=2, dim=1)
            fake_grd_global = F.normalize(fake_grd_global, p=2, dim=1)

            return sat_global, grd_global, fake_sat_global, fake_grd_global
        else:
            sat_global = torch.matmul(sat_x, sat_sa).view(sat_x.shape[0], -1)
            grd_global = torch.matmul(grd_x, grd_sa).view(grd_x.shape[0], -1)

            sat_global = F.normalize(sat_global, p=2, dim=1)
            grd_global = F.normalize(grd_global, p=2, dim=1)

            return sat_global, grd_global

class WarmUpGamma():
    def __init__(self, n_epochs, warm_up_epoch, gamma=0.95):
        assert ((n_epochs - warm_up_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.warm_up_epoch = warm_up_epoch
        self.gamma = gamma

    def step(self, epoch):
        if epoch <= self.warm_up_epoch:
            return float(epoch**2 / self.warm_up_epoch**2)
        else:
            return self.gamma ** (epoch - self.warm_up_epoch)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def softMarginTripletLoss(sate_vecs, pano_vecs, loss_weight=10, hard_topk_ratio=1.0):
    dists = 2 - 2 * torch.matmul(sate_vecs, pano_vecs.permute(1, 0))  # Pairwise matches within batch
    pos_dists = torch.diag(dists)
    N = len(pos_dists)
    diag_ids = np.arange(N)
    num_hard_triplets = int(hard_topk_ratio * (N * (N - 1))) if hard_topk_ratio < 1.0 else N * (N - 1)

    # Match from satellite to street pano
    triplet_dist_s2p = pos_dists.unsqueeze(1) - dists
    loss_s2p = torch.log(1 + torch.exp(loss_weight * triplet_dist_s2p))
    loss_s2p[diag_ids, diag_ids] = 0  # Ignore diagnal losses

    if hard_topk_ratio < 1.0:  # Hard negative mining
        loss_s2p = loss_s2p.view(-1)
        loss_s2p, s2p_ids = torch.topk(loss_s2p, num_hard_triplets)
    loss_s2p = loss_s2p.sum() / num_hard_triplets

    # Match from street pano to satellite
    triplet_dist_p2s = pos_dists - dists
    loss_p2s = torch.log(1 + torch.exp(loss_weight * triplet_dist_p2s))
    loss_p2s[diag_ids, diag_ids] = 0  # Ignore diagnal losses

    if hard_topk_ratio < 1.0:  # Hard negative mining
        loss_p2s = loss_p2s.view(-1)
        loss_p2s, p2s_ids = torch.topk(loss_p2s, num_hard_triplets)
    loss_p2s = loss_p2s.sum() / num_hard_triplets
    # Total loss
    loss = (loss_s2p + loss_p2s) / 2.0
    return loss

# def CFLoss(vecs, hat_vecs, loss_weight=10, hard_topk_ratio=1.0):
#     print(vecs.shape)
#     print(hat_vecs.shape)
#     dists = torch.matmul(vecs, hat_vecs.permute(1, 0))  # Pairwise matches within batch
#     pos_dists = torch.diag(dists)
#     N = len(pos_dists)
#     diag_ids = np.arange(N)
#     num_hard_triplets = int(hard_topk_ratio * (N * (N - 1))) if hard_topk_ratio < 1.0 else N * (N - 1)

#     # Match from satellite to street pano
#     triplet_dist_s2p = pos_dists.unsqueeze(1) - dists
#     loss_s2p = torch.log(1 + torch.exp(loss_weight * triplet_dist_s2p))
#     loss_s2p[diag_ids, diag_ids] = 0  # Ignore diagnal losses

#     if hard_topk_ratio < 1.0:  # Hard negative mining
#         loss_s2p = loss_s2p.view(-1)
#         loss_s2p, s2p_ids = torch.topk(loss_s2p, num_hard_triplets)
#     loss_s2p = loss_s2p.sum() / num_hard_triplets

#     # Match from street pano to satellite
#     triplet_dist_p2s = pos_dists - dists
#     loss_p2s = torch.log(1 + torch.exp(loss_weight * triplet_dist_p2s))
#     loss_p2s[diag_ids, diag_ids] = 0  # Ignore diagnal losses

#     if hard_topk_ratio < 1.0:  # Hard negative mining
#         loss_p2s = loss_p2s.view(-1)
#         loss_p2s, p2s_ids = torch.topk(loss_p2s, num_hard_triplets)
#     loss_p2s = loss_p2s.sum() / num_hard_triplets
#     # Total loss
#     loss = (loss_s2p + loss_p2s) / 2.0
#     return loss

def CFLoss(vecs, hat_vecs, loss_weight=5.0):
    # loss = F.cosine_similarity(vecs, hat_vecs)
    # loss = torch.log(1 + torch.exp(loss_weight * loss))
    # loss = loss.sum() / vecs.shape[0]

    dists = 2 * torch.matmul(vecs, hat_vecs.permute(1, 0)) - 2 
    cf_dists = torch.diag(dists)
    loss = torch.log(1.0 + torch.exp(loss_weight * cf_dists))

    loss = loss.sum() / vecs.shape[0]

    return loss

def save_model(savePath, model, epoch):
    modelFolder = os.path.join(savePath, f"epoch_{epoch}")
    os.makedirs(modelFolder)
    torch.save(model.state_dict(), os.path.join(modelFolder, f'trans_{epoch}.pth'))

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
    parser.add_argument("--epochs", type=int, default=150, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--save_suffix", type=str, default='test_all', help='name of the model at the end')
    parser.add_argument("--data_dir", type=str, default='../scratch/CVUSA/dataset/', help='dir to the dataset')
    parser.add_argument("--model", type=str, help='model')
    parser.add_argument("--SAFA_heads", type=int, default=8, help='number of SAFA heads')
    parser.add_argument("--gamma", type=float, default=10.0, help='value for gamma')
    parser.add_argument('--cf', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument("--alpha", type=float, default=1.0, help='weight for final loss')
    opt = parser.parse_args()
    print(opt)

    batch_size = opt.batch_size
    number_of_epoch = opt.epochs
    learning_rate = opt.lr
    number_SAFA_heads = opt.SAFA_heads
    gamma = 10.0
    is_cf = opt.cf
    dataset_name = "CVUSA"

    save_name = f"{opt.model}_{opt.lr}_{is_cf}_{opt.alpha}_{number_SAFA_heads}_{batch_size}_{opt.save_suffix}"
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    else:
        print("Note! Saving path existed !")
    writer = SummaryWriter(save_name)

    val_transforms_sate = [transforms.Resize((SATELLITE_IMG_WIDTH, SATELLITE_IMG_HEIGHT)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                    ]
    val_transforms_street = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                    ]

    train_transforms_sate = [transforms.Resize((SATELLITE_IMG_WIDTH + 14, SATELLITE_IMG_HEIGHT + 14)),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.RandomResizedCrop([SATELLITE_IMG_WIDTH, SATELLITE_IMG_HEIGHT], scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                    transforms.ToTensor(),
                    transforms.RandomErasing(),
                    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                    ]
    train_transforms_street = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    transforms.RandomErasing(),
                    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(ImageDataset(data_dir = opt.data_dir, transforms_street=train_transforms_street,transforms_sat=train_transforms_sate, mode="train", zooms=[20]),\
         batch_size=batch_size, shuffle=True, num_workers=8)

    validateloader = DataLoader(ImageDataset(data_dir = opt.data_dir, transforms_street=val_transforms_street,transforms_sat=val_transforms_sate, mode="val", zooms=[20]),\
         batch_size=batch_size, shuffle=True, num_workers=8)

    if opt.model == "SAFA_vgg":
        model = SAFA_vgg(n_heads = number_SAFA_heads)
    elif opt.model == "SAFA_TR":
        model = SAFA_TR(n_heads = number_SAFA_heads)
    else:
        raise RuntimeError(f"model {opt.model} is not implemented")
    model = nn.DataParallel(model)
    model.to(device)

    if opt.model == "SAFA_vgg":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
        lrSchedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    elif opt.model == "SAFA_TR":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.03, eps=1e-6)
        lrSchedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=WarmUpGamma(number_of_epoch, 5, 0.96).step)
    else:
        raise RuntimeError("configs not implemented")
    # lrSchedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(number_of_epoch, 0, 0).step)
    # lrSchedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    print("start training...")

    best_epoch = {'acc':0, 'epoch':0}
    for epoch in range(number_of_epoch):
        if is_cf:
            epoch_triplet_loss = 0
            epoch_cf_loss = 0
        else:
            epoch_loss = 0
        model.train()
        for batch in tqdm(dataloader, disable=opt.verbose):
            sat = batch['satellite'].to(device)
            # sat = sat.reshape(sat.shape[0], sat.shape[2], sat.shape[3], sat.shape[4])
            grd = batch['ground'].to(device)
            # grd = batch['street'][:,batch['street'].shape[1] // 2]
            grd = grd.to(device)

            if is_cf:
                sat_global, grd_global, fake_sat_global, fake_grd_global = model(sat, grd, is_cf)
            else:
                sat_global, grd_global = model(sat, grd, is_cf)

            triplet_loss = softMarginTripletLoss(sat_global, grd_global, gamma)

            if is_cf:
                CFLoss_sat= CFLoss(sat_global, fake_sat_global)
                CFLoss_grd = CFLoss(grd_global, fake_grd_global)
                CFLoss_total = opt.alpha * (CFLoss_sat + CFLoss_grd) / 2.0
                loss = triplet_loss + CFLoss_total
                # print(loss)
                # print(CFLoss_sat)
                # print(CFLoss_grd)
                # print("============")
                epoch_triplet_loss += triplet_loss.item()
                epoch_cf_loss += CFLoss_total.item()
            else:
                loss = triplet_loss
                epoch_loss += loss.item()
                # print(loss)
                # print("============")

            if opt.model == "SAFA_TR":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lrSchedule.step()
        print(f"==============Summary of epoch {epoch} ====================")
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


        valSateFeatures = None
        valStreetFeature = None

        model.eval()
        with torch.no_grad():
            for batch in tqdm(validateloader, disable=opt.verbose):
                sat = batch['satellite'].to(device)
                # sat = sat.reshape(sat.shape[0], sat.shape[2], sat.shape[3], sat.shape[4])
                grd = batch['ground'].to(device)
                # grd = batch['street'][:,batch['street'].shape[1] // 2]
                grd = grd.to(device)

                sat_global, grd_global = model(sat, grd, is_cf=False)
                sat_global = F.normalize(sat_global)
                grd_global = F.normalize(grd_global)

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

    # print(f'best acc: {best_epoch['acc']} at epoch {best_epoch['epoch']}')
    print("best acc : ", best_epoch['acc'])
    print("best epoch : ", best_epoch['epoch'])
