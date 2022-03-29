import torch
import numpy as np
import os
import math
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import json

def ReadConfig(path):
    all_files = os.listdir(path)
    config_file =  list(filter(lambda x: x.endswith('parameter.json'), all_files))
    with open(os.path.join(path, config_file[0]), 'r') as f:
        p = json.load(f)
        return p

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

class WarmUpGamma():
    def __init__(self, n_epochs, warm_up_epoch, gamma=0.95):
        assert ((n_epochs - warm_up_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.warm_up_epoch = warm_up_epoch
        self.gamma = gamma

    def step(self, epoch):
        if epoch <= self.warm_up_epoch:
            return float(epoch / self.warm_up_epoch)
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

def IntraLoss(sate_vecs, pano_vecs, loss_weight=10, hard_topk_ratio=1.0):
    # satllite pairwise
    dists_sat = 2 - 2 * torch.matmul(sate_vecs, sate_vecs.permute(1, 0))
    pos_dists_sat = torch.diag(dists_sat)
    N = len(pos_dists_sat)
    diag_ids = np.arange(N)
    num_hard_triplets = int(hard_topk_ratio * (N * (N - 1))) if hard_topk_ratio < 1.0 else N * (N - 1)

    triplet_dist_sat = pos_dists_sat - dists_sat
    loss_sat = torch.log(1 + torch.exp(loss_weight * triplet_dist_sat))
    loss_sat[diag_ids, diag_ids] = 0  # Ignore diagnal losses

    if hard_topk_ratio < 1.0:  # Hard negative mining
        loss_sat = loss_sat.view(-1)
        loss_sat, p2s_ids = torch.topk(loss_sat, num_hard_triplets)
    loss_sat = loss_sat.sum() / num_hard_triplets

    dists_pano = 2 - 2 * torch.matmul(pano_vecs, pano_vecs.permute(1, 0))
    pos_dists_pano = torch.diag(dists_pano)
    N = len(pos_dists_pano)
    diag_ids = np.arange(N)
    num_hard_triplets = int(hard_topk_ratio * (N * (N - 1))) if hard_topk_ratio < 1.0 else N * (N - 1)

    # pano pairwise
    triplet_dist_pano = pos_dists_pano - dists_pano
    loss_pano = torch.log(1 + torch.exp(loss_weight * triplet_dist_pano))
    loss_pano[diag_ids, diag_ids] = 0  # Ignore diagnal losses

    if hard_topk_ratio < 1.0:  # Hard negative mining
        loss_pano = loss_pano.view(-1)
        loss_pano, p2s_ids = torch.topk(loss_pano, num_hard_triplets)
    loss_pano = loss_pano.sum() / num_hard_triplets

    loss = (loss_pano + loss_sat) / 2.0
    return loss

def CFLoss(vecs, hat_vecs, loss_weight=5.0):

    dists = 2 * torch.matmul(vecs, hat_vecs.permute(1, 0)) - 2 
    cf_dists = torch.diag(dists)
    loss = torch.log(1.0 + torch.exp(loss_weight * cf_dists))

    loss = loss.sum() / vecs.shape[0]

    return loss

def save_model(savePath, model, optimizer, scheduler, epoch, last=True):
    if last == True:
        save_folder_name = "epoch_last"
        model_name = "epoch_last.pth"
    else:
        save_folder_name = f"epoch_{epoch}"
        model_name = f'epoch_{epoch}.pth'
    modelFolder = os.path.join(savePath, save_folder_name)
    if os.path.isdir(modelFolder):
        pass
    else:
        os.makedirs(modelFolder)
    # torch.save(model.state_dict(), os.path.join(modelFolder, f'trans_{epoch}.pth'))
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join(modelFolder, model_name))


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
    a = torch.rand(10, 4096)
    b = torch.rand(10, 4096)

    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)

    # print(softMarginTripletLossMX(a, b))
    print(IntraLoss(a, b, 0.4))