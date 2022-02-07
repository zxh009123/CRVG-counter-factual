import torch
import numpy as np
import torch.nn.functional as F

def CFLoss(vecs, hat_vecs, loss_weight=1):
    dists = F.cosine_similarity(vecs, hat_vecs)
    print(dists)
    
    loss = torch.log(1 + torch.exp(loss_weight * dists))

    loss = loss.sum() / vecs.shape[0]

    return loss


a = torch.zeros((32,4096)).uniform_(0,1)
b = torch.zeros((32,4096)).uniform_(0,1)

a = F.normalize(a, dim=1, p=2)
b = F.normalize(b, dim=1, p=2)
print(CFLoss(a,b))