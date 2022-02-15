import torch
import numpy as np
import torch.nn.functional as F

# def CFLoss(vecs, hat_vecs, loss_weight=1):
#     dists = F.cosine_similarity(vecs, hat_vecs)
#     print(dists)
    
#     loss = torch.log(1 + torch.exp(loss_weight * dists))

#     loss = loss.sum() / vecs.shape[0]

#     return loss


# a = torch.zeros((32,4096)).uniform_(0,1)
# b = torch.zeros((32,4096)).uniform_(0,1)

# a = F.normalize(a, dim=1, p=2)
# b = F.normalize(b, dim=1, p=2)
# print(CFLoss(a,b))
# print("test")

class WarmUpGamma():
    def __init__(self, n_epochs, warm_up_epoch, gamma=0.95):
        assert ((n_epochs - warm_up_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.warm_up_epoch = warm_up_epoch
        self.gamma = gamma

    def step(self, epoch):
        if epoch <= self.warm_up_epoch:
            return float(epoch**2  / self.warm_up_epoch**2)
        else:
            return self.gamma ** (epoch - self.warm_up_epoch)


model = torch.nn.Conv2d(3,64,3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
lrSchedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(100, 10).step)
for i in range(20):
    lrSchedule.step()
    print(lrSchedule.get_last_lr())