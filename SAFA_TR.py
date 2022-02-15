import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model = 256, max_len = 550):
        super().__init__()
        self.dr = torch.nn.Dropout(p=0.2)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dr(x)

class Transformer(nn.Module):

    def __init__(self, d_model=256, nhead=4, nlayers=2, dropout = 0.3, d_hid=1024):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model


    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


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
        self.tr = Transformer(d_model=hid_dim)

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
        mask = mask.permute(0,2,1)
        mask = self.tr(mask)
        mask = mask.permute(0,2,1)
        mask = torch.einsum('bjd, jid -> bid', mask, self.w2) + self.b2
        return mask

class SAFA_TR(nn.Module):
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

        self.n_heads = n_heads
        self.backbone_grd = ResNet34()
        self.backbone_sat = ResNet34()

        self.spatial_aware_grd = SA(in_dim=1344, num=n_heads)
        self.spatial_aware_sat = SA(in_dim=1024, num=n_heads)

        self.tanh = nn.Tanh()

    def forward(self, sat, grd, is_cf):
        b = sat.shape[0]

        sat_x = self.backbone_sat(sat)
        grd_x = self.backbone_grd(grd)
        # print("sat_x : ",sat_x.shape)
        # print("grd_x : ",grd_x.shape)
        sat_x = sat_x.view(b, sat_x.shape[1],-1)
        grd_x = grd_x.view(b, grd_x.shape[1],-1)
        sat_sa = self.spatial_aware_sat(sat_x)
        grd_sa = self.spatial_aware_grd(grd_x)
        sat_sa = self.tanh(sat_sa)
        grd_sa = self.tanh(grd_sa)
        if is_cf:
            fake_sat_sa = torch.zeros_like(sat_sa).uniform_(-1, 1)
            fake_grd_sa = torch.zeros_like(grd_sa).uniform_(-1, 1)

            sat_global = torch.matmul(sat_x, sat_sa).view(b,-1)
            grd_global = torch.matmul(grd_x, grd_sa).view(b,-1)

            sat_global = F.normalize(sat_global, p=2, dim=1)
            grd_global = F.normalize(grd_global, p=2, dim=1)

            fake_sat_global = torch.matmul(sat_x, fake_sat_sa).view(b,-1)
            fake_grd_global = torch.matmul(grd_x, fake_grd_sa).view(b,-1)

            fake_sat_global = F.normalize(fake_sat_global, p=2, dim=1)
            fake_grd_global = F.normalize(fake_grd_global, p=2, dim=1)

            return sat_global, grd_global, fake_sat_global, fake_grd_global
        else:
            sat_global = torch.matmul(sat_x, sat_sa).view(b,-1)
            grd_global = torch.matmul(grd_x, grd_sa).view(b,-1)

            sat_global = F.normalize(sat_global, p=2, dim=1)
            grd_global = F.normalize(grd_global, p=2, dim=1)

            return sat_global, grd_global

if __name__ == "__main__":
    model = SAFA_TR(n_heads = 16)
    sat = torch.randn(5, 3, 256, 256)
    grd = torch.randn(5, 3, 122, 671)
    result = model(sat, grd, True)
    for i in result:
        print(i.shape)
