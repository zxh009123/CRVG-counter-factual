import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import random
from thop import profile
from thop import clever_format
if os.environ["USER"] == "xyli1905":
    from models.TR import TransformerEncoder, TransformerEncoderLayer
else:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class LearnablePE(nn.Module):

    def __init__(self, d_model, dropout = 0.3, max_len = 8):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model = 256, max_len = 16, dropout=0.3):
        super().__init__()
        self.dr = torch.nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)#batch first
        # print(position.shape)
        # print(div_term.shape)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape)
        # print(self.pe[:x.size(0)].shape)
        x = x + self.pe[:x.size(0)]
        return self.dr(x)

# Index-awared (learnable) PE
class IAPE(nn.Module):

    def __init__(self, d_model = 256, max_len = 16, dropout=0.3):
        super().__init__()
        self.dr = torch.nn.Dropout(p=dropout)

        self.linear = torch.empty(d_model*2, max_len, d_model)
        nn.init.normal_(self.linear, mean=0.0, std=0.005)
        self.linear = torch.nn.Parameter(self.linear)
        self.embedding_parameter = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x, pos):
        em_pos = torch.einsum('bi, idj -> bdj', pos, self.linear)
        em_pos = em_pos + self.embedding_parameter
        em_pos = F.hardtanh(em_pos)
        x = x + em_pos
        return self.dr(x)

class TRModule(nn.Module):

    def __init__(self, d_model=256, descriptors = 16, nhead=8, nlayers=6, dropout = 0.3, d_hid=2048):
        super().__init__()

        self.pos_encoder = IAPE(d_model, max_len = descriptors, dropout=dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation='gelu', batch_first=True)
        layer_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer = encoder_layers, num_layers = nlayers, norm=layer_norm)


    def forward(self, src, pos):
        src = self.pos_encoder(src, pos)
        output = self.transformer_encoder(src)
        return output

class EfficientNetB3(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.efficientnet_b3(weights='DEFAULT')
        layers = list(net.children())[:-2]
        layers = list(layers[0].children())[:-3]

        end_conv = [torch.nn.Conv2d(136, 128, 1), torch.nn.BatchNorm2d(128), torch.nn.SiLU(inplace=True)]

        self.layers = torch.nn.Sequential(*layers, *end_conv)

    def forward(self, x):
        return self.layers(x)


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet34(pretrained=True)
        layers = list(net.children())[:3]
        layers_end = list(net.children())[4:-2]

        # 4096
        # self.layers = nn.Sequential(*layers, *layers_end)

        # 1024
        end_conv = [torch.nn.Conv2d(512, 128, 1), torch.nn.BatchNorm2d(128)]
        self.layers = torch.nn.Sequential(*layers, *layers_end, *end_conv)

    def forward(self, x):
        return self.layers(x)

class GeoLayoutExtractor(nn.Module):
    def __init__(self, in_dim, descriptors=8, tr_heads=8, tr_layers=6, dropout = 0.3, d_hid=2048):
        super().__init__()

        self.tr_layers = tr_layers

        hid_dim = in_dim // 2
        self.w1, self.b1 = self.init_weights_(in_dim, hid_dim, descriptors)
        self.w2, self.b2 = self.init_weights_(hid_dim, in_dim, descriptors)
        if self.tr_layers != 0:
            self.tr_module = TRModule(d_model=hid_dim, descriptors=descriptors, nhead=tr_heads, nlayers=tr_layers, dropout=dropout,d_hid=d_hid)

    def init_weights_(self, din, dout, dnum):
        # weight = torch.empty(din, dout, dnum)
        weight = torch.empty(din, dnum, dout)
        nn.init.normal_(weight, mean=0.0, std=0.005)
        # bias = torch.empty(1, dout, dnum)
        bias = torch.empty(1, dnum, dout)
        nn.init.constant_(bias, val=0.1)
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
        return weight, bias

    def forward(self, x):
        channel = x.shape[1]
        mask, pos = x.max(1)

        pos_normalized = pos / channel

        mask = torch.einsum('bi, idj -> bdj', mask, self.w1) + self.b1

        if self.tr_layers != 0:
            mask = self.tr_module(mask, pos_normalized)

        mask = torch.einsum('bdj, jdi -> bdi', mask, self.w2) + self.b2
        mask = mask.permute(0,2,1)

        return mask



class GeoDTR(nn.Module):
    def __init__(self, descriptors = 16, tr_heads=8, tr_layers=6, dropout = 0.3, d_hid=2048, is_polar=True, ARCH="efficientnet"):
        super().__init__()

        if ARCH == "resnet":
            self.backbone_grd = ResNet34()
            self.backbone_sat = ResNet34()
        else:
            self.backbone_grd = EfficientNetB3()
            self.backbone_sat = EfficientNetB3()

        if is_polar:
            in_dim_sat = 336
            in_dim_grd = 336
        else:
            in_dim_sat = 256
            in_dim_grd = 336

        self.GLE_grd = GeoLayoutExtractor(in_dim=in_dim_grd, descriptors=descriptors, tr_heads=tr_heads, tr_layers=tr_layers, dropout = dropout, d_hid=d_hid)

        self.GLE_sat = GeoLayoutExtractor(in_dim=in_dim_sat, descriptors=descriptors, tr_heads=tr_heads, tr_layers=tr_layers, dropout = dropout, d_hid=d_hid)


    def forward(self, sat, grd, is_cf):
        b = sat.shape[0]

        sat_x = self.backbone_sat(sat)
        grd_x = self.backbone_grd(grd)

        sat_x = sat_x.view(b, sat_x.shape[1], -1)
        grd_x = grd_x.view(b, grd_x.shape[1], -1)
        sat_sa = self.GLE_sat(sat_x)
        grd_sa = self.GLE_grd(grd_x)
        sat_sa = F.hardtanh(sat_sa)
        grd_sa = F.hardtanh(grd_sa)

        # print("sat_sa shape : ", sat_x.shape)
        # print("grd_sa shape : ", grd_x.shape)

        if is_cf:
            fake_sat_sa = torch.zeros_like(sat_sa).uniform_(-1.0, 1.0)
            fake_grd_sa = torch.zeros_like(grd_sa).uniform_(-1.0, 1.0)


            sat_global = torch.matmul(sat_x, sat_sa).view(b,-1)
            grd_global = torch.matmul(grd_x, grd_sa).view(b,-1)

            sat_global = F.normalize(sat_global, p=2, dim=1)
            grd_global = F.normalize(grd_global, p=2, dim=1)

            fake_sat_global = torch.matmul(sat_x, fake_sat_sa).view(b,-1)
            fake_grd_global = torch.matmul(grd_x, fake_grd_sa).view(b,-1)

            fake_sat_global = F.normalize(fake_sat_global, p=2, dim=1)
            fake_grd_global = F.normalize(fake_grd_global, p=2, dim=1)

            return sat_global, grd_global, fake_sat_global, fake_grd_global, sat_sa, grd_sa

        else:
            sat_global = torch.matmul(sat_x, sat_sa).view(b,-1)
            grd_global = torch.matmul(grd_x, grd_sa).view(b,-1)

            sat_global = F.normalize(sat_global, p=2, dim=1)
            grd_global = F.normalize(grd_global, p=2, dim=1)

            return sat_global, grd_global, sat_sa, grd_sa

if __name__ == "__main__":
    model = GeoDTR(descriptors=8, tr_heads=4, tr_layers=4, dropout = 0.3, d_hid=512, is_polar=True)
    sat = torch.randn(7, 3, 122, 671)
    # sat = torch.randn(7, 3, 256, 256)
    grd = torch.randn(7, 3, 122, 671)
    result = model(sat, grd, True)

    for i in result:
        print(i.shape)

    macs, params = profile(model, inputs=(sat, grd, False, ))
    macs, params = clever_format([macs, params], "%.3f")

    print(macs)
    print(params)


