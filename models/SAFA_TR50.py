import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.TR import TransformerEncoder, TransformerEncoderLayer
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

class SA_PE(nn.Module):

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

class Transformer(nn.Module):

    def __init__(self, d_model=256, safa_heads = 16, nhead=8, nlayers=6, dropout = 0.3, d_hid=2048):
        super().__init__()


        self.pos_encoder = PositionalEncoding(d_model, max_len=safa_heads, dropout=dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation='gelu', batch_first=True)
        layer_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer = encoder_layers, num_layers = nlayers, norm=layer_norm)


    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class SA_TR(nn.Module):

    def __init__(self, d_model=256, safa_heads = 16, nhead=8, nlayers=6, dropout = 0.3, d_hid=2048):
        super().__init__()

        self.pos_encoder = SA_PE(d_model, max_len=safa_heads, dropout=dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation='gelu', batch_first=True)
        layer_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer = encoder_layers, num_layers = nlayers, norm=layer_norm)


    def forward(self, src, pos):
        src = self.pos_encoder(src, pos)
        output = self.transformer_encoder(src)
        return output


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet50(local_file='resnet50-0676ba61.pth')

        layers = list(net.children())[:3]
        layers_end = list(net.children())[4:-3] #torch.Size([7, 1024, 16, 84]); 16*84=1344
        self.layers = nn.Sequential(*layers, *layers_end)

    def forward(self, x):
        return self.layers(x)

class SA(nn.Module):
    def __init__(self, in_dim, safa_heads=8, tr_heads=8, tr_layers=6, dropout = 0.3, d_hid=2048, pos = 'learn_pos'):
        super().__init__()

        hid_dim = in_dim // 2
        self.w1, self.b1 = self.init_weights_(in_dim, hid_dim, safa_heads)
        self.w2, self.b2 = self.init_weights_(hid_dim, in_dim, safa_heads)
        self.pos = pos
        if pos == 'learn_pos':
            self.safa_tr = SA_TR(d_model=hid_dim, safa_heads=safa_heads, nhead=tr_heads, nlayers=tr_layers, dropout=dropout,d_hid=d_hid)
        else:
            self.safa_tr = Transformer(d_model=hid_dim, safa_heads=safa_heads, nhead=tr_heads, nlayers=tr_layers, dropout=dropout,d_hid=d_hid)

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

        if self.pos == 'learn_pos':
            mask = self.safa_tr(mask, pos_normalized)
        else:
            mask = self.safa_tr(mask)

        mask = torch.einsum('bdj, jdi -> bdi', mask, self.w2) + self.b2
        mask = mask.permute(0,2,1)

        return mask



class SAFA_TR50(nn.Module):
    def __init__(self, safa_heads=16, tr_heads=8, tr_layers=6, dropout = 0.3, d_hid=2048, is_polar=True, pos='learn_pos'):
        super().__init__()

        self.backbone_grd = ResNet50()
        self.backbone_sat = ResNet50()

        if is_polar:
            in_dim_sat = 1344
            in_dim_grd = 1344
        else:
            in_dim_sat = 256
            in_dim_grd = 336

        self.spatial_aware_grd = SA(in_dim=in_dim_grd, safa_heads=safa_heads, tr_heads=tr_heads, tr_layers=tr_layers, dropout = dropout, d_hid=d_hid, pos=pos)

        self.spatial_aware_sat = SA(in_dim=in_dim_sat, safa_heads=safa_heads, tr_heads=tr_heads, tr_layers=tr_layers, dropout = dropout, d_hid=d_hid, pos=pos)


    def forward(self, sat, grd, is_cf):
        b = sat.shape[0]

        sat_x = self.backbone_sat(sat)
        grd_x = self.backbone_grd(grd)

        sat_x = sat_x.view(b, sat_x.shape[1], -1)
        grd_x = grd_x.view(b, grd_x.shape[1], -1)
        sat_sa = self.spatial_aware_sat(sat_x)
        grd_sa = self.spatial_aware_grd(grd_x)
        sat_sa = F.hardtanh(sat_sa)
        grd_sa = F.hardtanh(grd_sa)
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
    model = SAFA_TR50(safa_heads=12, tr_heads=8, tr_layers=6, dropout = 0.3, d_hid=2048, pos = 'learn_pos', is_polar=True)
    sat = torch.randn(7, 3, 122, 671)
    # sat = torch.randn(7, 3, 256, 256)
    grd = torch.randn(7, 3, 122, 671)
    result = model(sat, grd, True)
    for i in result:
        print(i.shape)

    # model = SA_PE()
    # x = torch.rand(5, 16, 256)
    # pos = torch.rand(5, 512)
    # result = model(x, pos)

