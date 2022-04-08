import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class LearnablePE(nn.Module):

    def __init__(self, d_model, dropout = 0.3, max_len = 8, CLS=True):
        super().__init__()
        #CLS token
        self.is_cls = CLS
        self.max_len = max_len
        if self.is_cls:
            cls_token = torch.zeros(1, 1, d_model)
            self.cls_token = nn.Parameter(cls_token)
            self.max_len += 1
            

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(1, self.max_len, d_model)
        self.pe = torch.nn.Parameter(pe)

    def forward(self, x):
        if self.is_cls:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

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
        x = x + self.pe[:x.size(0)]
        return self.dr(x)


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet34(pretrained=True)
        layers = list(net.children())[:3]
        layers_end = list(net.children())[4:-2]
        self.layers = nn.Sequential(*layers, *layers_end)
        # print(self.layers)
        #(256, H/8, W/8)
        #(512, H/32, W/32)

        #SAFA
        #(512, H/16, W/16)
        #L2LTR
        #(2048, H/16, W/16)

    def forward(self, x):
        return self.layers(x)

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet50(pretrained=True)

        layers = list(net.children())[:3]
        layers_end = list(net.children())[4:-2]
        self.layers = nn.Sequential(*layers, *layers_end)

        # print(self.layers)

        # layers = list(net.children())[:-2]
        # self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SA_TR_TOPK(nn.Module):

    def __init__(self, d_model=256, top_k = 16, nhead=8, nlayers=6, dropout = 0.3, d_hid=2048):
        super().__init__()
        #positional embedding
        self.pos_encoder = LearnablePE(d_model, max_len=top_k, dropout=dropout, CLS=True)
        # self.pos_encoder = LearnablePE(d_model, max_len=top_k, dropout=dropout, CLS=False)
        # Transformer
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation='gelu', batch_first=True)
        layer_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer = encoder_layers, num_layers = nlayers, norm=layer_norm)


    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class SA_TOPK(nn.Module):
    def __init__(self, in_dim, top_k=100, tr_heads=8, tr_layers=6, dropout = 0.3, is_TKPool = True, embed_dim = 768):
        super().__init__()
        self.topk = top_k
        hid_dim = in_dim // 2

        self.is_TKPool = is_TKPool

        #dim of output tensor
        n_channel = embed_dim

        self.channel_conv = torch.nn.Conv2d(512, n_channel, 1)

        if not self.is_TKPool:
            # 512 is the output channel of Res34
            # 2048 is the output channel of Res50
            self.conv_pool = torch.nn.Conv2d(n_channel, self.topk, 3, stride=1, padding=1, bias=True)
            
        self.w1, self.b1 = self.init_weights_(self.topk, in_dim, hid_dim)
        self.w2, self.b2 = self.init_weights_(self.topk, hid_dim, in_dim)


        self.safa_tr = SA_TR_TOPK(d_model=n_channel, top_k = top_k, nhead=tr_heads, nlayers=tr_layers, dropout = dropout, d_hid=2048)

    def init_weights_(self, channel, din, dout):
        weight = torch.empty(channel, din, dout)
        nn.init.normal_(weight, mean=0.0, std=0.005)
        bias = torch.empty(1, channel, dout)
        nn.init.constant_(bias, val=0.1)
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
        return weight, bias


    def forward(self, x, is_cf):
        x = self.channel_conv(x)
        batch, channel = x.shape[0], x.shape[1]

        if self.is_TKPool:
            x = x.view(batch, channel, -1)
            mask, _ = torch.topk(x, self.topk, dim=1, sorted=True)
        else:
            mask = self.conv_pool(x)
            mask = mask.view(batch, self.topk, -1)
            x = x.view(batch, channel, -1)


        mask = torch.einsum("bci, cij -> bcj", mask, self.w1)+self.b1
        mask = torch.einsum("bcj, cji -> bci", mask, self.w2)+self.b2
        

        # hardtanh for mapping the value from -1 to 1
        mask = F.hardtanh(mask)
        x = x.permute(0,2,1)

        if is_cf:
            # (B,T,i) x (B, i, C) -> (B,T,C)
            features = torch.matmul(mask, x)

            features = self.safa_tr(features)

            feature = features[:, 0]
            feature = F.normalize(feature, p=2, dim=1)

            #random generate fake masks
            # Remaining steps are similar as previous
            fake_mask = torch.zeros_like(mask).uniform_(-1, 1)
            fake_features = torch.matmul(fake_mask, x)
            fake_features = self.safa_tr(fake_features)
            fake_feature = fake_features[:, 0]
            fake_feature = F.normalize(fake_feature, p=2, dim=1)

            return feature, fake_feature
        else: # Similar to previous
            # features = torch.matmul(x, mask).reshape(batch, top_k, -1)
            features = torch.matmul(mask, x)

            features = self.safa_tr(features)

            feature = features[:, 0]
            feature = F.normalize(feature, p=2, dim=1)

            return feature

class TK_SAFF(nn.Module):
    def __init__(self, top_k=8, tr_heads=8, tr_layers=6, dropout = 0.3, is_polar=True, pos='learn_pos', TK_Pool=True, embed_dim=768):
        super().__init__()

        #res34
        self.backbone_grd = ResNet34()
        self.backbone_sat = ResNet34()
        if is_polar:
            in_dim_sat = 336
            in_dim_grd = 336
        else:
            in_dim_sat = 256
            in_dim_grd = 336

        # self.backbone_grd = ResNet50()
        # self.backbone_sat = ResNet50()
        # #res50
        # if is_polar:
        #     in_dim_sat = 336
        #     in_dim_grd = 336
        # else:
        #     in_dim_sat = 256
        #     in_dim_grd = 336


        self.spatial_aware_grd = SA_TOPK(in_dim=in_dim_grd, top_k=top_k, tr_heads=tr_heads,\
             tr_layers=tr_layers, dropout = dropout, is_TKPool = TK_Pool, embed_dim=embed_dim)

        self.spatial_aware_sat = SA_TOPK(in_dim=in_dim_sat, top_k=top_k, tr_heads=tr_heads,\
                tr_layers=tr_layers, dropout = dropout, is_TKPool = TK_Pool, embed_dim=embed_dim)

    def forward(self, sat, grd, is_cf):
        b = sat.shape[0]

        sat_x = self.backbone_sat(sat)
        grd_x = self.backbone_grd(grd)

        if is_cf:
            sat_feature, fake_sat_feature = self.spatial_aware_sat(sat_x, is_cf=True)
            grd_feature, fake_grd_feature = self.spatial_aware_grd(grd_x, is_cf=True)
            return sat_feature, grd_feature, fake_sat_feature, fake_grd_feature
        else:
            sat_feature = self.spatial_aware_sat(sat_x, is_cf=False)
            grd_feature = self.spatial_aware_grd(grd_x, is_cf=False)
            return sat_feature, grd_feature

if __name__ == "__main__":
    model = TK_SAFF(top_k=8, tr_heads=4, tr_layers=2, dropout = 0.3, pos = 'learn_pos', is_polar=True, TK_Pool=False, embed_dim=768)
    sat = torch.randn(7, 3, 122, 671)
    # sat = torch.randn(7, 3, 256, 256)
    grd = torch.randn(7, 3, 122, 671)
    result = model(sat, grd, False)
    for i in result:
        print(i.shape)

    # model = ResNet34()
    # r = model(sat)
    # print(r.shape)


