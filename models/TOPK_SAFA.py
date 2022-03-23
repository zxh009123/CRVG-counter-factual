import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class LearnablePE(nn.Module):

    def __init__(self, d_model, dropout = 0.3, max_len = 8):
        super().__init__()
        #CLS token
        cls_token = torch.zeros(1, 1, d_model)
        self.cls_token = nn.Parameter(cls_token)

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(1, max_len, d_model)
        self.pe = torch.nn.Parameter(pe)

    def forward(self, x):

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
        layers_end = list(net.children())[4:-3]
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
        self.pos_encoder = LearnablePE(d_model, max_len=top_k+1, dropout=dropout)
        # Transformer
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation='gelu', batch_first=True)
        layer_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer = encoder_layers, num_layers = nlayers, norm=layer_norm)


    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class SA_TOPK(nn.Module):
    def __init__(self, in_dim, top_k=100, tr_heads=8, tr_layers=6, dropout = 0.3, d_hid=2048):
        super().__init__()
        self.topk = top_k
        #in_dim = H*W
        hid_dim = in_dim // 2
        
        #First layer parameter initialization
        self.w1 = torch.empty(in_dim, hid_dim)
        nn.init.normal_(self.w1, mean=0.0, std=0.005)
        self.b1 = torch.empty(1, hid_dim)
        nn.init.constant_(self.b1, val=0.1)

        self.w1 = torch.nn.Parameter(self.w1)
        self.b1 = torch.nn.Parameter(self.b1)

        #Second layer parameter initialization
        self.w2 = torch.empty(hid_dim, in_dim)
        nn.init.normal_(self.w2, mean=0.0, std=0.005)
        self.b2 = torch.empty(1, in_dim)
        nn.init.constant_(self.b2, val=0.1)

        self.w2 = torch.nn.Parameter(self.w2)
        self.b2 = torch.nn.Parameter(self.b2)
        # 256 is the output channel of Res34
        # 2048 is the output channel of Res34
        self.safa_tr = SA_TR_TOPK(d_model=2048, top_k = top_k, nhead=tr_heads, nlayers=tr_layers, dropout = dropout, d_hid=d_hid)


    def forward(self, x, is_cf):
        channel = x.shape[1]
        mask, _ = torch.topk(x, self.topk, dim=1, sorted=True)
        mask = torch.einsum('bci, id -> bcd', mask, self.w1) + self.b1
       
        mask = torch.einsum('bcd, di -> bci', mask, self.w2) + self.b2

        # hardtanh for mapping the value from -1 to 1
        mask = F.hardtanh(mask)

        batch, top_k, feat_dim = mask.shape[0], mask.shape[1], mask.shape[2]

        # permute original feature to (B,H*W,C) for matmul later
        x = x.permute(0,2,1)

        if is_cf:
            # Usning matmul to filter feature reshape to (B, topk, feature)
            features = torch.matmul(mask, x).reshape(batch, top_k, -1)

            # Feed to transformer
            features = self.safa_tr(features)
            # Get CLS token feaure
            feature = features[:, 0]
            # L2 normalize 
            feature = F.normalize(feature, p=2, dim=1)

            #random generate fake masks
            # Remaining steps are similar as previous
            fake_mask = torch.zeros_like(mask).uniform_(-1, 1)
            fake_features = torch.matmul(fake_mask, x).reshape(batch, top_k, -1)

            fake_features = self.safa_tr(fake_features)
            fake_feature = fake_features[:, 0]
            fake_feature = F.normalize(fake_feature, p=2, dim=1)
            return feature, fake_feature
        else: # Similar to previous
            features = torch.matmul(mask, x).reshape(batch, top_k, -1)

            features = self.safa_tr(features)
            feature = features[:, 0]
            feature = F.normalize(feature, p=2, dim=1)

            return feature

class TK_SAFA(nn.Module):
    def __init__(self, top_k=10, tr_heads=8, tr_layers=6, dropout = 0.3, d_hid=2048, is_polar=True, pos='learn_pos'):
        super().__init__()

        #res34
        # self.backbone_grd = ResNet34()
        # self.backbone_sat = ResNet34()
        # if is_polar:
        #     in_dim_sat = 1344
        #     in_dim_grd = 1344
        # else:
        #     in_dim_sat = 1024
        #     in_dim_grd = 1344

        self.backbone_grd = ResNet50()
        self.backbone_sat = ResNet50()
        #res50
        if is_polar:
            in_dim_sat = 336
            in_dim_grd = 336
        else:
            in_dim_sat = 256
            in_dim_grd = 336

        self.spatial_aware_grd = SA_TOPK(in_dim=in_dim_grd, top_k=top_k, tr_heads=tr_heads, tr_layers=tr_layers, dropout = dropout, d_hid=d_hid)
        if is_polar:
            self.spatial_aware_sat = SA_TOPK(in_dim=in_dim_sat, top_k=top_k, tr_heads=tr_heads, tr_layers=tr_layers, dropout = dropout, d_hid=d_hid)
        else:
            self.spatial_aware_sat = SA_TOPK(in_dim=in_dim_sat, top_k=top_k, tr_heads=tr_heads, tr_layers=tr_layers, dropout = dropout, d_hid=d_hid)

    def forward(self, sat, grd, is_cf):
        b = sat.shape[0]

        sat_x = self.backbone_sat(sat)
        grd_x = self.backbone_grd(grd)

        sat_x = sat_x.view(b, sat_x.shape[1], -1)
        grd_x = grd_x.view(b, grd_x.shape[1], -1)

        if is_cf:
            sat_feature, fake_sat_feature = self.spatial_aware_sat(sat_x, is_cf=True)
            grd_feature, fake_grd_feature = self.spatial_aware_grd(grd_x, is_cf=True)
            return sat_feature, grd_feature, fake_sat_feature, fake_grd_feature
        else:
            sat_feature = self.spatial_aware_sat(sat_x, is_cf=False)
            grd_feature = self.spatial_aware_grd(grd_x, is_cf=False)
            return sat_feature, grd_feature

if __name__ == "__main__":
    # model = TK_SAFA(top_k=100, tr_heads=4, tr_layers=2, dropout = 0.3, d_hid=2048, pos = 'learn_pos', is_polar=True)
    # sat = torch.randn(7, 3, 122, 671)
    sat = torch.randn(7, 3, 256, 256)
    # grd = torch.randn(7, 3, 122, 671)
    # result = model(sat, grd, True)
    # for i in result:
    #     print(i.shape)

    model = ResNet34()
    r = model(sat)
    print(r.shape)


