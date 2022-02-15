import torch
from torch import nn
import torch.nn.functional as F
import sys

from resnet import ResNet
sys.path.append('.')


EPSILON = 1e-12


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return  y


class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions, counterfactual=False):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if counterfactual:
            if self.training:
                fake_att = torch.zeros_like(attentions).uniform_(0, 2)
            else:
                fake_att = torch.ones_like(attentions)
            # mean_feature = features.mean(3).mean(2).view(B, 1, C)
            # counterfactual_feature = mean_feature.expand(B, M, C).contiguous().view(B, -1)
            counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

            counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(torch.abs(counterfactual_feature) + EPSILON)

            counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
            return feature_matrix, counterfactual_feature
        else:
            return feature_matrix

class MultiHeadAtt(nn.Module):
    """
    Extend the channel attention into MultiHeadAtt. 
    It is modified from "Zhang H, Wu C, Zhang Z, et al. Resnest: Split-attention networks." 
    """
    def __init__(self, in_channels, channels,
                 radix=4, reduction_factor=4,
                 rectify=False, norm_layer=nn.BatchNorm2d):
        super(MultiHeadAtt, self).__init__()

        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.channels = channels
        
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=1)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels*radix, 1, groups=1)


    def forward(self, x):
        batch, channel = x.shape[:2]
        splited = torch.split(x, channel//self.radix, dim=1)
        gap = sum(splited)
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        atten = torch.split(atten, channel//self.radix, dim=1)

        out= torch.cat([att*split for (att, split) in zip(atten, splited)],1)
        return out.contiguous()


class BN2d(nn.Module):
    def __init__(self, planes):
        super(BN2d, self).__init__()
        self.bottleneck2 = nn.BatchNorm2d(planes)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)

    def forward(self, x):
        return self.bottleneck2(x)




class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, last_stride=1, using_cal=True):
        super(Baseline, self).__init__()
        self.using_cal = using_cal
        self.base = ResNet(last_stride)
        # self.base.load_param(model_path)
        self.radix = 2
        self.base_1 = nn.Sequential(*list(self.base.children())[0:3])
        self.BN1 = BN2d(64)
        self.att1 = SELayer(64,8)
        self.att_s1=MultiHeadAtt(64,int(64/self.radix),radix=self.radix)
        self.base_2 = nn.Sequential(*list(self.base.children())[3:4])
        self.BN2 = BN2d(256)
        self.att2 = SELayer(256,32)
        self.att_s2=MultiHeadAtt(256,int(256/self.radix),radix=self.radix)
        self.base_3 = nn.Sequential(*list(self.base.children())[4:5])
        self.BN3 = BN2d(512)
        self.att3 = SELayer(512,64)
        self.att_s3 = MultiHeadAtt(512,int(512/self.radix),radix=self.radix)
        self.base_4 = nn.Sequential(*list(self.base.children())[5:6])
        self.BN4 = BN2d(1024)
        self.att4 = SELayer(1024,128)
        self.att_s4=MultiHeadAtt(1024,int(1024/self.radix),radix=self.radix)
        self.base_5 = nn.Sequential(*list(self.base.children())[6:])
        self.BN5 = BN2d(2048)
        self.att5 = SELayer(2048,256)
        self.att_s5=MultiHeadAtt(2048,int(2048/self.radix),radix=self.radix)

        self.M = 8

        self.attentions = BasicConv2d(2048, self.M, kernel_size=1)
        self.bap = BAP(pool='GAP')

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)


        self.classifier_bap = nn.Linear(self.in_planes*self.M, self.in_planes, bias=False)

        self.classifier_bap.apply(weights_init_classifier)

        
    def forward(self, x):

        ############
        # x_1 = self.base_1(x)
        # x_1 = self.att_s1(x_1)
        # x_1 = self.BN1(x_1)
        # y_1 = self.att1(x_1)
        # x_att1=x_1*y_1.expand_as(x_1)


        # x_2 = self.base_2(x_att1)
        # x_2 = self.att_s2(x_2)
        # x_2 = self.BN2(x_2)
        # y_2 = self.att2(x_2)
        # x_att2=x_2*y_2.expand_as(x_2)

        # x_3 = self.base_3(x_att2)
        # x_3 = self.att_s3(x_3)
        # x_3 = self.BN3(x_3)
        # y_3 = self.att3(x_3)
        # x_att3=x_3*y_3.expand_as(x_3)

        # x_4 = self.base_4(x_att3)
        # x_4 = self.att_s4(x_4)
        # x_4 = self.BN4(x_4)
        # y_4 = self.att4(x_4)
        # x_att4=x_4*y_4.expand_as(x_4)

        # x_5 = self.base_5(x_att4)
        # x_5 = self.att_s5(x_5)
        # x_5 = self.BN5(x_5)
        # y_5 = self.att5(x_5)
        # x=x_5*y_5.expand_as(x_5) 
        ############

        x = self.base(x) #replace above with this to use base network

        attention_maps = self.attentions(x)

        # print(x.shape)
        # print(attention_maps.shape)

        global_feat,global_feat_hat = self.bap(x, attention_maps,counterfactual=True)

        global_feat = global_feat.view(global_feat.shape[0], -1)
        global_feat_hat = global_feat_hat.view(global_feat.shape[0], -1)

        global_feat = self.classifier_bap(global_feat)
        global_feat_hat = self.classifier_bap(global_feat_hat)
        print(global_feat.shape)
        print(global_feat_hat.shape)
      
        
        feat_hat = self.bottleneck(global_feat_hat)
        feat = self.bottleneck(global_feat)  # normalize for angular softmax


        if self.training:
            if self.using_cal:    
                return global_feat, global_feat_hat  # global feature for triplet loss
            else:
                return global_feat
        else:
            return global_feat

if __name__ == "__main__":
    model = Baseline()
    sat = torch.randn(5, 3, 256, 256)
    grd = torch.randn(5, 3, 122, 671)
    result = model(sat)