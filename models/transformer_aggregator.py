# import torch
# import vision_transformer
# from pprint import pprint

# model = vision_transformer.vit_large_patch16_224(pretrained=True)
# a = torch.rand(1,3,224,224)
# pprint(model)

# b = model(a)

# print(b.shape)

import timm
import torch
from pprint import pprint
import torch.nn as nn


def Create_MHSA(model='vit_base_patch16_224', pretrained=True, num_layers=4):
    model = timm.create_model(model, pretrained=pretrained)

    for name, param in model.named_parameters():
        if name == 'cls_token':
            cls_token = param
        if name == 'pos_embed':
            pos_embed = param

    layers = list(model.children())

    MHSA = layers[2]

    MHSA_layers = list(MHSA.children())
    MHSA_layers_in = MHSA_layers[0:int(num_layers/2)]
    MHSA_layers_out = MHSA_layers[len(MHSA_layers)-int(num_layers/2):]
    # model = nn.Sequential(*MHSA_layers_in, *MHSA_layers_out)
    model = MHSA
    return model, cls_token, pos_embed


if __name__ == "__main__":

    model, ct = Create_MHSA()

    print(ct)
    print(type(ct))
    #batch first
    # a = torch.zeros(5,7,768)

    # b = model(a)

    # print(model)

    # print(b.shape)
    # print(b[:,0, 0:3])