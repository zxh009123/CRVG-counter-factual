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

model = timm.create_model('vit_base_patch16_224', pretrained=True)


layers = list(model.children())[2:]
model = nn.Sequential(*layers)

pprint(model)

# a = torch.zeros(1,3,224,224)

# b = model(a)

# print(b[:,0:3])