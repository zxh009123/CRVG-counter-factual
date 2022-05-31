import torch
import torch.nn.functional as F
import numpy as np
from .SAFA_TR import SAFA_TR



class GeoDTRmdes(SAFA_TR):
    def __init__(
        self, 
        in_dim, 
        safa_heads = 8, 
        tr_heads = 4, 
        tr_layers = 2, 
        dropout = 0.3, 
        d_hid = 2048, 
        pos = 'learn_pos',
        des_path = None
    ):
        super(GeoDTRmdes, self).__init__(
            in_dim = in_dim, 
            safa_heads = safa_heads, 
            tr_heads = tr_heads, 
            tr_layers = tr_layers, 
            dropout = dropout, 
            d_hid = d_hid, 
            pos = pos
        )
        self._load_des(des_path)
        self.allindevice = False

    def _load_des(self, des_path):
        # self.sat_sa
        # self.grd_sa
        with np.load(des_path, allow_pickle=True) as data:
            adict = data['des_holder'].item()
        self.sat_sa = torch.from_numpy(adict["sat"])
        self.grd_sa = torch.from_numpy(adict["grd"])

    def _des2device(self, device):
        self.sat_sa = self.sat_sa.to(device)
        self.grd_sa = self.grd_sa.to(device)
        self.allindevice = True

    def forward(self, sat, grd, is_cf=False): # is_cf, legacy
        b = sat.shape[0]
        if not self.allindevice:
            self._des2device(sat.device)

        sat_x = self.backbone_sat(sat)
        grd_x = self.backbone_grd(grd)

        sat_x = sat_x.view(b, sat_x.shape[1], -1)
        grd_x = grd_x.view(b, grd_x.shape[1], -1)
    
        sat_sa = torch.tile(self.sat_sa, (b, 1, 1))
        grd_sa = torch.tile(self.grd_sa, (b, 1, 1))

        sat_global = torch.matmul(sat_x, sat_sa).view(b,-1)
        grd_global = torch.matmul(grd_x, grd_sa).view(b,-1)

        sat_global = F.normalize(sat_global, p=2, dim=1)
        grd_global = F.normalize(grd_global, p=2, dim=1)

        return sat_global, grd_global