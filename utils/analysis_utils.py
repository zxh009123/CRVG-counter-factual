import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from dataset.usa_dataset import USADataset
from dataset.act_dataset import ACTDataset

from models.SAFA_TR import SAFA_TR
from models.SAFA_TR50 import SAFA_TR50
from models.SAFA_TR50_backup import SAFA_TR50 as SAFA_TR50_old
from models.SAFA_vgg import SAFA_vgg
from models.TK_SAFF import TK_SAFF
from models.TK_FFusion import TK_FFusion
from models.TK_FA_TR import TK_FA_TR


def GetBestModel(path):
    all_files = os.listdir(path)
    if "epoch_last" in all_files:
        all_files.remove("epoch_last")
    config_files =  list(filter(lambda x: x.startswith('epoch_'), all_files))
    config_files = sorted(list(map(lambda x: int(x.split("_")[1]), config_files)), reverse=True)
    best_epoch = config_files[0]
    return os.path.join('epoch_'+str(best_epoch), 'epoch_'+str(best_epoch)+'.pth')

def GetAllModel(path):
    all_files = os.listdir(path)
    if "epoch_last" in all_files:
        all_files.remove("epoch_last")
    config_files =  list(filter(lambda x: x.startswith('epoch_'), all_files))
    config_files = sorted(list(map(lambda x: int(x.split("_")[1]), config_files)), reverse=False)
    path_list = [
        os.path.join('epoch_'+str(fn), 'epoch_'+str(fn)+'.pth') for fn in config_files
    ]
    return path_list


def set_dataset(opt, geo_aug="none", sem_aug="none"):
    # set params
    batch_size = opt.batch_size
    polar_transformation = not opt.no_polar

    if opt.no_polar:
        SATELLITE_IMG_WIDTH = 256
        SATELLITE_IMG_HEIGHT = 256
        polar_transformation = False
    else:
        SATELLITE_IMG_WIDTH = 671
        SATELLITE_IMG_HEIGHT = 122
        polar_transformation = True
    print("SATELLITE_IMG_WIDTH:", SATELLITE_IMG_WIDTH, flush=True)
    print("SATELLITE_IMG_HEIGHT:", SATELLITE_IMG_HEIGHT, flush=True)

    STREET_IMG_WIDTH = 671
    STREET_IMG_HEIGHT = 122

    # select dataset
    if opt.dataset == 'CVACT':
        data_path = opt.data_dir
        validateloader = DataLoader(
            ACTDataset(
                data_dir = data_path, geometric_aug=geo_aug, sematic_aug=sem_aug, is_polar=polar_transformation, mode='val'
            ), 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=8
        )
    elif opt.dataset == 'CVUSA':
        data_path = opt.data_dir
        validateloader = DataLoader(
            USADataset(
                data_dir = data_path, geometric_aug=geo_aug, sematic_aug=sem_aug, mode="val", is_polar=polar_transformation
            ), 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=8
        )
    return validateloader


def set_model(opt):
    # set params
    number_SAFA_heads = opt.SAFA_heads
    polar_transformation = not opt.no_polar
    pos = "learn_pos" if opt.pos == "learn_pos" else None
    print("learnable positional embedding : ", pos, flush=True)

    # select model
    if opt.model == "SAFA_vgg":
        model = SAFA_vgg(safa_heads = number_SAFA_heads, is_polar=polar_transformation)
        embedding_dims = number_SAFA_heads * 512
    elif opt.model == "SAFA_TR":
        model = SAFA_TR(
            safa_heads=number_SAFA_heads, 
            tr_heads=opt.TR_heads, 
            tr_layers=opt.TR_layers, 
            dropout = opt.dropout, 
            d_hid=opt.TR_dim, 
            is_polar=polar_transformation, 
            pos=pos
        )
        embedding_dims = number_SAFA_heads * 512
    elif opt.model == "SAFA_TR50":
        model = SAFA_TR50(
            safa_heads=number_SAFA_heads, 
            tr_heads=opt.TR_heads, 
            tr_layers=opt.TR_layers, 
            dropout = opt.dropout, 
            d_hid=opt.TR_dim, 
            is_polar=polar_transformation, 
            pos=pos
        )
        embedding_dims = number_SAFA_heads * 512 * 2
    elif opt.model == "SAFA_TR50_old":
        model = SAFA_TR50_old(
            safa_heads=number_SAFA_heads, 
            tr_heads=opt.TR_heads, 
            tr_layers=opt.TR_layers, 
            dropout = opt.dropout, 
            d_hid=opt.TR_dim, 
            is_polar=polar_transformation, 
            pos=pos
        )
        embedding_dims = 8176
    elif opt.model == "TK_SAFF" or opt.model == "TK_FFusion" or opt.model == "TK_FA_TR":
        if opt.tkp == 'conv':
            TK_Pool = False
        else:
            TK_Pool = True

        if opt.model == "TK_SAFF":
            model = TK_SAFF(
                top_k=opt.topK, 
                tr_heads=opt.TR_heads, 
                tr_layers=opt.TR_layers, 
                dropout = opt.dropout, 
                is_polar=polar_transformation, 
                pos=pos, 
                TK_Pool=TK_Pool, 
                embed_dim=opt.embed_dim
            )
            embedding_dims = opt.embed_dim
        elif opt.model == "TK_FFusion":
            model = TK_FFusion(
                top_k=opt.topK, 
                tr_heads=opt.TR_heads, 
                tr_layers=opt.TR_layers, 
                dropout = opt.dropout, 
                pos = pos, 
                is_polar=polar_transformation, 
                TK_Pool=TK_Pool, 
                embed_dim=opt.embed_dim
            )
            embedding_dims = opt.embed_dim
        elif opt.model == "TK_FA_TR":
            model = TK_FA_TR(
                topk=opt.topK, 
                tr_heads=opt.TR_heads, 
                tr_layers=opt.TR_layers, 
                dropout = opt.dropout, 
                d_hid=2048, 
                pos = 'learn_pos', 
                is_polar=polar_transformation, 
                TKPool=TK_Pool
            )
            embedding_dims = opt.topK * 512
    else:
        raise RuntimeError(f"model {opt.model} is not implemented")

    return model, embedding_dims


def eval_model(
    model, loader, embedding_dims,
    device, verbose
):
    sat_global_descriptor = np.zeros([8884, embedding_dims])
    grd_global_descriptor = np.zeros([8884, embedding_dims])
    val_i = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, disable=verbose):
            sat = batch['satellite'].to(device)
            grd = batch['ground'].to(device)

            sat_global, grd_global = model(sat, grd, is_cf=False)

            sat_global_descriptor[val_i: val_i + sat_global.shape[0], :] = sat_global.detach().cpu().numpy()
            grd_global_descriptor[val_i: val_i + grd_global.shape[0], :] = grd_global.detach().cpu().numpy()

            val_i += sat_global.shape[0]
    # np.savez_compressed(
    #     f"./sat_grd_global_features_{opt.dataset}.npz",
    #     sat_global_descriptor = sat_global_descriptor,
    #     grd_global_descriptor = grd_global_descriptor
    # )
    return sat_global_descriptor, grd_global_descriptor