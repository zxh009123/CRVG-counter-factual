import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import argparse

from dataset.augmentations import Free_Flip, Free_Rotation, Free_Improper_Rotation
from utils.utils import ReadConfig, distancestat
from utils.analysis_utils import (
    GetBestModel, GetAllModel,
    set_dataset, set_model
)


args_do_not_overide = ['data_dir', 'verbose', 'dataset']

def validate_one(
    model, validateloader, embedding_dims, 
    aug_fn, degree, 
    device, verbose
):
    sat_global_descriptor = np.zeros([8884, embedding_dims])
    grd_global_descriptor = np.zeros([8884, embedding_dims])
    val_i = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(validateloader, disable=verbose):

            sat, grd = aug_fn(batch['satellite'], batch['ground'], degree)
            sat = sat.to(device)
            grd = grd.to(device)

            sat_global, grd_global = model(sat, grd, is_cf=False)

            sat_global_descriptor[val_i: val_i + sat_global.shape[0], :] = sat_global.detach().cpu().numpy()
            grd_global_descriptor[val_i: val_i + grd_global.shape[0], :] = grd_global.detach().cpu().numpy()

            val_i += sat_global.shape[0]
    
    valAcc = distancestat(sat_global_descriptor, grd_global_descriptor, fname=None)
    col_val_list = [
        valAcc[0, 0],
        valAcc[0, 1],
        valAcc[0, 2],
        valAcc[0, 3],
    ]
    row_val_list = [
        valAcc[1, 0],
        valAcc[1, 1],
        valAcc[1, 2],
        valAcc[1, 3],
    ]
    return col_val_list, row_val_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--data_dir", type=str, default='../scratch/', help='dir to the dataset')
    parser.add_argument('--dataset', default='CVUSA', choices=['CVUSA', 'CVACT'], help='choose between CVUSA or CVACT')
    parser.add_argument("--SAFA_heads", type=int, default=16, help='number of SAFA heads')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument("--model", type=str, help='model')
    parser.add_argument('--model_path', type=str, help='path to model weights')
    parser.add_argument('--no_polar', default=False, action='store_true', help='turn off polar transformation')
    parser.add_argument("--TR_heads", type=int, default=8, help='number of heads in Transformer')
    parser.add_argument("--TR_layers", type=int, default=6, help='number of layers in Transformer')
    parser.add_argument("--TR_dim", type=int, default=2048, help='dim of FFD in Transformer')
    parser.add_argument("--dropout", type=float, default=0.2, help='dropout in Transformer')
    parser.add_argument("--pos", type=str, default='learn_pos', help='positional embedding')
    parser.add_argument('--reg_mode', type=str, default="rotate", choices=["rotate", 'flip', 'improper'])
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--model_mode', type=str, default="best", choices=["best", "all"])

    opt = parser.parse_args()

    config = ReadConfig(opt.model_path)
    for k,v in config.items():
        if k in args_do_not_overide:
            continue
        setattr(opt, k, v)
    
    print(opt, flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    validateloader = set_dataset(opt)

    model, embedding_dims = set_model(opt)
    model = nn.DataParallel(model)
    model.to(device)

    print("start testing...", flush=True)
    if opt.model_mode == "best":
        best_model = GetBestModel(opt.model_path)
        model_list = [best_model]
    elif opt.model_mode == "all":
        model_list = GetAllModel(opt.model_path)

    if opt.reg_mode == "rotate":
        aug_fn = Free_Rotation
        deg_array = np.linspace(0., 360., 16)
    elif opt.reg_mode == "flip":
        aug_fn = Free_Flip
        deg_array = np.linspace(0., 180., 16)
    elif opt.reg_mode == "improper":
        aug_fn = Free_Improper_Rotation
        deg_array = np.linspace(0., 360., 16)

    for a_model in model_list:
        load_model = os.path.join(opt.model_path, a_model)
        print(f"loading model : {load_model}", flush=True)
        model.load_state_dict(torch.load(load_model)['model_state_dict'])

        res = {}
        for degree in deg_array:
            col_val_list, row_val_list = validate_one(
                model, validateloader, embedding_dims, aug_fn, degree, device, opt.verbose
            )
            res.update({
                degree: {
                    "col": col_val_list,
                    "row": row_val_list
                }
            })
        epoch = a_model.split("/")[0]
        fname = f"reg_{opt.model}_{opt.dataset}_{opt.reg_mode}_{epoch}"
        fname += ".npz" if opt.suffix is None else f"_{opt.suffix}.npz"
        np.savez_compressed(
            fname,
            reg_res = res
        )
        print(f"regression res saved to {fname}", flush=True)
    

