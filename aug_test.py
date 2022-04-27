import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import argparse

from utils.utils import ReadConfig, distancestat
from utils.analysis_utils import (
    GetBestModel, GetAllModel,
    set_dataset, set_model
)


args_do_not_overide = ['data_dir', 'verbose', 'dataset', "model"]

def validate_one(
    model, loader, embedding_dims, device, num_limit, fname, verbose
):
    num = 8884 if num_limit else len(loader.dataset)
    print(f"number of samples: {num}", flush=True)

    sat_global_descriptor = np.zeros([num, embedding_dims])
    grd_global_descriptor = np.zeros([num, embedding_dims])
    val_i = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, disable=verbose):

            sat, grd = batch['satellite'], batch['ground']
            sat = sat.to(device)
            grd = grd.to(device)

            sat_global, grd_global = model(sat, grd, is_cf=False)

            sat_global_descriptor[val_i: val_i + sat_global.shape[0], :] = sat_global.detach().cpu().numpy()
            grd_global_descriptor[val_i: val_i + grd_global.shape[0], :] = grd_global.detach().cpu().numpy()

            val_i += sat_global.shape[0]
    
    valAcc = distancestat(sat_global_descriptor, grd_global_descriptor, fname=fname)
    # col_val_list = [
    #     valAcc[0, 0],
    #     valAcc[0, 1],
    #     valAcc[0, 2],
    #     valAcc[0, 3],
    # ]
    # row_val_list = [
    #     valAcc[1, 0],
    #     valAcc[1, 1],
    #     valAcc[1, 2],
    #     valAcc[1, 3],
    # ]
    # return col_val_list, row_val_list
    return valAcc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--data_dir", type=str, default='../scratch/', help='dir to the dataset')
    parser.add_argument('--dataset', default='CVUSA', choices=['CVUSA', 'CVACT'], help='choose between CVUSA or CVACT')
    parser.add_argument("--dataset_type", default="val", choices=["val", "train"])
    parser.add_argument("--SAFA_heads", type=int, default=16, help='number of SAFA heads')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument("--model", type=str, help='model')
    parser.add_argument('--model_path', type=str, default=None, help='path to model weights')
    parser.add_argument('--no_polar', default=False, action='store_true', help='turn off polar transformation')
    parser.add_argument("--TR_heads", type=int, default=8, help='number of heads in Transformer')
    parser.add_argument("--TR_layers", type=int, default=6, help='number of layers in Transformer')
    parser.add_argument("--TR_dim", type=int, default=2048, help='dim of FFD in Transformer')
    parser.add_argument("--dropout", type=float, default=0.2, help='dropout in Transformer')
    parser.add_argument("--pos", type=str, default='learn_pos', help='positional embedding')
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--model_mode', type=str, default="best", choices=["best", "all"])
    parser.add_argument("--save_dist", default=False, action='store_true')
    parser.add_argument("--num_limit", default=False, action='store_true')
    parser.add_argument("--geo_aug", type=str, default="none", choices=["none", "weak", "strong"])
    parser.add_argument("--sem_aug", type=str, default="none", choices=["none", "weak", "strong"])

    opt = parser.parse_args()

    config = ReadConfig(opt.model_path)
    for k,v in config.items():
        if k in args_do_not_overide:
            continue
        # elif k == "model" and opt.model == "SAFA_TR50_old":
        #     continue
        setattr(opt, k, v)
    
    print(opt, flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    loader = set_dataset(opt, geo_aug=opt.geo_aug, sem_aug=opt.sem_aug, mode=opt.dataset_type)

    model, embedding_dims = set_model(opt)
    model = nn.DataParallel(model)
    model.to(device)

    print("start testing...", flush=True)
    if opt.model_path is None:
        model_list = ["dummy/dummy"]
    else:
        if opt.model_mode == "best":
            best_model = GetBestModel(opt.model_path)
            model_list = [best_model]
        elif opt.model_mode == "all":
            model_list = GetAllModel(opt.model_path)

    for a_model in model_list:
        if opt.model_path is not None:
            load_model = os.path.join(opt.model_path, a_model)
            print(f"loading model : {load_model}", flush=True)
            model.load_state_dict(torch.load(load_model)['model_state_dict'])

        if opt.save_dist:
            epoch = a_model.split("/")[0]
            fname = f"aug_ddist_{opt.model}_{opt.dataset}_{opt.dataset_type}_g{opt.geo_aug}_s{opt.sem_aug}_{epoch}"
            fname += ".npz" if opt.suffix is None else f"_{opt.suffix}.npz"
        else:
            fname = None

        valAcc = validate_one(
            model, loader, embedding_dims, device, 
            opt.num_limit, fname,
            opt.verbose
        )
        print(f"-----------validation result {epoch}---------------", flush=True)
        try:
            print(f'col_top1:  {valAcc[0, 0] * 100.0}', flush=True)
            print(f'col_top5:  {valAcc[0, 1] * 100.0}', flush=True)
            print(f'col_top10: {valAcc[0, 2] * 100.0}', flush=True)
            print(f'col_top1%: {valAcc[0, 3] * 100.0}', flush=True)
            print("", flush=True)
            print(f'row_top1:  {valAcc[1, 0] * 100.0}', flush=True)
            print(f'row_top5:  {valAcc[1, 1] * 100.0}', flush=True)
            print(f'row_top10: {valAcc[1, 2] * 100.0}', flush=True)
            print(f'row_top1%: {valAcc[1, 3] * 100.0}', flush=True)
        except:
            print(valAcc)
        print(f"===================================================\n", flush=True)

        

    

