# ========================================================
# measure the alignment between the 
#       distance in the latent space
# and
#       ground truth distance
# ========================================================

import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from utils.utils import ReadConfig, distancestat
from utils.analysis_utils import (
    GetBestModel,
    set_dataset, set_model, eval_model
)


args_do_not_overide = ['data_dir', 'verbose', 'dataset', 'model']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--data_dir", type=str, default=None, help='dir to the dataset')
    parser.add_argument('--dataset', default='CVACT', choices=['CVUSA', 'CVACT', "both"], help='choose between CVUSA or CVACT')
    parser.add_argument("--SAFA_heads", type=int, default=16, help='number of SAFA heads')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument("--model", default="SAFA_TR", type=str, help='model')
    parser.add_argument('--model_path', type=str, help='path to model weights')
    parser.add_argument('--no_polar', default=False, action='store_true', help='turn off polar transformation')
    parser.add_argument("--TR_heads", type=int, default=8, help='number of heads in Transformer')
    parser.add_argument("--TR_layers", type=int, default=6, help='number of layers in Transformer')
    parser.add_argument("--TR_dim", type=int, default=2048, help='dim of FFD in Transformer')
    parser.add_argument("--dropout", type=float, default=0.2, help='dropout in Transformer')
    parser.add_argument("--pos", type=str, default='learn_pos', help='positional embedding')
    parser.add_argument('--suffix', type=str, default=None)

    opt = parser.parse_args()

    config = ReadConfig(opt.model_path)
    for k,v in config.items():
        if k in args_do_not_overide:
            if k == "dataset":
                origin_dataset = v
            continue
        setattr(opt, k, v)
    print(opt, flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, embedding_dims = set_model(opt)
    model = nn.DataParallel(model)
    model.to(device)

    best_model = GetBestModel(opt.model_path)
    load_model = os.path.join(opt.model_path, best_model)
    print(f"loading model : {load_model}", flush=True)
    model.load_state_dict(torch.load(load_model)['model_state_dict'])

    print("start testing...", flush=True)

    data_dir_dict = {
        "CVUSA": "../scratch/CVUSA/dataset",
        "CVACT": "../scratch/CVACT/",
    }
    if opt.dataset == "both":
        dataset_list = ["CVUSA", "CVACT"]
    else:
        dataset_list = [opt.dataset]


    for ds in dataset_list:
        print(f"\ntesting on {ds}...", flush=True)
        setattr(opt, "dataset", ds)
        setattr(opt, "data_dir", data_dir_dict[ds])
        validateloader = set_dataset(opt, mode="val")

        sat_global_descriptor, grd_global_descriptor = eval_model(model, validateloader, embedding_dims, device, opt.verbose)

        ss_dist = 2.0 - 2.0 * np.matmul(sat_global_descriptor, sat_global_descriptor.T)
        sg_dist = 2.0 - 2.0 * np.matmul(sat_global_descriptor, grd_global_descriptor.T)
        gg_dist = 2.0 - 2.0 * np.matmul(grd_global_descriptor, grd_global_descriptor.T)

        fname = f"stats_corr_dist_{opt.model}_{opt.dataset}_{origin_dataset}_{opt.geo_aug}_{opt.sem_aug}"
        fname += ".npz" if opt.suffix is None else f"_{opt.suffix}.npz"
        np.savez_compressed(
            fname,
            ss_dist = ss_dist,
            sg_dist = sg_dist,
            gg_dist = gg_dist,
        )
        print(f"distance dist saved to {fname}", flush=True)