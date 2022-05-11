import os
import argparse
import torch
import torch.nn as nn

from utils.utils import ReadConfig, distancestat
from utils.analysis_utils import (
    GetBestModel, GetAllModel,
    set_dataset, set_model, eval_model
)


args_do_not_overide = ['data_dir', 'verbose', 'dataset', "model"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--data_dir", type=str, default='../scratch/', help='dir to the dataset')
    parser.add_argument('--dataset', default='both', choices=['CVUSA', 'CVACT', "both"], help='choose between CVUSA or CVACT')
    parser.add_argument("--SAFA_heads", type=int, default=16, help='number of SAFA heads')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument("--model", type=str, default="SAFA_TR", help='model')
    parser.add_argument('--model_path', type=str, help='path to model weights')
    parser.add_argument('--no_polar', default=False, action='store_true', help='turn off polar transformation')
    parser.add_argument("--TR_heads", type=int, default=8, help='number of heads in Transformer')
    parser.add_argument("--TR_layers", type=int, default=6, help='number of layers in Transformer')
    parser.add_argument("--TR_dim", type=int, default=2048, help='dim of FFD in Transformer')
    parser.add_argument("--dropout", type=float, default=0.2, help='dropout in Transformer')
    parser.add_argument("--pos", type=str, default='learn_pos', help='positional embedding')
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

    model, embedding_dims = set_model(opt)
    model = nn.DataParallel(model)
    model.to(device)

    print("start testing...", flush=True)

    if opt.model_mode == "best":
        best_model = GetBestModel(opt.model_path)
        model_list = [best_model]
    elif opt.model_mode == "all":
        model_list = GetAllModel(opt.model_path)

    data_dir_dict = {
        "CVUSA": "/OceanStor100D/home/zhouyi_lab/xyli1905/dataset/Dataset_/cross-view/CVUSA/",
        "CVACT": "/OceanStor100D/home/zhouyi_lab/xyli1905/dataset/Dataset_/cross-view/CVACT/",
    }
    if opt.dataset == "both":
        dataset_list = ["CVUSA", "CVACT"]
    else:
        dataset_list = [opt.dataset]

    for a_model in model_list:
        load_model = os.path.join(opt.model_path, a_model)
        print(f"loading model : {load_model}", flush=True)
        model.load_state_dict(torch.load(load_model)['model_state_dict'])

        for ds in dataset_list:
            print(f"\ntesting on {ds}...", flush=True)
            setattr(opt, "dataset", ds)
            setattr(opt, "data_dir", data_dir_dict[ds])
            validateloader = set_dataset(opt)

            sat_global_descriptor, grd_global_descriptor = eval_model(model, validateloader, embedding_dims, device, opt.verbose)

            epoch = a_model.split("/")[0]
            file_name=f"./stats_distance_{opt.model}_{opt.dataset}_{epoch}"
            file_name += ".npz" if opt.suffix is None else f"_{opt.suffix}.npz"
            valAcc = distancestat(sat_global_descriptor, grd_global_descriptor, fname=file_name)
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
