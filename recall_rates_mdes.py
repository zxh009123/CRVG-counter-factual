import os
import argparse
import torch
import torch.nn as nn

from models.SAFA_TR_manual_des import GeoDTRmdes
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
    parser.add_argument('--dataset', default='both', choices=['CVUSA', 'CVACT', "both"], help='choose between CVUSA or CVACT')
    parser.add_argument("--SAFA_heads", type=int, default=8, help='number of SAFA heads')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument("--model", default="SAFA_TR", type=str, help='model')
    parser.add_argument('--model_path', type=str, help='path to model weights')
    parser.add_argument('--no_polar', default=False, action='store_true', help='turn off polar transformation')
    parser.add_argument("--TR_heads", type=int, default=4, help='number of heads in Transformer')
    parser.add_argument("--TR_layers", type=int, default=2, help='number of layers in Transformer')
    parser.add_argument("--TR_dim", type=int, default=2048, help='dim of FFD in Transformer')
    parser.add_argument("--dropout", type=float, default=0.2, help='dropout in Transformer')
    parser.add_argument("--pos", type=str, default='learn_pos', help='positional embedding')
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--des_path', type=str, help='path to saved descriptors (np.ndarray)')

    opt = parser.parse_args()

    config = ReadConfig(opt.model_path)
    for k,v in config.items():
        if k in args_do_not_overide:
            continue
        setattr(opt, k, v)
    print(opt, flush=True)

    polar_transformation = not opt.no_polar
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, embedding_dims = set_model(opt)
    model = GeoDTRmdes(
        safa_heads = opt.SAFA_heads, 
        tr_heads = opt.TR_heads, 
        tr_layers = opt.TR_layers, 
        dropout = opt.dropout, 
        d_hid = opt.TR_dim, 
        is_polar = polar_transformation, 
        pos = opt.pos,
        des_path = opt.des_path
    )
    embedding_dims = opt.SAFA_heads * 512
    model = nn.DataParallel(model)
    model.to(device)

    best_model = GetBestModel(opt.model_path)
    load_model = os.path.join(opt.model_path, best_model)
    print(f"loading model : {load_model}", flush=True)
    model.load_state_dict(torch.load(load_model)['model_state_dict'])

    print("start testing with standalone descriptors...", flush=True)

    data_dir_dict = {
        "CVUSA": "/OceanStor100D/home/zhouyi_lab/xyli1905/dataset/Dataset_/cross-view/CVUSA/",
        "CVACT": "/OceanStor100D/home/zhouyi_lab/xyli1905/dataset/Dataset_/cross-view/CVACT/",
    }
    if opt.dataset == "both":
        dataset_list = ["CVUSA", "CVACT"]
    else:
        dataset_list = [opt.dataset]


    for ds in dataset_list:
        print(f"\ntesting with standalone descriptors on {ds}...", flush=True)
        setattr(opt, "dataset", ds)
        setattr(opt, "data_dir", data_dir_dict[ds])
        validateloader = set_dataset(opt)

        sat_global_descriptor, grd_global_descriptor = eval_model(model, validateloader, embedding_dims, device, opt.verbose)
        valAcc = distancestat(sat_global_descriptor, grd_global_descriptor, fname=None)

        print(f"-----------validation result {ds}---------------", flush=True)
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
            print("", flush=True)
            print("In short for col acc:", flush=True)
            print(
                valAcc[0, 0] * 100.0, valAcc[0, 1] * 100.0, valAcc[0, 2] * 100.0, valAcc[0, 3] * 100.0,
                flush=True
            )
        except:
            print(valAcc)
        print(f"===================================================\n", flush=True)
