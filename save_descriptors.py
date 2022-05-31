import os
import argparse
import torch
import torch.nn as nn

from utils.utils import ReadConfig
from utils.analysis_utils import (
    GetBestModel, GetAllModel,
    set_dataset, set_model, DesHook
)


args_do_not_overide = ['data_dir', 'verbose', 'dataset', "model"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--data_dir", type=str, default='../scratch/', help='dir to the dataset')
    parser.add_argument('--dataset', default='both', choices=['CVUSA', 'CVACT', "both"], help='choose between CVUSA or CVACT')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument("--model", type=str, default="SAFA_TR", help='model')
    parser.add_argument('--model_path', type=str, help='path to model weights')
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--model_mode', type=str, default="best", choices=["best", "all"])

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

    print("start collecting descriptors...", flush=True)

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

        deshook = DesHook()
        deshook.register_des_hook(model)

        for ds in dataset_list:
            print(f"\ncollecting descriptors on {ds}...", flush=True)
            setattr(opt, "dataset", ds)
            setattr(opt, "data_dir", data_dir_dict[ds])
            validateloader = set_dataset(opt)

            model.eval()
            with torch.no_grad():
                for batch in validateloader:
                    sat = batch['satellite'].to(device)
                    grd = batch['ground'].to(device)

                    sat_global, grd_global = model(sat, grd, is_cf=False)
                    break

            epoch = a_model.split("/")[0]
            deshook.save_results(fname=f"des_{opt.model}_{epoch}_{origin_dataset}_{opt.geo_aug}_{opt.sem_aug}_{ds}.npz")