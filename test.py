import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torchvision.models as models
from torch.utils.data import DataLoader
from dataset.usa_dataset import ImageDataset, USADataset
# from dataset.act_dataset import TestDataset, TrainDataset
from dataset.act_dataset import ACTDataset
# from SMTL import softMarginTripletLoss
from tqdm import tqdm
import os
import numpy as np
import argparse
import json
import scipy.io as sio
from utils.utils import ReadConfig, ValidateAll, validatenp

from models.SAFA_TR import SAFA_TR
from models.SAFA_TR50 import SAFA_TR50
from models.SAFA_vgg import SAFA_vgg
try:
    from models.TK_SAFF import TK_SAFF
    from models.TK_FFusion import TK_FFusion
    from models.TK_FA_TR import TK_FA_TR
except:
    pass

args_do_not_overide = ['data_dir', 'verbose', 'dataset']

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def ValidateOne(distArray, topK):
    acc = 0.0
    dataAmount = 0.0
    for i in range(distArray.shape[0]):
        groundTruths = distArray[i,i]
        pred = torch.sum(distArray[:,i] < groundTruths)
        if pred < topK:
            acc += 1.0
        dataAmount += 1.0
    return acc / dataAmount

def ValidateAll(streetFeatures, satelliteFeatures):
    distArray = 2 - 2 * torch.matmul(satelliteFeatures, torch.transpose(streetFeatures, 0, 1))
    topOnePercent = int(distArray.shape[0] * 0.01) + 1
    valAcc = torch.zeros((1, topOnePercent))
    for i in range(topOnePercent):
        valAcc[0,i] = ValidateOne(distArray, i)
    
    return valAcc

def GetBestModel(path):
    all_files = os.listdir(path)
    if "epoch_last" in all_files:
        all_files.remove("epoch_last")
    config_files =  list(filter(lambda x: x.startswith('epoch_'), all_files))
    config_files = sorted(list(map(lambda x: int(x.split("_")[1]), config_files)), reverse=True)
    best_epoch = config_files[0]
    return os.path.join('epoch_'+str(best_epoch), 'epoch_'+str(best_epoch)+'.pth')

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
    parser.add_argument('--validate', default=True, action='store_false', help='perform validate or not')
    parser.add_argument('--save', default='none', choices=['none', 'mat', 'pt'], help='save extracted features into .mat files or pt files')

    opt = parser.parse_args()

    config = ReadConfig(opt.model_path)
    for k,v in config.items():
        if k in args_do_not_overide:
            continue
        setattr(opt, k, v)
    
    print(opt)

    batch_size = opt.batch_size
    number_SAFA_heads = opt.SAFA_heads

    if opt.no_polar:
        SATELLITE_IMG_WIDTH = 256
        SATELLITE_IMG_HEIGHT = 256
        polar_transformation = False
    else:
        SATELLITE_IMG_WIDTH = 671
        SATELLITE_IMG_HEIGHT = 122
        polar_transformation = True
    print("SATELLITE_IMG_WIDTH:",SATELLITE_IMG_WIDTH)
    print("SATELLITE_IMG_HEIGHT:",SATELLITE_IMG_HEIGHT)

    STREET_IMG_WIDTH = 671
    STREET_IMG_HEIGHT = 122

    if opt.pos == "learn_pos":
        pos = "learn_pos"
    else:
        pos = None
    print("learnable positional embedding : ", pos)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if opt.dataset == 'CVACT':
        data_path = os.path.join(opt.data_dir, 'CVACT')
        dataset = ACTDataset(data_dir = data_path, geometric_aug='none', sematic_aug='none', is_polar=polar_transformation, mode='val')
        validateloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    if opt.dataset == 'CVUSA':
        data_path = os.path.join(opt.data_dir, 'CVUSA', 'dataset')
        dataset = USADataset(data_dir = data_path, geometric_aug='none', sematic_aug='none', mode='val', is_polar=polar_transformation)
        validateloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    print("number of test samples : ", len(dataset))

    if opt.model == "SAFA_vgg":
        model = SAFA_vgg(safa_heads = number_SAFA_heads, is_polar=polar_transformation)
        embedding_dims = number_SAFA_heads * 512
    elif opt.model == "SAFA_TR":
        model = SAFA_TR(safa_heads=number_SAFA_heads, tr_heads=opt.TR_heads, tr_layers=opt.TR_layers, dropout = opt.dropout, d_hid=opt.TR_dim, is_polar=polar_transformation, pos=pos)
        embedding_dims = number_SAFA_heads * 512
    elif opt.model == "SAFA_TR50":
        model = SAFA_TR50(safa_heads=number_SAFA_heads, tr_heads=opt.TR_heads, tr_layers=opt.TR_layers, dropout = opt.dropout, d_hid=opt.TR_dim, is_polar=polar_transformation, pos=pos)
        embedding_dims = number_SAFA_heads * 512
    elif opt.model == "TK_SAFF" or opt.model == "TK_FFusion" or opt.model == "TK_FA_TR":
        if opt.tkp == 'conv':
            TK_Pool = False
        else:
            TK_Pool = True

        if opt.model == "TK_SAFF":
            model = TK_SAFF(top_k=opt.topK, tr_heads=opt.TR_heads, tr_layers=opt.TR_layers, dropout = opt.dropout, is_polar=polar_transformation, pos=pos, TK_Pool=TK_Pool, embed_dim=opt.embed_dim)
            embedding_dims = opt.embed_dim
        elif opt.model == "TK_FFusion":
            model = TK_FFusion(top_k=opt.topK, tr_heads=opt.TR_heads, tr_layers=opt.TR_layers, dropout = opt.dropout, pos = pos, is_polar=polar_transformation, TK_Pool=TK_Pool, embed_dim=opt.embed_dim)
            embedding_dims = opt.embed_dim
        elif opt.model == "TK_FA_TR":
            model = TK_FA_TR(topk=opt.topK, tr_heads=opt.TR_heads, tr_layers=opt.TR_layers, dropout = opt.dropout, d_hid=2048, pos = 'learn_pos', is_polar=polar_transformation, TKPool=TK_Pool, project_dim=opt.project_dim)
            embedding_dims = opt.topK * 512
    else:
        raise RuntimeError(f"model {opt.model} is not implemented")
    model = nn.DataParallel(model)
    model.to(device)

    # best_model = GetBestModel(opt.model_path)
    best_epoch = '80'
    best_model = os.path.join(opt.model_path, 'epoch_'+str(best_epoch), 'epoch_'+str(best_epoch)+'.pth')
    # best_model = os.path.join(opt.model_path, best_model)
    print("loading model : ", best_model)
    model.load_state_dict(torch.load(best_model)['model_state_dict'])

    num_params = count_parameters(model)
    print(f"model parameters : {num_params}M")

    print("start testing...")

    # valSateFeatures = None
    # valStreetFeature = None

    sat_global_descriptor = np.zeros([len(dataset), embedding_dims])
    grd_global_descriptor = np.zeros([len(dataset), embedding_dims])
    val_i = 0

    model.eval()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings=np.zeros((len(validateloader),1))
    index = 0
    with torch.no_grad():
        for batch in tqdm(validateloader, disable=opt.verbose):
            sat = batch['satellite'].to(device)
            # sat = sat.reshape(sat.shape[0], sat.shape[2], sat.shape[3], sat.shape[4])
            grd = batch['ground'].to(device)
            # grd = batch['street'][:,batch['street'].shape[1] // 2]

            starter.record()
            sat_global, grd_global = model(sat, grd, is_cf=False)
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[index] = curr_time
            index += 1

            sat_global_descriptor[val_i: val_i + sat_global.shape[0], :] = sat_global.detach().cpu().numpy()
            grd_global_descriptor[val_i: val_i + grd_global.shape[0], :] = grd_global.detach().cpu().numpy()

            val_i += sat_global.shape[0]
    
    mean_syn = np.sum(timings) / len(validateloader)
    print("average inference time : ",mean_syn)

    if opt.validate:
        # valAcc = ValidateAll(valStreetFeature, valSateFeatures)
        valAcc = validatenp(sat_global_descriptor, grd_global_descriptor)
        print(f"-----------validation result---------------")
        try:
            #print epoch loss
            top1 = valAcc[0, 1]
            print('top1', ':', valAcc[0, 1] * 100.0)
            print('top5', ':', valAcc[0, 5] * 100.0)
            print('top10', ':', valAcc[0, 10] * 100.0)
            print('top1%', ':', valAcc[0, -1] * 100.0)
        except:
            print(valAcc)
        print(f"=================================================")

    #save features
    if opt.save == 'mat':
        features_dict = {'sat_global_descriptor':sat_global_descriptor, 'grd_global_descriptor':grd_global_descriptor}
        sio.savemat("descriptors.mat", features_dict)
    elif opt.save == 'pt':
        torch.save(valSateFeatures, 'sat_global_descriptor.pt')
        torch.save(valStreetFeature, 'grd_global_descriptor.pt')
