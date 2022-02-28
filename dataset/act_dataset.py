import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import scipy.io as sio
import torchvision
import argparse
import torchvision.transforms as transforms
# __all__ = ['TrainDataloader','TestDataloader']


class TrainDataset(Dataset):
    def __init__(self, data_dir, transforms_sat, transforms_grd, is_polar=True):

        # self.polar = args.polar

        self.img_root = data_dir
        self.transform_sat = transforms.Compose(transforms_sat)
        self.transform_grd = transforms.Compose(transforms_grd)

        self.allDataList = '/mnt/CVACT/ACT_data.mat'

        __cur_allid = 0  # for training
        id_alllist = []
        id_idx_alllist = []

        # load the mat
        anuData = sio.loadmat(self.allDataList)


        idx = 0
        for i in range(0,len(anuData['panoIds'])):
            
            # if self.polar:
            #     grd_id_align = self.img_root + 'streetview/' + anuData['panoIds'][i] + '_grdView.png'
            #     sat_id_ori = self.img_root + 'polarmap/' + anuData['panoIds'][i] + '_satView_polish.png'
            # else:
            #     grd_id_align = self.img_root + 'streetview/' + anuData['panoIds'][i] + '_grdView.jpg'
            #     sat_id_ori = self.img_root + 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.jpg'
            grd_id_align = os.path.join(self.img_root, 'ANU_data_small', 'streetview_processed', anuData['panoIds'][i] + '_grdView.png')
            if is_polar:
                sat_id_ori = os.path.join(self.img_root, 'ANU_data_small', 'polarmap', anuData['panoIds'][i] + '_satView_polish.png')
            else:
                sat_id_ori = os.path.join(self.img_root, 'ANU_data_small', 'satview_polish', anuData['panoIds'][i] + '_satView_polish.jpg')
            id_alllist.append([ grd_id_align, sat_id_ori])
            id_idx_alllist.append(idx)
            idx += 1

        all_data_size = len(id_alllist)
        print('InputData::__init__: load', self.allDataList, ' data_size =', all_data_size)



        training_inds = anuData['trainSet']['trainInd'][0][0] - 1
        trainNum = len(training_inds)
        print('trainSet:' ,trainNum)
        self.trainList = []
        self.trainIdList = []

        
        for k in range(trainNum):
           
            self.trainList.append(id_alllist[training_inds[k][0]])
            self.trainIdList.append(k)  


    def __getitem__(self, idx):

        
        x = Image.open(self.trainList[idx][0])
        # width, height = x.size
        # x = x.crop((0, 265, width, 265+302))
        x = self.transform_sat(x)
        
        y = Image.open(self.trainList[idx][1])
        # if self.polar:
        #     y = self.transform(y)
        # else:
        #     y = self.transform_1(y)
        y = self.transform_grd(y)

        # return x, y
        return {'satellite':y, 'ground':x}

    def __len__(self):
        return len(self.trainList)


class TestDataset(Dataset):
    def __init__(self, data_dir, transforms_sat, transforms_grd, is_polar=True):

        # self.polar = args.polar
        self.img_root = data_dir
        self.transform_sat = transforms.Compose(transforms_sat)
        self.transform_grd = transforms.Compose(transforms_grd)

        self.allDataList = '/mnt/CVACT/ACT_data.mat'

        __cur_allid = 0  # for training
        id_alllist = []
        id_idx_alllist = []

        # load the mat
        anuData = sio.loadmat(self.allDataList)
        # print(anuData)

        idx = 0
        for i in range(0,len(anuData['panoIds'])):
            
            # if self.polar:
            #     # polar transform and crop the ground view
            #     grd_id_align = self.img_root + 'streetview/' + anuData['panoIds'][i] + '_grdView.png'
            #     sat_id_ori = self.img_root + 'polarmap/' + anuData['panoIds'][i] + '_satView_polish.png'
            # else:
            #     grd_id_align = self.img_root + 'streetview/' + anuData['panoIds'][i] + '_grdView.jpg'
            #     sat_id_ori = self.img_root + 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.jpg'
            grd_id_align = os.path.join(self.img_root, 'ANU_data_test', 'streetview_processed', anuData['panoIds'][i] + '_grdView.png')
            if is_polar:
                sat_id_ori = os.path.join(self.img_root, 'ANU_data_test', 'polarmap', anuData['panoIds'][i] + '_satView_polish.png')
            else:
                sat_id_ori = os.path.join(self.img_root, 'ANU_data_test', 'satview_polish', anuData['panoIds'][i] + '_satView_polish.jpg')
            id_alllist.append([ grd_id_align, sat_id_ori])
            id_idx_alllist.append(idx)
            idx += 1

        all_data_size = len(id_alllist)
        print('InputData::__init__: load', self.allDataList, ' data_size =', all_data_size)


        self.val_inds = anuData['valSet']['valInd'][0][0] - 1
        # self.val_inds = anuData['valSetAll']['valInd'][0][0] - 1
        self.valNum = len(self.val_inds)
        print('valSet:' ,self.valNum)
        self.valList = []

        for k in range(self.valNum):
            self.valList.append(id_alllist[self.val_inds[k][0]])

        self.__cur_test_id = 0      

    def __getitem__(self, idx):
        x = Image.open(self.valList[idx][0])
        x = self.transform_grd(x)

        y = Image.open(self.valList[idx][1])
        y = self.transform_sat(y)

        # return x, y
        return {'satellite':y, 'ground':x}

    def __len__(self):
        return len(self.valList)

if __name__ == "__main__":
    transforms_sat = [transforms.Resize((122, 671)),
                        transforms.ColorJitter(0.1, 0.1, 0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                        ]
    transforms_grd = [transforms.Resize((122, 671)),
                        transforms.ColorJitter(0.1, 0.1, 0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                        ]
    # dataloader = DataLoader(TrainDataset(data_dir = "/mnt/CVACT/", transforms_sat=transforms_sat, transforms_grd=transforms_grd, is_polar=False),batch_size=32, shuffle=True, num_workers=8)
    dataloader = DataLoader(TestDataset(data_dir = "/mnt/CVACT/", transforms_sat=transforms_sat, transforms_grd=transforms_grd, is_polar=True),batch_size=4, shuffle=True, num_workers=8)

    i = 0
    for k in dataloader:
        i += 1
        print("---batch---")
        print("satellite : ", k['satellite'].shape)
        print("grd : ", k['ground'].shape)
        print("-----------")
        if i > 2:
            break
    