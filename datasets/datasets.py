
import os
import gdal
import random
from torch.utils.data import Dataset
import torch
from torchvision import transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, folder,crop_size):
        self.folder = folder
        self.hyper_path = os.listdir(folder)
        self.crop_size=crop_size

    def __len__(self):
        return len(self.hyper_path)

    def __getitem__(self, index):

        hyper_path = self.folder+'/'+self.hyper_path[index]

        dataset = gdal.Open(hyper_path)
        if dataset is None:
            print("cannot open %s" % hyper_path)
            exit(-1)
        im_data = dataset.ReadAsArray()
        hs, ws, h, w = self.randomcrop(im_data, self.crop_size)
        hyper = im_data[:, hs:hs+h,ws:ws+w]

        # hyper_img=hyper.transpose(1, 2, 0)
        hyper_img = torch.tensor(hyper.astype('float32')/4095)
        multi_imgt = hyper_img[[35, 18, 8], :, :]
        mean_multi = tuple([0.5292, 0.5223, 0.3793])
        std_multi = tuple([0.2660, 0.2559, 0.1760])

        transform_m = transforms.Normalize(mean=mean_multi, std=std_multi)
        multi_img=transform_m(multi_imgt)

        return multi_img, hyper_img

    def randomcrop(self,img,crop_size):
        c,h,w= img.shape
        th= tw = crop_size
        if w == tw and h == th:
            return 0,0,h,w
        if w < tw or h < th:
            print('the size of the image is not enough for crop!')
            exit(-1)
        i = random.randint(0, h - th-10)
        j = random.randint(0, w - tw-10)
        return i,j,th,tw






