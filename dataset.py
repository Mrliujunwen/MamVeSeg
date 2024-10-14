import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import cv2
from PIL import Image
from torchvision import transforms


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class MyDataset(Dataset):
    '''
    dir_path: path to data, having two folders named data and label respectively
    '''
    def __init__(self,dir_path,transform = None,in_chan = 3):
        self.dir_path = dir_path
        self.transform = transform
        self.data_path = os.path.join(dir_path,"data")
        self.data_lists = sorted(glob.glob(os.path.join(self.data_path,"*.png")))
        self.label_path = os.path.join(dir_path,"label")
        self.label_lists = sorted(glob.glob(os.path.join(self.label_path,"*.png")))

        self.in_chan = in_chan

    def __getitem__(self,index):
        img_path = self.data_lists[index]
        label_path = self.label_lists[index]
        if self.in_chan == 3:
            img = Image.open(img_path).convert("RGB")

        else:
            img = Image.open(img_path).convert("L")


        # img = img /255.0
        label = cv2.imread(label_path)


        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_LINEAR)
        label_copy = label.copy()

        semantic_mask = label.copy()
        semantic_mask[(semantic_mask != 0)] = 1


        label_copy[(label_copy == 255)|(label_copy != 0)]=1

        instance_mask = label_copy
        normal_edge_mask = label_copy
        cluster_edge_mask = label_copy


        transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Resize((512,512)),
            transforms.Normalize(np.array([x for x in [0.6020, 0.3329, 0.1107]]),
                                     np.array([x for x in [0.3528, 0.1922, 0.1083]]))
        ])

        img = transform(img)
        # semantic_mask = transform(semantic_mask)
        semantic_mask = torch.tensor(semantic_mask)
        img = img.to(device)
        semantic_mask = semantic_mask.to(device)
        instance_mask = torch.tensor(instance_mask).to(device)
        normal_edge_mask = torch.tensor(normal_edge_mask).to(device)
        cluster_edge_mask = torch.tensor(cluster_edge_mask).to(device)


        return img,instance_mask,semantic_mask, normal_edge_mask,cluster_edge_mask

    def __len__(self):
        return len(self.data_lists)


    def sem2ins(self,label):
        seg_mask_g = label.copy()

        seg_mask_g[seg_mask_g != 255] = 0
        seg_mask_g[seg_mask_g == 255] = 1

        return seg_mask_g

    def generate_normal_edge_mask(self,label):

        normal_edge_mask = label.copy()


        normal_edge_mask[(normal_edge_mask < 255) & (normal_edge_mask > 0) ] = 1
        normal_edge_mask[normal_edge_mask == 255] = 0





        return normal_edge_mask
    def generate_cluster_edge_mask(self,label):

        cluster_edge_mask = label.copy()


        cluster_edge_mask[(cluster_edge_mask < 255) & (cluster_edge_mask > 0)] = 1
        cluster_edge_mask[cluster_edge_mask == 255] = 0



        return cluster_edge_mask

