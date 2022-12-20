import os

import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms

from PIL import Image

class EyeDataset(Dataset):
    def __init__(self,data_dir, transform=None):
        '''
        睁眼、闭眼二分类任务数据集
        :param data_dir: str,数据集所在路径
        :param transform: torch.transform,数据预处理
        '''
        self.label_name = {'bg_closed':0,'closed':1}
        self.data_dir = data_dir
        self.data_info = self.get_img_info()
        self.transform = transform
    
    def __getitem__(self, index):
        path_img,label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    
    def __len__(self):
        return len(self.data_info)
    
    # @staticmethod
    def get_img_info(self):# 获取数据路径和标签
        data_info = list()
        for root, dirs, _ in os.walk(self.data_dir):
            # 依据文件夹名字遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root,sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                
                # 遍历该类别所有图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root,sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img,int(label)))
        return data_info
    
class BinaryClassifyDataset(Dataset):
    def __init__(self,data_dir, label_name, transform=None):
        '''
        二分类任务数据集
        :param data_dir: str,数据集所在路径
        :param transform: torch.transform,数据预处理
        '''
        self.data_dir = data_dir
        self.label_name = label_name
        self.data_info = self.get_img_info()
        self.transform = transform
    
    def __getitem__(self, index):
        path_img,label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    
    def __len__(self):
        return len(self.data_info)
    

    # @staticmethod
    def get_img_info(self):# 获取数据路径和标签
        data_info = list()
        for root, dirs, _ in os.walk(self.data_dir):
            # 依据文件夹名字遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root,sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                
                # 遍历该类别所有图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root,sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img,int(label)))
        return data_info

class MultiClassifyDataset(Dataset):
    def __init__(self,data_dir, label_name, transform=None):
        '''
        多分类任务数据集
        :param data_dir: str,数据集所在路径
        :param transform: torch.transform,数据预处理
        '''
        self.data_dir = data_dir
        self.label_name = label_name
        self.data_info = self.get_img_info()
        self.transform = transform
    
    def __getitem__(self, index):
        path_img,label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    
    def __len__(self):
        return len(self.data_info)
    

    # @staticmethod
    def get_img_info(self):# 获取数据路径和标签
        data_info = list()
        for root, dirs, _ in os.walk(self.data_dir):
            # 依据文件夹名字遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root,sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                
                # 遍历该类别所有图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root,sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img,int(label)))
        return data_info
    


