from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
from PIL import Image
import shutil

from my_dataset import EyeDataset,BinaryClassifyDataset
from models import mybinarynet,lenet,resnet,vgg,densenet,googlenet,efficientnet,mobilenet,mobilenetv2




# label_name = {'bg_flag':0,'flag':1}
# base_dir = '/media/NAS_share/vinki/MarautecData/national_flag/'
# base_dir = '/media/NAS_share/vinki/MarautecData/test/'
# label_name = {'bg_life_jacket':0,'life_jacket':1}
# base_dir = '/media/NAS_share/vinki/IsmsData/life_jacket/'
# save_dir = '/media/NAS_share/vinki/MarautecData/national_flag/wrong'
# label_name = {'bg_uniform':0,'uniform':1}
# base_dir = '/media/NAS_share/vinki/IsmsData/uniform/'

label_name = {'bg':0,'uniform':1,'lifejacket':2,'reflectvest':3}
base_dir = '/media/NAS_share/vinki/IsmsData/multi_classification'

save_dir = base_dir + '/error'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



src_data_dir = base_dir + '/marked'


class_names = []
for e_key in label_name.keys():
    class_names.append(e_key)
print(class_names)


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)])
test_transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1)),
    transforms.Normalize(norm_mean, norm_std)])

test_data = BinaryClassifyDataset(src_data_dir,label_name,test_transform)
total = test_data.__len__()
print('Total images:{}'.format(total))




# DenseNet121
model_path = '/home/vinki/project/multi_classification/checkpoint/densenet5/Epoch_059_Acc:_96.8971.pth'
model = densenet.DenseNet121()

# BinaryNet
# model_path = '/home/vinki/project/binary_classification/checkpoint/national_flag/mybinarynet/Epoch_016_Acc:_99.8591.pth'
# model = mybinarynet.BinaryNet()

# ResNet18
# model_path = '/home/vinki/project/binary_classification/checkpoint/eye/resnet18/Acc:_99.9221.pth'
# model = resnet.ResNet18()



# LeNet
# model_path = '/home/vinki/project/binary_classification/checkpoint/eye/lenet/Epoch_011_Acc:_99.7403.pth'
# model = lenet.LeNet()

# MoblieNetV2
# model_path = '/home/vinki/project/binary_classification/checkpoint/national_flag/mobilenetv2/Epoch_023_Acc:_100.0000.pth'
# model = mobilenetv2.MobileNetV2()

# MoblieNet
# model_path = '/home/vinki/project/binary_classification/checkpoint/eye/mobilenet/Epoch_060_Acc:_99.9221.pth'
# model = mobilenet.MobileNet()

# # GoogleNet
# model_path = '/home/vinki/project/binary_classification/checkpoint/eye/googlenet/GoogleNet_Acc_99.8961.pth'
# model = googlenet.GoogLeNet()



checkpoint = torch.load(model_path)

model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()


img_label = test_data.get_img_info()

right_cnt = 0
for path_img,label in img_label:
    # print(path_img,label)
    
    img = Image.open(path_img)
    # print(img.size)
    if img.mode =='L': #灰度图
        inputs = test_transform_gray(img).unsqueeze(0)
    else:
        inputs = test_transform(img).unsqueeze(0)
    
    inputs = inputs.to(device)
    outputs = model(inputs)
    
    m = nn.Sigmoid()
    prob = m(outputs)
    # print(prob)
    
    
    _, preds = torch.max(outputs, 1)
    # print('predicted: {}  {}'.format(preds[0],class_names[preds[0]]))
    # print(path_img,label)
    # print('------**********----\n')


    if preds[0]==label:
        right_cnt += 1
    else:
        print(path_img,class_names[label])
        print('predicted: {}  {}'.format(preds[0],class_names[preds[0]]))
        
        
        
        
        new_img = os.path.join(save_dir, path_img.split('/')[-1])
        print(new_img)
        os.chmod(save_dir,777)
        
        shutil.copy(path_img,save_dir)

    # print('predicted: {}  {}'.format(preds[0],class_names[preds[0]]))
    # '''
    # # inp = inputs.cpu().data[0]
    
    # # inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # # inp = std * inp + mean
    
    # img = inputs[0].to('cpu',torch.float32) # 消去batch那一维
    # img = img.numpy() # # FloatTensor转为ndarray
    # img = np.transpose(img, (1,2,0)) # 把channel那一维放到最后 C,H,W-->H,W,C
    # img = std * img + mean
    # img = img.astype(np.float32)
    
    # img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
    # # print(img.shape)
    
    # cv2.imshow("img",img)
    
    # key = cv2.waitKey(0)
    # if key == 32:
    #     continue
    # elif key == 27:
    #     break
    # '''
print(right_cnt,right_cnt*1.0/(total*1.0))
    
print('ok')
