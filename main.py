import os
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn

from matplotlib import pyplot as plt
from models import mybinarynet,lenet,resnet,vgg,densenet,googlenet,efficientnet,mobilenet,mobilenetv2

from my_dataset import EyeDataset,BinaryClassifyDataset,MultiClassifyDataset
from utils import prepare_classes_data,progress_bar

def train(train_loader,model,criterion,optimizer,epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx,(inputs, targets) in enumerate(train_loader):
        inputs,targets = inputs.to(device),targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
           
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Train---Loss: %.5f | Acc: %.4f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def validate(val_loader,model,criterion):
    model.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Valid---Loss: %.5f | Acc: %.4f%% (%d/%d)'
    
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    if total == 0:
        return 0;
    acc = 100.*correct/total
    
    return acc

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__=='__main__':
    ######## 参数设置
    MAX_EPOCH = 100
    BATCH_SIZE = 128
    LR = 0.01
    best_acc = 0.0
    
    model_save_dir = '/home/vinki/project/multi_classification/checkpoint/densenet6'
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # # # uniform lifejacket reflectvest
    label_name = {'bg':0,'uniform':1,'lifejacket':2,'reflectvest':3}
    base_dir = '/media/NAS_share/vinki/IsmsData/multi_classification'
    
    
    src_data_dir = base_dir + '/marked'
    data_dir = base_dir + '/training'
    
    #####################preprocess data：split to 'train' and ' val' #########
    prepare_classes_data(src_data_dir,data_dir)
    
    train_dir = data_dir +'/train'
    valid_dir =  data_dir + '/val'

    # =============== step 1/5 加载数据===================

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])   # Resize的功能是缩放，RandomCrop的功能是裁剪，ToTensor的功能是把图片变为张量

    valid_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    
    
    train_data = MultiClassifyDataset(train_dir,label_name,train_transform)
    valid_data = MultiClassifyDataset(valid_dir,label_name,valid_transform)


    train_loader = DataLoader(train_data,BATCH_SIZE,shuffle=True)
    valid_loader = DataLoader(valid_data,BATCH_SIZE)
    # print(train_data.__len__())


    # =============== step 2/5 创建模型===================
    # net = mybinarynet.BinaryNet()
    # net = lenet.LeNet()
    # net = resnet.ResNet18() # 18,34,50,101,154
    # net = mobilenetv2.MobileNetV2()
    # net = googlenet.GoogLeNet()

    net = densenet.DenseNet121() # 121 169 201 161
    # net = vgg.VGG('VGG11') # VGG11 VGG13 VGG16 VGG19
    # net = efficientnet.EfficientNetB0()
    # net = mobilenet.MobileNet()
 


    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    # 多GPU，显卡型号不同时，会出现性能不均衡，速度更慢
    # if torch.cuda.device_count() >= 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")

    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True
    net = net.to(device)

    # =============== step 3/5 损失函数===================
    criterion = nn.CrossEntropyLoss()

    # =============== step 4/5 优化器===================
    optimizer = optim.SGD(net.parameters(),lr = LR, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

    # =============== step 5/5 训练===================
    for epoch in range(MAX_EPOCH):
        train(train_loader,net,criterion,optimizer,epoch)      
        # evaluate on validation set
        acc = validate(valid_loader, net, criterion)
        exp_lr_scheduler.step()
        # remember best Accuracy and save checkpoint
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(net.state_dict())
            torch.save(best_model_wts, model_save_dir + '/Epoch_{:0>3d}_Acc:_{:.4f}.pth'.format(epoch,best_acc)) 
        