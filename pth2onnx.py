# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import argparse

from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch
import onnx
import onnxsim
import torch.nn as nn
from models import mybinarynet,lenet,resnet,vgg,densenet,googlenet,efficientnet,mobilenet,mobilenetv2
# parser = argparse.ArgumentParser(description='pytorch2onnx')
# parser.add_argument('--torch_model',
#                     default="/home/vinki/project/binary_classification/checkpoint/Loss_0.0575Acc:_0.9847.pth")
# parser.add_argument('--onnx_model', default="/home/vinki/project/binary_classification/checkpoint/eye_best.onnx")
# parser.add_argument('--onnx_model_sim',
#                     help='Output ONNX model',
#                     default="./output/pfld-sim.onnx")
# args = parser.parse_args()

##-------------------------------resnet18------------------------------#
# torch_model = "/home/vinki/project/binary_classification/checkpoint/Loss_0.0575Acc:_0.9847.pth"
# onnx_model = "/home/vinki/project/binary_classification/checkpoint/eye_best.onnx"
# print("=====> load pytorch checkpoint...")
# checkpoint = torch.load(torch_model, map_location=torch.device('cpu'))
# res18 = models.resnet18()
# num_ftrs = res18.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# res18.fc = nn.Linear(num_ftrs, 2)
# res18.load_state_dict(checkpoint)
# # print("res18:", res18)

##-------------------------------densenet121------------------------------#
BASE_PATH = '/home/vinki/project/binary_classification/checkpoint/life_jacket/densenet3'
torch_model =  BASE_PATH + "/Epoch_076_Acc:_99.6070.pth" 
onnx_model =  BASE_PATH + "/life_jacket_dense_20220412_32.onnx"
print("=====> load pytorch checkpoint...")




net = densenet.DenseNet121()


checkpoint = torch.load(torch_model, map_location=torch.device('cpu'))

## 如果多GPU训练时，pth文件中的键值有“module.”，需要删除
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in checkpoint.items():
#     name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
#     new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。 
# # load params
# net.load_state_dict(new_state_dict)

net.load_state_dict(checkpoint)
print("=====> convert pytorch model to onnx...")
dummy_input = Variable(torch.randn(32, 3, 32, 32))
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(net,
                  dummy_input,
                  onnx_model,
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)

print("====> check onnx model...")

model = onnx.load(onnx_model)
onnx.checker.check_model(model)

print("====> Simplifying...")
model_opt = onnxsim.simplify(onnx_model)

model_opt, check = onnxsim.simplify(onnx_model)
print("onnx model simplify Ok!")
# '''