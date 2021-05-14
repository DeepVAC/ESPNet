import math
import torch
import pickle
import os

import torch.optim as optim
from torchvision import transforms as trans

from deepvac import config, AttrDict
from deepvac.loss import MultiBoxLoss
from deepvac import is_ddp

import aug.aug as myTransforms
from data.dataloader import myDataLoader, getData
from modules.model import EESPNet_Seg

## ------------------ common ------------------
config.train.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.train.output_dir = 'output'
config.train.log_every = 100
config.train.disable_git = False

config.train.model_reinterpret_cast = True
config.train.cast_state_dict_strict = False
# load script and quantize model path
#config.train.jit_model_path = "<your-script-or-quantize-model-path>"

# # # training # # #
## -------------------- training ------------------
config.train.epoch_num = 250 # 100 for resnet, 250 for others
config.train.save_num = 1
config.train.cls_num = 4
config.train.shuffle = True
config.train.batch_size = 8
config.train.model_path = "/opt/public/airlock/lihang/github/ESPNetv2/imagenet/pretrained_weights/espnetv2_s_2.0.pth"

## -------------------- tensorboard ------------------
#config.train.tensorboard_port = "6007"
#config.train.tensorboard_ip = None


## -------------------- script and quantize ------------------
#config.train.trace_model_dir = "./trace.pt"
#config.train.static_quantize_dir = "./script.sq"
#config.train.dynamic_quantize_dir = "./quantize.sq"


# check if processed data file exists or not
config.train.cached_data_file = 'clothes.p'
data = getData(config.train.cached_data_file, config.train.cls_num)

## -------------------- net and criterion ------------------
config.train.net = EESPNet_Seg(4, s=2)
weight = torch.from_numpy(data['classWeights']).to(config.train.device)
config.train.criterion = torch.nn.CrossEntropyLoss(weight)

## -------------------- optimizer and scheduler ------------------
config.train.optimizer = torch.optim.Adam(config.train.net.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

lambda_lr = lambda epoch: round ((1 - epoch/config.train.epoch_num) ** 0.99, 8)
config.train.scheduler = optim.lr_scheduler.LambdaLR(config.train.optimizer, lr_lambda=lambda_lr)

## -------------------- loader ------------------
config.train.num_workers = 4

#compose the data with transforms
trainDataset_main = myTransforms.Compose([
    myTransforms.Normalize(mean=data['mean'], std=data['std']),
    myTransforms.RandomCropResize(size = (int(1.0*384), int(1.0*384))),
    myTransforms.RandomFlip(),
    myTransforms.ToTensor(1),
    #
])

trainDataset_scale1 = myTransforms.Compose([
    myTransforms.Normalize(mean=data['mean'], std=data['std']),
    myTransforms.RandomCropResize(size = (int(1.25*384), int(1.25*384))),
    myTransforms.RandomFlip(),
    myTransforms.ToTensor(1),
        #
])

trainDataset_scale2 = myTransforms.Compose([
    myTransforms.Normalize(mean=data['mean'], std=data['std']),
    myTransforms.RandomCropResize(size = (int(1.5*384), int(1.5*384))), # 1536, 768
    myTransforms.RandomFlip(),
    myTransforms.ToTensor(1),
        #
])

trainDataset_scale3 = myTransforms.Compose([
    myTransforms.Normalize(mean=data['mean'], std=data['std']),
    myTransforms.RandomCropResize(size = (int(1.75*384), int(1.75*384))),
    myTransforms.RandomFlip(),
    myTransforms.ToTensor(1),
])

trainDataset_scale4 = myTransforms.Compose([
    myTransforms.Normalize(mean=data['mean'], std=data['std']),
    myTransforms.RandomCropResize(size = (int(2.0*384), int(2.0*384))),
    myTransforms.RandomFlip(),
    myTransforms.ToTensor(1),
])


valDataset = myTransforms.Compose([
    myTransforms.Normalize(mean=data['mean'], std=data['std']),
    myTransforms.Scale(384, 384),
    myTransforms.ToTensor(1),
])


train_loader = torch.utils.data.DataLoader(
    myDataLoader(data['trainIm'], data['trainAnnot'], transform=trainDataset_main),
    batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, pin_memory=True)

train_loader_scale1 = torch.utils.data.DataLoader(
    myDataLoader(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale1),
    batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, pin_memory=True)

train_loader_scale2 = torch.utils.data.DataLoader(
    myDataLoader(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale2),
    batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, pin_memory=True)

train_loader_scale3 = torch.utils.data.DataLoader(
    myDataLoader(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale3),
    batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, pin_memory=True)

train_loader_scale4 = torch.utils.data.DataLoader(
    myDataLoader(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale4),
    batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, pin_memory=True)

config.train.train_loader_list = [train_loader_scale1, train_loader_scale2, train_loader_scale4, train_loader_scale3, train_loader]

config.train.train_dataset = myDataLoader(data['trainIm'], data['trainAnnot'], transform=trainDataset_main)
config.train.train_loader = torch.utils.data.DataLoader(
    config.train.train_dataset,
    batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, pin_memory=True)
config.train.val_dataset = myDataLoader(data['valIm'], data['valAnnot'], transform=valDataset)
config.train.val_loader = torch.utils.data.DataLoader(
    config.train.val_dataset,
    batch_size=4, shuffle=False, num_workers=config.train.num_workers, pin_memory=True)

config.train.test_dataset = myDataLoader(data['valIm'], data['valAnnot'], transform=valDataset)
config.train.test_loader = torch.utils.data.DataLoader(
    config.train.val_dataset,
    batch_size=4, shuffle=False, num_workers=config.train.num_workers, pin_memory=True)
## ------------------ ddp --------------------
# config.dist_url = 'tcp://localhost:27030'
# config.world_size = 2

