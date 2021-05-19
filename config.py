import math
import torch
import pickle
import os

import torch.optim as optim
from torchvision import transforms as trans

from deepvac import config, AttrDict
from deepvac import is_ddp
from deepvac.datasets import FileLineCvSegDataset
from deepvac.aug import MultiInputCompose

from data.dataloader import FileLineCvSegWithMetaInfoDataset
from modules.model import EESPNet_Seg

## ------------------ common ------------------
config.core.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.output_dir = 'output'
config.core.log_every = 100
config.core.disable_git = False

config.core.model_reinterpret_cast = True
config.core.cast_state_dict_strict = False
# load script and quantize model path
#config.core.jit_model_path = "<your-script-or-quantize-model-path>"

# # # training # # #
## -------------------- training ------------------
config.core.epoch_num = 250 # 100 for resnet, 250 for others
config.core.save_num = 1
config.core.cls_num = 4
config.core.shuffle = True
config.core.batch_size = 16
config.core.model_path = "/opt/public/airlock/lihang/github/ESPNetv2/imagenet/pretrained_weights/espnetv2_s_2.0.pth"

## -------------------- tensorboard ------------------
#config.core.tensorboard_port = "6007"
#config.core.tensorboard_ip = None


## -------------------- script and quantize ------------------
#config.core.trace_model_dir = "./trace.pt"
#config.core.static_quantize_dir = "./script.sq"
#config.core.dynamic_quantize_dir = "./quantize.sq"

from aug.aug import *
# check if processed data file exists or not
config.datasets.FileLineCvSegWithMetaInfoDataset = AttrDict()
config.datasets.FileLineCvSegWithMetaInfoDataset.cached_data_file = 'clothes.p'
config.datasets.FileLineCvSegWithMetaInfoDataset.sample_path_prefix = './data'
config.datasets.FileLineCvSegWithMetaInfoDataset.fileline_path = './data/train.txt'
config.datasets.FileLineCvSegWithMetaInfoDataset.classes = config.core.cls_num
config.datasets.FileLineCvSegWithMetaInfoDataset.transform = None
config.datasets.FileLineCvSegWithMetaInfoDataset.norm_val = 1.10
load_data = FileLineCvSegWithMetaInfoDataset(config)
config.datasets.data = load_data()

## -------------------- net and criterion ------------------
config.core.net = EESPNet_Seg(4)
weight = torch.from_numpy(config.datasets.data['classWeights']).to(config.core.device)
config.core.criterion = torch.nn.CrossEntropyLoss(weight)

# config.core.teacher = AttrDict()
# config.core.teacher.net = EESPNet_Seg(4)
# config.core.teacher.criterion = torch.nn.CrossEntropyLoss(weight)
# config.core.teacher.optimizer = torch.optim.Adam(config.core.teacher.net.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

config.core.mean = config.datasets.data['mean']
config.core.std = config.datasets.data['std']
config.core.in_width = 384
config.core.in_height = 384
## -------------------- optimizer and scheduler ------------------
config.core.optimizer = torch.optim.Adam(config.core.net.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

lambda_lr = lambda epoch: round ((1 - epoch/config.core.epoch_num) ** 0.99, 8)
config.core.scheduler = optim.lr_scheduler.LambdaLR(config.core.optimizer, lr_lambda=lambda_lr)

## -------------------- loader ------------------
config.core.num_workers = 0

config.sample_path_prefix = './data'
config.fileline_path = './data/train.txt'

config.aug.ImageWithMasksNormalizeAug = AttrDict()
config.aug.ImageWithMasksNormalizeAug.mean = config.datasets.data['mean']
config.aug.ImageWithMasksNormalizeAug.std = config.datasets.data['std']

config.aug.ImageWithMasksToTensorAug = AttrDict()
config.aug.ImageWithMasksToTensorAug.scale = 1

config.aug.ImageWithMasksRandomCropResizeAug = AttrDict()
config.aug.ImageWithMasksRandomCropResizeAug.size = (int(1.0*384), int(1.0*384))

config.aug.ImageWithMasksScaleAug = AttrDict()
config.aug.ImageWithMasksScaleAug.w = 384
config.aug.ImageWithMasksScaleAug.h = 384
# config.FileLineCvSegDataset = AttrDict()

config.datasets.FileLineCvSegDataset = AttrDict()
config.datasets.FileLineCvSegDataset.composer = ESPNetMainComposer(config)
train_loader = torch.utils.data.DataLoader(
    FileLineCvSegDataset(config, config.fileline_path, ',', config.sample_path_prefix),
    batch_size=config.core.batch_size, shuffle=True, num_workers=config.core.num_workers, pin_memory=True)

config.aug.ImageWithMasksRandomCropResizeAug.size = (int(1.25*384), int(1.25*384))
config.datasets.FileLineCvSegDataset.composer = ESPNetScale1Composer(config)
train_loader_scale1 = torch.utils.data.DataLoader(
    FileLineCvSegDataset(config, config.fileline_path, ',', config.sample_path_prefix),
    batch_size=config.core.batch_size, shuffle=True, num_workers=config.core.num_workers, pin_memory=True)

config.aug.ImageWithMasksRandomCropResizeAug.size = (int(1.5*384), int(1.5*384))
config.datasets.FileLineCvSegDataset.composer = ESPNetScale2Composer(config)
train_loader_scale2 = torch.utils.data.DataLoader(
    FileLineCvSegDataset(config, config.fileline_path, ',', config.sample_path_prefix),
    batch_size=config.core.batch_size, shuffle=True, num_workers=config.core.num_workers, pin_memory=True)

config.aug.ImageWithMasksRandomCropResizeAug.size = (int(1.75*384), int(1.75*384))
config.datasets.FileLineCvSegDataset.composer = ESPNetScale3Composer(config)
train_loader_scale3 = torch.utils.data.DataLoader(
    FileLineCvSegDataset(config, config.fileline_path, ',', config.sample_path_prefix),
    batch_size=config.core.batch_size, shuffle=True, num_workers=config.core.num_workers, pin_memory=True)

config.aug.ImageWithMasksRandomCropResizeAug.size = (int(2.0*384), int(2.0*384))
config.datasets.FileLineCvSegDataset.composer = ESPNetScale4Composer(config)
train_loader_scale4 = torch.utils.data.DataLoader(
    FileLineCvSegDataset(config, config.fileline_path, ',', config.sample_path_prefix),
    batch_size=config.core.batch_size, shuffle=True, num_workers=config.core.num_workers, pin_memory=True)

config.core.train_loader_list = [train_loader_scale1, train_loader_scale2, train_loader_scale4, train_loader_scale3, train_loader]

config.aug.ImageWithMasksRandomCropResizeAug.size = (int(1.0*384), int(1.0*384))
config.datasets.FileLineCvSegDataset.composer = ESPNetMainComposer(config)
config.core.train_dataset = FileLineCvSegDataset(config, config.fileline_path, ',', config.sample_path_prefix)
config.core.train_loader = torch.utils.data.DataLoader(
    config.core.train_dataset,
    batch_size=config.core.batch_size, shuffle=True, num_workers=config.core.num_workers, pin_memory=True)

config.fileline_path = './data/val.txt'
config.datasets.FileLineCvSegDataset.composer = ESPNetValComposer(config)

config.core.val_dataset = FileLineCvSegDataset(config, config.fileline_path, ',', config.sample_path_prefix)
config.core.val_loader = torch.utils.data.DataLoader(
    config.core.val_dataset,
    batch_size=4, shuffle=False, num_workers=config.core.num_workers, pin_memory=True)

config.datasets.FileLineCvSegDataset.composer = ESPNetValComposer(config)

config.core.test_dataset = FileLineCvSegDataset(config, config.fileline_path, ',', config.sample_path_prefix)
config.core.test_loader = torch.utils.data.DataLoader(
    config.core.test_dataset,
    batch_size=4, shuffle=False, num_workers=config.core.num_workers, pin_memory=True)
## ------------------ ddp --------------------
# config.dist_url = 'tcp://localhost:27030'
# config.world_size = 2

