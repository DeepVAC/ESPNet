import os
import math
import pickle
import torch
import torch.optim as optim
from torchvision import transforms as trans
from deepvac import config, AttrDict, new, interpret, fork
from deepvac import is_ddp
from deepvac.datasets import FileLineCvSegDataset
from deepvac.aug import MultiInputCompose
from data.dataloader import FileLineCvSegWithMetaInfoDataset
from modules.model import EESPNet_Seg
from aug.aug import *

## -------------------- global ------------------
config.train_txt = './data/train.txt'
config.val_txt = './data/val.txt'
config.sample_path_prefix = './data'
config.delimiter = ','
config.cls_num = 4
## -------------------- datasets & aug ------------------
config.datasets.FileLineCvSegWithMetaInfoDataset = AttrDict()
config.datasets.FileLineCvSegWithMetaInfoDataset.cached_data_file = 'clothes.p'
config.datasets.FileLineCvSegWithMetaInfoDataset.classes = config.cls_num
config.datasets.FileLineCvSegWithMetaInfoDataset.norm_val = 1.10
config.data = FileLineCvSegWithMetaInfoDataset(config, config.train_txt, config.sample_path_prefix)()
config.datasets.FileLineCvSegDataset = AttrDict()

config.aug.ImageWithMasksNormalizeAug = AttrDict()
config.aug.ImageWithMasksNormalizeAug.mean = config.data['mean']
config.aug.ImageWithMasksNormalizeAug.std = config.data['std']

config.aug.ImageWithMasksToTensorAug = AttrDict()
config.aug.ImageWithMasksToTensorAug.scale = 1

config.aug.ImageWithMasksRandomRotateAug = AttrDict()
config.aug.ImageWithMasksRandomRotateAug.max_angle = 45
config.aug.ImageWithMasksRandomRotateAug.fill_color = True

config.aug.ImageWithMasksScaleAug = AttrDict()
config.aug.ImageWithMasksScaleAug.w = 384
config.aug.ImageWithMasksScaleAug.h = 384

## ------------------ common ------------------
config.core.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.output_dir = 'output'
config.core.log_every = 10
config.core.disable_git = False
config.core.model_reinterpret_cast = True
config.core.cast_state_dict_strict = False
# load script and quantize model path
#config.core.jit_model_path = "<your-script-or-quantize-model-path>"

## -------------------- training ------------------
config.core.epoch_num = 300
config.core.save_num = 1
config.core.shuffle = True
config.core.batch_size = 8
# config.core.model_path = "your model path"

## -------------------- tensorboard ------------------
# config.core.tensorboard_port = "6007"
# config.core.tensorboard_ip = None

## -------------------- script and quantize ------------------
# config.cast.TraceCast = AttrDict()
# config.cast.TraceCast.model_dir = "./script.pt"
# config.cast.TraceCast.static_quantize_dir = "./script.sq"
# config.cast.TraceCast.dynamic_quantize_dir = "./quantize.sq"

## -------------------- net and criterion ------------------
config.core.net = EESPNet_Seg(config.cls_num)
weight = torch.from_numpy(config.data['classWeights']).to(config.core.device)
config.core.criterion = torch.nn.CrossEntropyLoss(weight)

# config.core.teacher = AttrDict()
# config.core.teacher.net = EESPNet_Seg(config.cls_num)
# config.core.teacher.criterion = torch.nn.CrossEntropyLoss(weight)
# config.core.teacher.optimizer = torch.optim.Adam(config.core.teacher.net.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

config.core.mean = config.data['mean']
config.core.std = config.data['std']
## -------------------- optimizer and scheduler ------------------
config.core.optimizer = torch.optim.Adam(config.core.net.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

lambda_lr = lambda epoch: round ((1 - epoch/config.core.epoch_num) ** 0.9, 8)
config.core.scheduler = optim.lr_scheduler.LambdaLR(config.core.optimizer, lr_lambda=lambda_lr)

## -------------------- loader ------------------
config.core.num_workers = 2

#just for fool deepvac
config.datasets.FileLineCvSegDataset.composer = ESPNetTrainComposer(config)
config.core.train_dataset = FileLineCvSegDataset(config, config.train_txt, config.delimiter, config.sample_path_prefix)
config.core.train_loader = torch.utils.data.DataLoader(config.core.train_dataset, batch_size=config.core.batch_size, shuffle=True, num_workers=config.core.num_workers, pin_memory=True)
config.core.train_loader.is_last_loader = False
#fool end

config.datasets.FileLineCvSegDataset.composer = ESPNetValComposer(config)
config.core.val_dataset = FileLineCvSegDataset(config, config.val_txt, config.delimiter, config.sample_path_prefix)
config.core.val_loader = torch.utils.data.DataLoader(config.core.val_dataset,batch_size=4, shuffle=False, num_workers=config.core.num_workers, pin_memory=True)

config.datasets.FileLineCvSegDataset.composer = ESPNetValComposer(config)
config.core.test_dataset = FileLineCvSegDataset(config, config.val_txt, config.delimiter, config.sample_path_prefix)
config.core.test_loader = torch.utils.data.DataLoader(config.core.test_dataset, batch_size=4, shuffle=False, num_workers=config.core.num_workers, pin_memory=True)

## ------------------ ddp --------------------
# config.dist_url = 'tcp://localhost:27030'
# config.world_size = 2

config.datasets.FileLineCvSegDataset.composer = ESPNetTrainComposer(config)
last_train_loader = torch.utils.data.DataLoader(FileLineCvSegDataset(config, config.train_txt, config.delimiter, config.sample_path_prefix),
    batch_size=config.core.batch_size, shuffle=True, num_workers=config.core.num_workers, pin_memory=True)
last_train_loader.is_last_loader = True

scale1_config = fork(config)
scale1_config.aug.ImageWithMasksScaleAug.w = int(384 * 1.25)
scale1_config.aug.ImageWithMasksScaleAug.h = int(384 * 1.25)
scale1_config.datasets.FileLineCvSegDataset.composer = ESPNetTrainComposer(scale1_config)
scale1_train_loader = torch.utils.data.DataLoader(FileLineCvSegDataset(scale1_config, config.train_txt, config.delimiter, config.sample_path_prefix),
    batch_size=config.core.batch_size, shuffle=True, num_workers=config.core.num_workers, pin_memory=True)
scale1_train_loader.is_last_loader = False

scale2_config = fork(config)
scale2_config.aug.ImageWithMasksScaleAug.w = int(384 * 1.5)
scale2_config.aug.ImageWithMasksScaleAug.h = int(384 * 1.5)
scale2_config.datasets.FileLineCvSegDataset.composer = ESPNetTrainComposer(scale2_config)
scale2_train_loader = torch.utils.data.DataLoader(FileLineCvSegDataset(scale2_config, config.train_txt, config.delimiter, config.sample_path_prefix),
    batch_size=config.core.batch_size, shuffle=True, num_workers=config.core.num_workers, pin_memory=True)
scale2_train_loader.is_last_loader = False

scale3_config = fork(config)
scale3_config.aug.ImageWithMasksScaleAug.w = int(384 * 1.75)
scale3_config.aug.ImageWithMasksScaleAug.h = int(384 * 1.75)
scale3_config.datasets.FileLineCvSegDataset.composer = ESPNetTrainComposer(scale3_config)
scale3_train_loader = torch.utils.data.DataLoader(FileLineCvSegDataset(scale3_config, config.train_txt, config.delimiter, config.sample_path_prefix),
    batch_size=config.core.batch_size, shuffle=True, num_workers=config.core.num_workers, pin_memory=True)
scale3_train_loader.is_last_loader = False

scale4_config = fork(config)
scale4_config.aug.ImageWithMasksScaleAug.w = int(384 * 2.0)
scale4_config.aug.ImageWithMasksScaleAug.h = int(384 * 2.0)
scale4_config.datasets.FileLineCvSegDataset.composer = ESPNetTrainComposer(scale4_config)
scale4_train_loader = torch.utils.data.DataLoader(FileLineCvSegDataset(scale4_config, config.train_txt, config.delimiter, config.sample_path_prefix),
    batch_size=config.core.batch_size, shuffle=True, num_workers=config.core.num_workers, pin_memory=True)
scale4_train_loader.is_last_loader = False
# multi loader
config.core.train_loader_list = [scale1_train_loader, scale2_train_loader, scale4_train_loader, scale3_train_loader, last_train_loader]
