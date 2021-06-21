import os
import math
import pickle
import torch
from torch import optim
from torchvision import transforms as trans
from deepvac import config, AttrDict, new, interpret, fork
from deepvac import is_ddp
from deepvac.datasets import FileLineCvSegDataset, OsWalkBaseDataset
from deepvac.aug import MultiInputCompose
from deepvac.backbones import makeDivisible
from data.dataloader import FileLineCvSegWithMetaInfoDataset
from modules.model import EESPNet_Seg
from aug.aug import *

config = new('ESPNetTrain')
## -------------------- global ------------------
config.train_txt = './data/train.txt'
config.val_txt = './data/val.txt'
config.sample_path_prefix = './data'
# config.test_sample_path = "your test images dir"
config.delimiter = ','
config.cls_num = 4
config.input_w = 384
config.input_h = 384

config.pin_memory = True if torch.cuda.is_available() else False
torch.backends.cudnn.benchmark=True
config.batch_size = 16

## -------------------- datasets & aug ------------------
config.datasets.FileLineCvSegWithMetaInfoDataset = AttrDict()
config.datasets.FileLineCvSegWithMetaInfoDataset.cached_data_file = 'data/clothes.p'
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
config.aug.ImageWithMasksScaleAug.w = config.input_w
config.aug.ImageWithMasksScaleAug.h = config.input_h

## ------------------ common ------------------
config.core.ESPNetTrain.cls_num = config.cls_num
config.core.ESPNetTrain.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.ESPNetTrain.output_dir = 'output'
config.core.ESPNetTrain.show_output_dir = 'output/show'
config.core.ESPNetTrain.log_every = 10
config.core.ESPNetTrain.disable_git = False
config.core.ESPNetTrain.model_reinterpret_cast = True
config.core.ESPNetTrain.cast_state_dict_strict = False
# load script and quantize model path
#config.core.ESPNetTrain.jit_model_path = "<your-script-or-quantize-model-path>"

## -------------------- training ------------------
config.core.ESPNetTrain.epoch_num = 100
config.core.ESPNetTrain.save_num = 1
config.core.ESPNetTrain.model_path = "/opt/public/pretrain/ESPNetv2/imagenet/espnetv2_s_2.0.pth"

## -------------------- tensorboard ------------------
# config.core.ESPNetTrain.tensorboard_port = "6007"
# config.core.ESPNetTrain.tensorboard_ip = None

## -------------------- script and quantize ------------------
# config.cast.TraceCast = AttrDict()
# config.cast.TraceCast.model_dir = "./script.pt"
# config.cast.TraceCast.static_quantize_dir = "./script.sq"
# config.cast.TraceCast.dynamic_quantize_dir = "./quantize.sq"

## -------------------- net and criterion ------------------
config.core.ESPNetTrain.net = EESPNet_Seg(config.cls_num)
weight = torch.from_numpy(config.data['classWeights']).to(config.core.ESPNetTrain.device)
config.core.ESPNetTrain.criterion = torch.nn.CrossEntropyLoss(weight)

## -------------------- optimizer and scheduler ------------------
config.core.ESPNetTrain.optimizer = torch.optim.Adam(config.core.ESPNetTrain.net.parameters(), 3e-4, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
lambda_lr = lambda epoch: round ((1 - epoch/config.ESPNetTrain.core.epoch_num) ** 0.9, 8)
config.core.ESPNetTrain.scheduler = optim.lr_scheduler.LambdaLR(config.core.ESPNetTrain.optimizer, lr_lambda=lambda_lr)

## -------------------- loader ------------------
config.num_workers = 3

#just for fool deepvac
config.datasets.FileLineCvSegDataset.composer = ESPNetTrainComposer(config)
config.core.ESPNetTrain.train_dataset = FileLineCvSegDataset(config, config.train_txt, config.delimiter, config.sample_path_prefix)
config.core.ESPNetTrain.train_loader = torch.utils.data.DataLoader(config.core.ESPNetTrain.train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
config.core.ESPNetTrain.train_loader.is_last_loader = False
#fool end

config.datasets.FileLineCvSegDataset.composer = ESPNetValComposer(config)
config.core.ESPNetTrain.val_dataset = FileLineCvSegDataset(config, config.val_txt, config.delimiter, config.sample_path_prefix)
config.core.ESPNetTrain.val_loader = torch.utils.data.DataLoader(config.core.ESPNetTrain.val_dataset,batch_size=8, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)

config.datasets.OsWalkBaseDataset = AttrDict()
config.datasets.OsWalkBaseDataset.transform = trans.Compose([trans.ToPILImage(),
    trans.Resize((config.input_h, config.input_w)),
    trans.ToTensor(),
    trans.Normalize(mean=(config.data["mean"] / 255.), std=config.data["std"])])

## ------------------ ddp --------------------
# config.dist_url = 'tcp://localhost:27030'
# config.world_size = 2

config.datasets.FileLineCvSegDataset.composer = ESPNetTrainComposer(config)
last_train_loader = torch.utils.data.DataLoader(FileLineCvSegDataset(config, config.train_txt, config.delimiter, config.sample_path_prefix),
    batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
last_train_loader.is_last_loader = True

scale1_config = fork(config)
scale1_config.aug.ImageWithMasksScaleAug.w = makeDivisible(config.input_w * 1.5, 32)
scale1_config.aug.ImageWithMasksScaleAug.h = makeDivisible(config.input_h * 1.5, 32)
scale1_config.datasets.FileLineCvSegDataset.composer = ESPNetTrainComposer(scale1_config)
scale1_train_loader = torch.utils.data.DataLoader(FileLineCvSegDataset(scale1_config, config.train_txt, config.delimiter, config.sample_path_prefix),
    batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
scale1_train_loader.is_last_loader = False

scale2_config = fork(config)
scale2_config.aug.ImageWithMasksScaleAug.w = makeDivisible(config.input_w * 1.25, 32)
scale2_config.aug.ImageWithMasksScaleAug.h = makeDivisible(config.input_h * 1.25, 32)
scale2_config.datasets.FileLineCvSegDataset.composer = ESPNetTrainComposer(scale2_config)
scale2_train_loader = torch.utils.data.DataLoader(FileLineCvSegDataset(scale2_config, config.train_txt, config.delimiter, config.sample_path_prefix),
    batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
scale2_train_loader.is_last_loader = False

scale3_config = fork(config)
scale3_config.aug.ImageWithMasksScaleAug.w = makeDivisible(config.input_w * 0.75, 32)
scale3_config.aug.ImageWithMasksScaleAug.h = makeDivisible(config.input_h * 0.75, 32)
scale3_config.datasets.FileLineCvSegDataset.composer = ESPNetTrainComposer(scale3_config)
scale3_train_loader = torch.utils.data.DataLoader(FileLineCvSegDataset(scale3_config, config.train_txt, config.delimiter, config.sample_path_prefix),
    batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
scale3_train_loader.is_last_loader = False

scale4_config = fork(config)
scale4_config.aug.ImageWithMasksScaleAug.w = makeDivisible(config.input_w * 0.5, 32)
scale4_config.aug.ImageWithMasksScaleAug.h = makeDivisible(config.input_h * 0.5, 32)

scale4_config.datasets.FileLineCvSegDataset.composer = ESPNetTrainComposer(scale4_config)
scale4_train_loader = torch.utils.data.DataLoader(FileLineCvSegDataset(scale4_config, config.train_txt, config.delimiter, config.sample_path_prefix),
    batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
scale4_train_loader.is_last_loader = False
# multi loader
config.core.ESPNetTrain.train_loader_list = [scale1_train_loader, scale2_train_loader, scale4_train_loader, scale3_train_loader, last_train_loader]
