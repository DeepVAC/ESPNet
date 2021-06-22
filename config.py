# -*- coding:utf-8 -*-
import torch
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms as trans

from deepvac import config, AttrDict, new, interpret, fork, is_ddp
from deepvac.datasets import OsWalkDataset, OsWalkBaseDataset 

from modules.model import EESPNet_Seg

config = new('ESPNetTrain')
## -------------------- global ------------------
config.sample_path = "your image path"
config.cls_num = 4
config.mean = torch.Tensor([111.504555, 120.78698 , 128.28732]) / 255.
config.std  = torch.Tensor([63.338905, 62.412697, 63.73896])
config.classes_weight = torch.Tensor([1.5393538, 10.010507, 5.790331, 5.704213])

## -------------------- datasets & aug ------------------
config.datasets.OsWalkDataset = AttrDict()
config.datasets.OsWalkDataset.transform = trans.Compose([trans.ToPILImage(),
    trans.RandomHorizontalFlip(p=0.5),
    trans.Resize((384, 384)),
    trans.ToTensor(),
    trans.Normalize(mean=config.mean, std=config.std)])
config.datasets.OsWalkBaseDataset = AttrDict()
config.datasets.OsWalkBaseDataset.transform = trans.Compose([trans.ToPILImage(),
    trans.Resize((384, 384)),
    trans.ToTensor(),
    trans.Normalize(mean=config.mean, std=config.std)])

## ------------------ common ------------------
config.core.ESPNetTrain.cls_num = config.cls_num
config.core.ESPNetTrain.classes_weight = config.classes_weight
config.core.ESPNetTrain.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.ESPNetTrain.output_dir = 'output'
config.core.ESPNetTrain.show_output_dir = 'output/show'
config.core.ESPNetTrain.log_every = 10
config.core.ESPNetTrain.disable_git = False
config.core.ESPNetTrain.model_reinterpret_cast = True
config.core.ESPNetTrain.cast_state_dict_strict = False
config.core.ESPNetTrain.no_val = True
# load script and quantize model path
#config.core.ESPNetTrain.jit_model_path = "<your-script-or-quantize-model-path>"

## -------------------- training ------------------
config.core.ESPNetTrain.epoch_num = 10 
config.core.ESPNetTrain.save_num = 1
config.core.ESPNetTrain.model_path = "output/trained_espnet_2.0.pth"

## -------------------- tensorboard ------------------
# config.core.ESPNetTrain.tensorboard_port = "6007"
# config.core.ESPNetTrain.tensorboard_ip = None

## -------------------- script and quantize ------------------
# config.cast.TraceCast = AttrDict()
# config.cast.TraceCast.model_dir = "./script.pt"
# config.cast.TraceCast.static_quantize_dir = "./script.sq"
# config.cast.TraceCast.dynamic_quantize_dir = "./quantize.sq"

## -------------------- net and criterion ------------------
config.core.ESPNetTrain.criterion=torch.nn.CrossEntropyLoss()
config.core.ESPNetTrain.net = EESPNet_Seg(config.cls_num)

## -------------------- optimizer and scheduler ------------------
config.core.ESPNetTrain.optimizer = torch.optim.Adam(config.core.ESPNetTrain.net.parameters(), 1e-5, (0.9, 0.999))
config.core.ESPNetTrain.scheduler = lr_scheduler.StepLR(config.core.ESPNetTrain.optimizer, step_size=10, gamma=0.1, last_epoch=-1, verbose=False)

## -------------------- loader ------------------
config.num_workers = 6
config.core.ESPNetTrain.train_dataset = OsWalkDataset(config, config.sample_path)
config.core.ESPNetTrain.train_loader = torch.utils.data.DataLoader(config.core.ESPNetTrain.train_dataset, batch_size=1, num_workers=config.num_workers)


config.core.ESPNetTest = config.core.ESPNetTrain.clone()
config.core.ESPNetTest.model_reinterpret_cast = False
