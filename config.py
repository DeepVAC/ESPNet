# -*- coding:utf-8 -*-
import torch
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms as trans

from deepvac import config, AttrDict, new, interpret, fork, is_ddp
from deepvac.datasets import OsWalkDataset

from modules.model import EESPNet_Seg

## -------------------- global ------------------
config.sample_path = "your image path"
config.core.cls_num = 4
config.core.pallete = [[255, 255, 255],
                        [128, 64,  128],
                        [244, 35,  232],
                        [70,  70,  70],
                        [102, 102, 156],
                        [190, 153, 153],
                        [153, 153, 153],
                        [250, 170, 30],
                        [220, 220, 0],
                        [107, 142, 35],
                        [152, 251, 152],
                        [70,  130, 180],
                        [220, 20,  60],
                        [255, 0,   0],
                        [0,   0,   142],
                        [0,   0,   70],
                        [0,   60,  100],
                        [0,   80,  100],
                        [0,   0,   230],
                        [119, 11,  32]]
assert config.core.cls_num <= len(config.core.pallete), "seg cls num rathor than pallete length not support!"
config.core.mean = torch.Tensor([137.78314, 141.16818, 149.62434]) / 255.
config.core.std  = torch.Tensor([63.96097 , 64.199165, 64.6029])
config.core.classes_weight = torch.Tensor([1.5393538, 10.010507, 5.790331, 5.704213])

## -------------------- datasets & aug ------------------
config.datasets.OsWalkDataset = AttrDict()
config.datasets.OsWalkDataset.transform = trans.Compose([trans.ToPILImage(),
    trans.RandomHorizontalFlip(p=0.5),
    trans.Resize((384, 384)),
    trans.ToTensor(),
    trans.Normalize(mean=config.core.mean, std=config.core.std)])
config.datasets.OsWalkDataset2 = AttrDict()
config.datasets.OsWalkDataset2.transform = trans.Compose([trans.ToPILImage(),
    trans.Resize((384, 384)),
    trans.ToTensor(),
    trans.Normalize(mean=config.core.mean, std=config.core.std)])

## ------------------ common ------------------
config.core.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.output_dir = 'output'
config.core.show_output_dir = 'output/show'
config.core.log_every = 10
config.core.disable_git = False
config.core.model_reinterpret_cast = True
config.core.cast_state_dict_strict = False
config.core.no_val = True
# load script and quantize model path
#config.core.jit_model_path = "<your-script-or-quantize-model-path>"

## -------------------- training ------------------
config.core.epoch_num = 10 
config.core.save_num = 1
config.core.shuffle = True
config.core.batch_size = 1
config.core.model_path = "output/trained_espnet_2.0.pth"

## -------------------- tensorboard ------------------
# config.core.tensorboard_port = "6007"
# config.core.tensorboard_ip = None

## -------------------- script and quantize ------------------
# config.cast.TraceCast = AttrDict()
# config.cast.TraceCast.model_dir = "./script.pt"
# config.cast.TraceCast.static_quantize_dir = "./script.sq"
# config.cast.TraceCast.dynamic_quantize_dir = "./quantize.sq"

## -------------------- net and criterion ------------------
config.core.criterion=torch.nn.CrossEntropyLoss()
config.core.net = EESPNet_Seg(config.core.cls_num)

## -------------------- optimizer and scheduler ------------------
config.core.optimizer = torch.optim.Adam(config.core.net.parameters(), 1e-5, (0.9, 0.999))
config.core.scheduler = lr_scheduler.StepLR(config.core.optimizer, step_size=10, gamma=0.1, last_epoch=-1, verbose=False)

## -------------------- loader ------------------
config.core.num_workers = 6
config.core.train_dataset = OsWalkDataset(config, config.sample_path)
config.core.train_loader = torch.utils.data.DataLoader(config.core.train_dataset, batch_size=1, num_workers=config.core.num_workers)
# # # 
# config.core.val_dataset = OsWalkDataset(config, config.sample_path)
# config.core.val_loader = torch.utils.data.DataLoader(config.core.train_dataset, batch_size=1, num_workers=config.core.num_workers)

## ------------------ ddp --------------------
# config.dist_url = 'tcp://localhost:27030'
# config.world_size = 2
