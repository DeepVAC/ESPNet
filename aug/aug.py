import random
import cv2
import numpy as np
import torch
from torchvision import transforms
from deepvac.aug.base_aug import AugBase
from deepvac.aug.factory import AugFactory
from deepvac.aug import Composer, PickOneComposer
from config import config

class ESPNetTrainComposer(Composer):
    def __init__(self, deepvac_config):
        super(ESPNetTrainComposer, self).__init__(deepvac_config)
        ac1 = AugFactory('SpeckleAug@0.1 => GaussianAug@0.1 => HorlineAug@0.1 => VerlineAug@0.1 => LRmotionAug@0.1 => UDmotionAug@0.1 \
            => NoisyAug@0.1 => DarkAug@0.1 => ColorJitterAug@0.15 => BrightnessJitterAug@0.15 => ContrastJitterAug@0.15 => \
            ImageWithMasksRandomRotateAug@0.6 => ImageWithMasksNormalizeAug => ImageWithMasksCenterCropAug => ImageWithMasksScaleAug => \
            ImageWithMasksHFlipAug@0.5 => ImageWithMasksToTensorAug', deepvac_config)
        self.addAugFactory('ac1', ac1)

class ESPNetValComposer(Composer):
    def __init__(self, deepvac_config):
        super(ESPNetValComposer, self).__init__(deepvac_config)
        ac1 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksScaleAug => ImageWithMasksToTensorAug', deepvac_config)
        self.addAugFactory('ac1', ac1)