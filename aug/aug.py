import random
import cv2
import numpy as np
import torch
from torchvision import transforms
from deepvac.aug.base_aug import AugBase
from deepvac.aug.factory import AugFactory
from deepvac.aug import Composer, PickOneComposer
from config import config

class ESPNetTrainComposer(PickOneComposer):
    def __init__(self, deepvac_config):
        self.config = deepvac_config
        super(ESPNetTrainComposer, self).__init__(deepvac_config)

        ac1 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksRandomCropResizeAug => ImageWithMasksHFlipAug@0.5 => ImageWithMasksToTensorAug', deepvac_config)
        ac2 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksRandomCropResizeAug => ImageWithMasksVFlipAug@0.5 => ImageWithMasksToTensorAug', deepvac_config)

        self.addAugFactory('ac1', ac1, 0.5)
        self.addAugFactory('ac2', ac2, 0.5)


class ESPNetValComposer(Composer):
    def __init__(self, deepvac_config):
        self.config = deepvac_config
        super(ESPNetValComposer, self).__init__(deepvac_config)

        ac1 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksScaleAug => ImageWithMasksToTensorAug', deepvac_config)

        self.addAugFactory('ac1', ac1, 1)