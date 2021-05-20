import numpy as np
import torch
import random
import cv2

from torchvision import transforms

from deepvac.aug.base_aug import AugBase
from deepvac.aug.factory import AugFactory
from deepvac.aug import Composer, PickOneComposer
from config import config
    
class ESPNetMainComposer(PickOneComposer):
    def __init__(self, deepvac_config):
        self.config = deepvac_config
        super(ESPNetMainComposer, self).__init__(deepvac_config)

        ac1 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksRandomCropResizeAug => ImageWithMasksHFlipAug@0.5 => ImageWithMasksToTensorAug', deepvac_config)
        ac2 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksRandomCropResizeAug => ImageWithMasksVFlipAug@0.5 => ImageWithMasksToTensorAug', deepvac_config)

        self.addAugFactory('ac1', ac1, 0.5)
        self.addAugFactory('ac1', ac2, 0.5)

class ESPNetScale1Composer(PickOneComposer):
    def __init__(self, deepvac_config):
        self.config = deepvac_config
        super(ESPNetScale1Composer, self).__init__(deepvac_config)

        ac1 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksRandomCropResizeAug => ImageWithMasksHFlipAug@0.5 => ImageWithMasksToTensorAug', deepvac_config)
        ac2 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksRandomCropResizeAug => ImageWithMasksVFlipAug@0.5 => ImageWithMasksToTensorAug', deepvac_config)

        self.addAugFactory('ac1', ac1, 0.5)
        self.addAugFactory('ac1', ac2, 0.5)

class ESPNetScale2Composer(PickOneComposer):
    def __init__(self, deepvac_config):
        self.config = deepvac_config
        super(ESPNetScale2Composer, self).__init__(deepvac_config)

        ac1 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksRandomCropResizeAug => ImageWithMasksHFlipAug@0.5 => ImageWithMasksToTensorAug', deepvac_config)
        ac2 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksRandomCropResizeAug => ImageWithMasksVFlipAug@0.5 => ImageWithMasksToTensorAug', deepvac_config)

        self.addAugFactory('ac1', ac1, 0.5)
        self.addAugFactory('ac1', ac2, 0.5)

class ESPNetScale3Composer(PickOneComposer):
    def __init__(self, deepvac_config):
        self.config = deepvac_config
        super(ESPNetScale3Composer, self).__init__(deepvac_config)

        ac1 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksRandomCropResizeAug => ImageWithMasksHFlipAug@0.5 => ImageWithMasksToTensorAug', deepvac_config)
        ac2 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksRandomCropResizeAug => ImageWithMasksVFlipAug@0.5 => ImageWithMasksToTensorAug', deepvac_config)

        self.addAugFactory('ac1', ac1, 0.5)
        self.addAugFactory('ac1', ac2, 0.5)

class ESPNetScale4Composer(PickOneComposer):
    def __init__(self, deepvac_config):
        self.config = deepvac_config
        super(ESPNetScale4Composer, self).__init__(deepvac_config)

        ac1 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksRandomCropResizeAug => ImageWithMasksHFlipAug@0.5 => ImageWithMasksToTensorAug', deepvac_config)
        ac2 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksRandomCropResizeAug => ImageWithMasksVFlipAug@0.5 => ImageWithMasksToTensorAug', deepvac_config)

        self.addAugFactory('ac1', ac1, 0.5)
        self.addAugFactory('ac1', ac2, 0.5)

class ESPNetValComposer(Composer):
    def __init__(self, deepvac_config):
        self.config = deepvac_config
        super(ESPNetValComposer, self).__init__(deepvac_config)

        ac1 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksScaleAug => ImageWithMasksToTensorAug', deepvac_config)

        self.addAugFactory('ac1', ac1, 1)