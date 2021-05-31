import random
import cv2
import numpy as np
import torch
from torchvision import transforms
from deepvac.aug.base_aug import AugBase
from deepvac.aug.factory import AugFactory
from deepvac.aug import Composer, PickOneComposer, CvAugBase, CvAugBase2
from deepvac.utils import addUserConfig
from config import config

class BorderTargetAug(CvAugBase2):
    def forward(self, img):
        image, label = img
        kernel = np.ones((15, 15), np.uint8)

        label_detail = np.zeros(label.shape, np.uint8)

        hat = np.zeros(label.shape, np.uint8)
        hat[np.where(label==1)] = 1
        hat_dilate = cv2.dilate(hat, kernel, iterations=1)
        hat_erode = cv2.erode(hat, kernel, iterations=1)
        hat_detail = hat_dilate - hat_erode
        label_detail[np.where(hat_detail==1)] = 1

        upclothes = np.zeros(label.shape, np.uint8)
        upclothes[np.where(label==2)] = 1
        upclothes_dilate = cv2.dilate(upclothes, kernel, iterations=1)
        upclothes_erode = cv2.erode(upclothes, kernel, iterations=1)
        upclothes_detail = upclothes_dilate - upclothes_erode
        label_detail[np.where(upclothes_detail==1)] = 2

        downclothes = np.zeros(label.shape, np.uint8)
        downclothes[np.where(label==3)] = 1
        downclothes_dilate = cv2.dilate(downclothes, kernel, iterations=1)
        downclothes_erode = cv2.erode(downclothes, kernel, iterations=1)
        downclothes_detail = downclothes_dilate - downclothes_erode
        label_detail[np.where(downclothes_detail==1)] = 3
        
        return [image, [label, label_detail]]

class ImageWithTwoMasksToTensorAug(CvAugBase2):
    def auditConfig(self):
        self.config.scale = self.addUserConfig('scale', self.config.scale, 1)
        self.config.force_div255 = self.addUserConfig('force_div255', self.config.force_div255, True)

    def forward(self, imgs):
        img, [label, label_detail] = imgs

        if self.config.scale != 1:
            h, w = label.shape[:2]
            img = cv2.resize(img, (int(w), int(h)))
            label = cv2.resize(label, (int(w/self.config.scale), int(h/self.config.scale)), interpolation=cv2.INTER_NEAREST)
            label_detail = cv2.resize(label_detail, (int(w/self.config.scale), int(h/self.config.scale)), interpolation=cv2.INTER_NEAREST)

        default_float_dtype = torch.get_default_dtype()
        image_tensor = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
        # backward compatibility
        if isinstance(image_tensor, torch.ByteTensor) or self.config.force_div255:
            image_tensor = image_tensor.to(dtype=default_float_dtype).div(255)
        label_tensor = torch.LongTensor(np.array(label, dtype=np.int)) #torch.from_numpy(label)
        label_detail_tensor = torch.LongTensor(np.array(label_detail, dtype=np.int))

        return [image_tensor, [label_tensor, label_detail_tensor]]

class BorderTargetAugFactory(AugFactory):
    def initProducts(self):
        super(BorderTargetAugFactory, self).initProducts()
        aug_name = 'BorderTargetAug'
        self.addProduct(aug_name, eval(aug_name))
        aug_name = 'ImageWithTwoMasksToTensorAug'
        self.addProduct(aug_name, eval(aug_name))

class ESPNetTrainComposer(Composer):
    def __init__(self, deepvac_config):
        super(ESPNetTrainComposer, self).__init__(deepvac_config)
        ac1 = BorderTargetAugFactory('SpeckleAug@0.1 => GaussianAug@0.1 => HorlineAug@0.1 => VerlineAug@0.1 => LRmotionAug@0.1 => UDmotionAug@0.1 \
            => NoisyAug@0.1 => DarkAug@0.1 => ColorJitterAug@0.25 => BrightnessJitterAug@0.25 => ContrastJitterAug@0.25 => \
            ImageWithMasksRandomRotateAug@0.6 =>  ImageWithMasksCenterCropAug => ImageWithMasksScaleAug => ImageWithMasksHFlipAug@0.5 => \
            ImageWithMasksNormalizeAug => BorderTargetAug => ImageWithTwoMasksToTensorAug', deepvac_config)
        self.addAugFactory('ac1', ac1)

class ESPNetValComposer(Composer):
    def __init__(self, deepvac_config):
        super(ESPNetValComposer, self).__init__(deepvac_config)
        ac1 = AugFactory('ImageWithMasksNormalizeAug => ImageWithMasksScaleAug => ImageWithMasksToTensorAug', deepvac_config)
        self.addAugFactory('ac1', ac1)
