import sys
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import deepvac
from deepvac import LOG, Deepvac

class ESPNetTest(Deepvac):
    def __init__(self, deepvac_config):
        super(ESPNetTest, self).__init__(deepvac_config)

    def post_process(self):
        return self.config.output[0][0].max(0)[1].byte().cpu().data.numpy()

    def pre_process(self, img_path):
        img = cv2.imread(img_path)
        img_orig = np.copy(img)
        
        img = img.astype(np.float32)
        for j in range(3):
            img[:, :, j] -= self.config.mean[j]
        for j in range(3):
            img[:, :, j] /= self.config.std[j]

        # resize the image to 1024x512x3
        img = cv2.resize(img, (self.config.in_width, self.config.in_height))
        img_orig = cv2.resize(img_orig, (self.config.in_width, self.config.in_height))

        img /= 255
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension

        return img_tensor

def getFileList(input_dir):
    f_list = []
    fs = os.listdir(input_dir)
    for f in fs:
        f_list.append(os.path.join(input_dir, f))
    
    return f_list

if __name__ == "__main__":
    from config import config

    espnet_test = ESPNetTest(config)

    files = getFileList('your test images path')
    for f in files:
        input_tensor = espnet_test.pre_process(f)
        espnet_test(input_tensor)
        result = espnet_test.post_process()
        cv2.imwrite('./res_imgs/%s' % f.split('/')[-1].replace('jpg', 'png'), result)
