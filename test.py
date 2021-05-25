import sys
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import deepvac
from deepvac import LOG, Deepvac
from data.dataloader import OsWalkDataset2

class ESPNetTest(Deepvac):
    def __init__(self, deepvac_config):
        super(ESPNetTest, self).__init__(deepvac_config)
        os.makedirs("output/show_results", exist_ok=True)

    def postIter(self):
        self.config.mask = self.config.output[0][0].argmax(0).cpu().numpy()
        LOG.logI('{}: [output shape: {}] [{}/{}]'.format(self.config.phase, self.config.mask.shape, self.config.test_step + 1, len(self.config.test_loader)))

        cv_img = cv2.imread(self.config.filepath[0], 1)
        h, w = cv_img.shape[:2]
        self.config.mask = cv2.resize(np.uint8(self.config.mask), (w, h), cv2.INTER_NEAREST)

        filename = self.config.filepath[0].split('/')[-1]
        savepath = os.path.join("output/show_results", filename)
        cv_img[:, :, 1][self.config.mask == 1] = 255
        cv2.imwrite(savepath, cv_img)
        LOG.logI('{}: [out cv image save to {}] [{}/{}]\n'.format(self.config.phase, savepath, self.config.test_step + 1, len(self.config.test_loader)))

    def test(self):
        LOG.logI("config.core.test_load has been set, do test() with config.core.test_loader")
        for self.config.test_step, (self.config.filepath, self.config.sample) in enumerate(self.config.test_loader):
            self.preIter()
            self.doFeedData2Device()
            self.doForward()
            LOG.logI('{}: [input shape: {}] [{}/{}]'.format(self.config.phase, self.config.sample.shape, self.config.test_step + 1, len(self.config.test_loader)))
            self.postIter()


if __name__ == "__main__":
    from config import config

    def check_args(idx, argv):
        return (len(argv) > idx) and (os.path.exists(argv[idx]))

    if check_args(1, sys.argv):
        config.core.model_path = sys.argv[1]
    if check_args(2, sys.argv):
        config.test_sample_path = sys.argv[2]

    if (config.core.model_path is None) or (config.test_sample_path is None):
        helper = '''model_path or test_sample_path not found, please check:
                config.core.model_path or sys.argv[1] to init model path
                config.test_sample_path or sys.argv[2] to init test sample path
                for example:
                python3 test.py <trained-model-path> <test sample path>'''
        print(helper)
        sys.exit(1)

    config.core.test_dataset = OsWalkDataset2(config, config.test_sample_path)
    config.core.test_loader = torch.utils.data.DataLoader(config.core.test_dataset, batch_size=1, shuffle=False, num_workers=config.core.num_workers, pin_memory=True)
    ESPNetTest(config)()
