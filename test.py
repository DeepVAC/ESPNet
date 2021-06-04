# -*- coding:utf-8 -*-
import sys
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import deepvac
from deepvac import LOG, Deepvac
from modules.utils_IOU_eval import IOUEval
from data.dataloader import OsWalkDataset2

class ESPNetTest(Deepvac):
    def __init__(self, deepvac_config):
        super(ESPNetTest, self).__init__(deepvac_config)
        os.makedirs(self.config.show_output_dir, exist_ok=True)
        if self.config.test_label_path is not None:
            self.config.iou_eval = IOUEval(self.config.cls_num)

    def preIter(self):
        assert len(self.config.target) == 1, 'config.core.test_batch_size must be set to 1 in current test mode.'
        self.config.filepath = self.config.target[0]

    def postIter(self):
        self.config.output = self.config.output[0].squeeze().cpu().numpy()
        if self.config.output.ndim == 2:
            self.config.mask = (self.config.output > 0.5)
        elif self.config.output.ndim == 3:
            self.config.mask = self.config.output.argmax(0)
        LOG.logI('{}: [output shape: {}] [{}/{}]'.format(self.config.phase, self.config.mask.shape, self.config.test_step + 1, len(self.config.test_loader)))

        cv_img = cv2.imread(self.config.filepath, 1)
        h, w = cv_img.shape[:2]
        self.config.mask = cv2.resize(np.uint8(self.config.mask), (w, h), cv2.INTER_NEAREST)

        filename = self.config.filepath.split('/')[-1]
        mask_filename = filename + "_mask.png"
        savepath = os.path.join(self.config.show_output_dir, filename)
        mask_savepath = os.path.join(self.config.show_output_dir, mask_filename)

        if self.config.test_label_path:
            label_file = os.path.join(self.config.test_label_path, filename.replace(".jpg", ".png"))
            self.config.label = cv2.imread(label_file, 0)
            self.config.iou_eval.addBatch(self.config.mask, self.config.label)

        classMap_numpy_color = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in np.unique(self.config.mask):
            [r, g, b] = self.config.pallete[idx]
            classMap_numpy_color[self.config.mask == idx] = [b, g, r]
        overlayed = cv2.addWeighted(cv_img, 0.5, classMap_numpy_color, 0.5, 0)
        cv2.imwrite(savepath, overlayed)
        cv2.imwrite(mask_savepath, classMap_numpy_color)
        LOG.logI('{}: [out cv image save to {}] [{}/{}]\n'.format(self.config.phase, savepath, self.config.test_step + 1, len(self.config.test_loader)))

    def testFly(self):
        if self.config.test_loader:
            self.test()
            if self.config.test_label_path is None:
                return
            *_, self.config.mIOU = self.config.iou_eval.getMetric()
            LOG.logI(">>> {}: [dataset: {}, mIOU: {:.3f}]".format(self.config.phase, self.config.filepath.split('/')[-2], self.config.mIOU))
            return

        LOG.logE("You have to reimplement testFly() in subclass {} if you didn't set any valid input, e.g. config.core.test_loader.".format(self.name()), exit=True)


if __name__ == "__main__":
    from config import config

    def check_args(idx, argv):
        return (len(argv) > idx) and (os.path.exists(argv[idx]))

    if check_args(1, sys.argv):
        config.core.model_path = sys.argv[1]
    if check_args(2, sys.argv):
        config.test_sample_path = sys.argv[2]
    if check_args(3, sys.argv):
        config.core.test_label_path = sys.argv[3]

    if (config.core.model_path is None) or (config.test_sample_path is None):
        helper = '''model_path or test_sample_path not found, please check:
                config.core.model_path or sys.argv[1] to init model path
                config.test_sample_path or sys.argv[2] to init test sample path
                config.test_label_path or sys.argv[3] to init test sample path (not required)
                for example:
                python3 test.py <trained-model-path> <test sample path> [test label path(not required)]'''
        print(helper)
        sys.exit(1)

    config.core.test_dataset = OsWalkDataset2(config, config.test_sample_path)
    config.core.test_loader = torch.utils.data.DataLoader(config.core.test_dataset, batch_size=1, shuffle=False, num_workers=config.core.num_workers, pin_memory=True)
    ESPNetTest(config)()
