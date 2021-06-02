# -*- coding:utf-8 -*-
import cv2
from deepvac.datasets import OsWalkDataset


class OsWalkDataset2(OsWalkDataset):

    def __init__(self, deepvac_config, sample_path):
        super(OsWalkDataset2, self).__init__(deepvac_config, sample_path)

    def __getitem__(self, index):
        filepath = self.files[index]
        sample = cv2.imread(filepath)
        if self.config.transform is not None:
            sample = self.config.transform(sample)
        if self.config.composer is not None:
            sample = self.config.composer(sample)
        return sample, filepath
