import torch
from torch import nn
from torch import optim

import cv2
import math
import time
import numpy as np
import os

import deepvac
from deepvac import LOG, DeepvacTrain

from utils.utils_IOU_eval import iouEval

class ESPNetTrain(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(ESPNetTrain, self).__init__(deepvac_config)
        self.config.save_flag = False
        self.config.epoch_loss = []
        self.iou_eval_val = iouEval(self.config.cls_num)
        self.iou_eval_train = iouEval(self.config.cls_num)

    def train(self):
        for i, loader in enumerate(self.config.train_loader_list):
            print('loader: {}'.format(i))
            self.config.save_flag = False
            if i == 4:
                self.config.save_flag = True
                self.iou_eval_train = iouEval(self.config.cls_num)
            self.config.train_loader = loader
            super(ESPNetTrain, self).train()

    def postIter(self):
        if not self.config.save_flag:
            return
        self.config.epoch_loss.append(self.config.loss.item())
        if self.config.phase == 'TRAIN':
            self.iou_eval_train.addBatch(self.config.output[0].max(1)[1].data, self.config.target.data)
        else:
            self.iou_eval_val.addBatch(self.config.output[0].max(1)[1].data, self.config.target.data)

    def preEpoch(self):
        self.config.epoch_loss = []

    def postEpoch(self):
        if not self.config.save_flag:
            return
        average_epoch_loss = sum(self.config.epoch_loss) / len(self.epoch_loss)

        if self.config.phase == 'TRAIN':
            overall_acc, per_class_acc, per_class_iu, mIOU = self.iou_eval_train.getMetric()
        else:
            overall_acc, per_class_acc, per_class_iu, mIOU = self.iou_eval_val.getMetric()
        print("Epoch : " + str(epoch) + ' Details')
        print("\nEpoch No.: %d\t%s Loss = %.4f\t %s mIOU = %.4f\t" % (self.config.epoch, self.config.phase, average_epoch_loss, self.config.phase, mIOU))

    def doSchedule(self):
        if not self.config.save_flag:
            return
        self.config.scheduler.step()

    def doLoss(self):
        if not self.config.is_train:
            return
        loss1, loss2 = self.config.criterion(self.config.output[0], self.config.target), self.config.criterion(self.config.output[1], self.config.target)
        self.config.loss = loss1 + loss2

if __name__ == "__main__":
    from config import config
    train = ESPNetTrain(config.train)
    train()
