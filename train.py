import sys
import os
import math
import time
import numpy as np
import cv2
import torch
from torch import nn
from torch import optim
from deepvac import LOG, DeepvacTrain
from utils.utils_IOU_eval import IOUEval

class ESPNetTrain(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(ESPNetTrain, self).__init__(deepvac_config)
        self.config.epoch_loss = []

    def train(self):
        self.iou_eval_val = IOUEval(self.config.cls_num)
        self.iou_eval_train = IOUEval(self.config.cls_num)
        for i, loader in enumerate(self.config.train_loader_list):
            self.config.train_loader = loader
            super(ESPNetTrain, self).train()
        
    def postIter(self):
        if not self.config.train_loader.is_last_loader:
            return

        self.config.epoch_loss.append(self.config.loss.item())
        if self.config.phase == 'TRAIN':
            self.iou_eval_train.addBatch(self.config.output[0].max(1)[1].data, self.config.target.data)
        else:
            self.iou_eval_val.addBatch(self.config.output[0].max(1)[1].data, self.config.target.data)

    def preEpoch(self):
        self.config.epoch_loss = []

    def postEpoch(self):
        if not self.config.train_loader.is_last_loader:
            return
        average_epoch_loss = sum(self.config.epoch_loss) / len(self.config.epoch_loss)

        if self.config.phase == 'TRAIN':
            overall_acc, per_class_acc, per_class_iu, mIOU = self.iou_eval_train.getMetric()
        else:
            overall_acc, per_class_acc, per_class_iu, mIOU = self.iou_eval_val.getMetric()
        LOG.logI("Epoch : {} Details".format(self.config.epoch))
        LOG.logI("\nEpoch No.: %d\t%s Loss = %.4f\t %s mIOU = %.4f\t" % (self.config.epoch, self.config.phase, average_epoch_loss, self.config.phase, mIOU))

    def doSchedule(self):
        if not self.config.train_loader.is_last_loader:
            return
        self.config.scheduler.step()

    def doLoss(self):
        if not self.config.is_train:
            return
        loss1, loss2 = self.config.criterion(self.config.output[0], self.config.target), self.config.criterion(self.config.output[1], self.config.target)
        self.config.loss = loss1 + loss2

if __name__ == "__main__":
    from config import config
    train = ESPNetTrain(config)
    train()
