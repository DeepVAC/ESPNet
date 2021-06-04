import sys
import os
import math
import time
import numpy as np
import cv2
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from deepvac import LOG, DeepvacTrain
from modules.utils_IOU_eval import IOUEval

class ESPNetTrain(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(ESPNetTrain, self).__init__(deepvac_config)
        self.config.epoch_loss = []

    def doFeedData2Device(self):
        self.config.sample = self.config.sample.to(self.config.device)
        self.config.target = [tar.to(self.config.device) for tar in self.config.target]

    def train(self):
        self.iou_eval_val = IOUEval(self.config.cls_num)
        self.iou_eval_train = IOUEval(self.config.cls_num)
        for i, loader in enumerate(self.config.train_loader_list):
            self.config.train_loader = loader
            super(ESPNetTrain, self).train()

    #only save model for last loader
    def doSave(self):
        if not self.config.train_loader.is_last_loader:
            return
        super(ESPNetTrain, self).doSave()
        
    def postIter(self):
        if not self.config.train_loader.is_last_loader:
            return

        self.config.epoch_loss.append(self.config.loss.item())
        if self.config.phase == 'TRAIN':
            self.iou_eval_train.addBatch(self.config.output[0].max(1)[1].data.cpu().numpy(), self.config.target[0].data.cpu().numpy())
        else:
            self.iou_eval_val.addBatch(self.config.output[0].max(1)[1].data.cpu().numpy(), self.config.target[0].data.cpu().numpy())

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


        pred_fusion, pred_detail, pred_sematic = self.config.output
        gt_fusion, gt_detail = self.config.target
        # do semantic loss
        gt_semantic = F.interpolate(gt_fusion.unsqueeze(1).float(), scale_factor=1/8, mode='nearest').long().squeeze()
        semantic_loss = self.config.criterion[0](pred_sematic, gt_semantic)

        # do detail loss
        detail_loss = self.config.criterion[1](pred_detail, gt_fusion)
        boundaries = (gt_detail!=0)
        detail_loss = torch.mean(boundaries*detail_loss)

        # do fusion loss
        fusion_loss = self.config.criterion[0](pred_fusion, gt_fusion)
        self.config.loss = 10*semantic_loss + 10*detail_loss + fusion_loss


if __name__ == "__main__":
    from config import config
    train = ESPNetTrain(config)
    train()
