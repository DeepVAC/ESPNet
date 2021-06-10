# -*- coding:utf-8 -*-
import sys
import os
import math
import time
import copy
import numpy as np
import cv2
from scipy.ndimage import grey_dilation, grey_erosion
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from deepvac import LOG, DeepvacTrain

class ESPNetTrain(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(ESPNetTrain, self).__init__(deepvac_config)
        self.config.net.freeze_norm()

    def preEpoch(self):
        LOG.logI("backup modnet copying ... ...")
        self.config.net_backup = copy.deepcopy(self.config.net)
        self.config.net_backup.eval()

    def doForward(self):
        super(ESPNetTrain, self).doForward()
        with torch.no_grad():
            self.config.output_backup = self.config.net_backup(self.config.sample)

    def doLoss(self):
        if not self.config.is_train:
            return
        pred_fusion, pred_detail, pred_semantic = self.config.output

        n, _, h, w = pred_fusion.shape
        np_pred_fg  = pred_fusion.max(1)[1].cpu().numpy()
        np_boundaries = np.zeros([n, h, w])
        for sdx in range(0, n):
            sample_np_boundaries = np_boundaries[sdx, ...]
            sample_np_pred_fg = np_pred_fg[sdx, ...]

            side = int((h + w) / 2 * 0.05)
            for cls_idx in range(self.config.cls_num):
                if cls_idx==0:
                    continue
                cls_mask = np.zeros(sample_np_pred_fg.shape, np.uint8)
                cls_mask[np.where(sample_np_pred_fg==cls_idx)] = 1
                dilated = grey_dilation(cls_mask, size=(side, side))
                eroded = grey_erosion(cls_mask, size=(side, side))
                sample_np_boundaries[np.where(dilated - eroded != 0)] = 1
            np_boundaries[sdx, ...] = sample_np_boundaries

        self.config.classes_weight = self.config.classes_weight.to(self.config.device)
        boundaries = torch.tensor(np_boundaries).float().to(self.config.device)

        # soc semantic loss
        downsampled_fusion = F.interpolate(pred_fusion, scale_factor=1/8, mode='nearest')
        downsampled_pseudo_gt_fusion = downsampled_fusion.max(1)[1]
        pseudo_gt_semantic = pred_semantic.max(1)[1]
        soc_semantic_loss = F.cross_entropy(pred_semantic, downsampled_pseudo_gt_fusion.detach()) + \
                            F.cross_entropy(downsampled_fusion, pseudo_gt_semantic.detach())

        backup_fusion, backup_detail, _ = self.config.output_backup
        # sub-objectives consistency between `pred_detail` and `pred_backup_detail` (on boundaries only)
        backup_detail_loss = boundaries * F.cross_entropy(pred_detail, backup_detail.max(1)[1], weight=self.config.classes_weight, reduction='none')
        backup_detail_loss = torch.mean(backup_detail_loss)

        # sub-objectives consistency between pred_matte` and `pred_backup_matte` (on boundaries only)
        backup_fusion_loss = boundaries * F.cross_entropy(pred_fusion, backup_fusion.max(1)[1], reduction='none')
        backup_fusion_loss = torch.mean(backup_fusion_loss)

        self.config.loss = 5*soc_semantic_loss + backup_detail_loss + backup_fusion_loss

if __name__ == "__main__":
    from config import config

    train = ESPNetTrain(config)
    train()
