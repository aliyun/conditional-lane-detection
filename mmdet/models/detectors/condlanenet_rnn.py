import os
import math
import random

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from .single_stage import SingleStageDetector
from ..builder import DETECTORS
from ..losses import CondLaneRNNLoss


@DETECTORS.register_module
class CurvelanesRnn(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_weights={},
                 num_classes=1):
        super(CurvelanesRnn, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=head,
            train_cfg=None,
            test_cfg=None,
            pretrained=pretrained)
        self.num_classes = num_classes
        self.head = head

        self.loss = CondLaneRNNLoss(loss_weights, num_classes)

    def parse_gt(self, gts, device):
        regs, reg_masks, rows, row_masks, lane_ranges = [], [], [], [], []
        for gt in gts['gt_masks']:
            reg = (torch.from_numpy(gt['reg']).to(device)).unsqueeze(0)
            reg_mask = (torch.from_numpy(
                gt['reg_mask']).to(device)).unsqueeze(0)
            regs.append(reg)
            reg_masks.append(reg_mask)
            row = (torch.from_numpy(
                gt['row']).to(device)).unsqueeze(0).unsqueeze(0)
            rows.append(row)
            row_mask = (torch.from_numpy(
                gt['row_mask']).to(device)).unsqueeze(0).unsqueeze(0)
            row_masks.append(row_mask)
            lane_range = (torch.from_numpy(gt['range']).to(device))
            lane_ranges.append(lane_range)
        return regs, reg_masks, rows, row_masks, lane_ranges

    def parse_pos(self, gt_masks, hm_shape, device, mask_shape=None):
        b = len(gt_masks)
        n = self.num_classes
        hm_h, hm_w = hm_shape[:2]
        if mask_shape is None:
            mask_h, mask_w = hm_shape[:2]
        else:
            mask_h, mask_w = mask_shape[:2]
        poses = []
        regs = []
        reg_masks = []
        rows = []
        row_masks = []
        lane_ranges = []
        labels = []
        num_ins = []
        for idx, gt_mask_batch in enumerate(gt_masks):
            num = 0
            for point_idx, gt_mask in enumerate(gt_mask_batch):
                gts = self.parse_gt(gt_mask, device=device)
                reg, reg_mask, row, row_mask, lane_range = gts
                label = gt_mask['label']
                num += len(gt_mask['points']) * len(reg)
                for p in gt_mask['points']:
                    pos = idx * n * hm_h * hm_w + label * hm_h * hm_w + p[
                        1] * hm_w + p[0]
                    # pos = [idx, label, p[1], p[0]]
                    poses.append([pos, len(reg)])
                # m['label'] = torch.from_numpy(np.array(m['label'])).to(device)
                for i in range(len(gt_mask['points'])):
                    for j in range(len(reg)):
                        labels.append(label)
                        regs.append(reg[j])
                        reg_masks.append(reg_mask[j])
                        rows.append(row[j])
                        row_masks.append(row_mask[j])
                        lane_ranges.append(lane_range[j])

            if num == 0:
                reg = torch.zeros((1, 1, mask_h, mask_w)).to(device)
                reg_mask = torch.zeros((1, 1, mask_h, mask_w)).to(device)
                row = torch.zeros((1, 1, mask_h)).to(device)
                row_mask = torch.zeros((1, 1, mask_h)).to(device)
                lane_range = torch.zeros((1, mask_h),
                                         dtype=torch.int64).to(device)
                label = 0
                pos = idx * n * hm_h * hm_w + random.randint(
                    0, n * hm_h * hm_w - 1)
                num = 1
                labels.append(label)
                poses.append([pos, 1])
                regs.append(reg)
                reg_masks.append(reg_mask)
                rows.append(row)
                row_masks.append(row_mask)
                lane_ranges.append(lane_range)

            num_ins.append(num)
        if len(regs) > 0:
            regs = torch.cat(regs, 1)
            reg_masks = torch.cat(reg_masks, 1)
            rows = torch.cat(rows, 1)
            row_masks = torch.cat(row_masks, 1)
            lane_ranges = torch.cat(lane_ranges, 0)

        gts = dict(
            gt_reg=regs,
            gt_reg_mask=reg_masks,
            gt_rows=rows,
            gt_row_masks=row_masks,
            gt_ranges=lane_ranges)

        return poses, labels, num_ins, gts

    def forward(self, img, img_metas=None, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if img_metas is None:
            return self.test_inference(img)
        elif return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, img_metas, **kwargs):
        gt_batch_masks = [m['gt_masks'] for m in img_metas]
        hm_shape = img_metas[0]['hm_shape']
        mask_shape = img_metas[0]['mask_shape']
        poses, labels, num_ins, gts = self.parse_pos(
            gt_batch_masks, hm_shape, img.device, mask_shape=mask_shape)
        state_tgt = []
        for pos in poses:
            if pos[1] < 2:
                state_tgt.append(0)
            else:
                for i in range(pos[1] - 1):
                    state_tgt.append(1)
                state_tgt.append(0)

        state_tgt = torch.from_numpy(np.array(state_tgt,
                                              np.int32)).long().to(img.device)
        gts['gt_states'] = state_tgt
        kwargs.update(gts)

        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output, memory = self.neck(output)
        output = self.bbox_head.forward_train(output, poses, num_ins, memory)
        losses = self.loss(output, img_metas, **kwargs)
        return losses

    def test_inference(self, img):
        # with Timer("Elapsed time in model inference: %f"):
        # output = self.backbone(img.type(torch.cuda.FloatTensor))
        output = self.backbone(img)
        if hasattr(self, 'neck') and self.neck:
            output, memory = self.neck(output)
        if self.head:
            seeds, hm = self.bbox_head.forward_test(output, 0.5)
        return [seeds, hm]

    def forward_test(self,
                     img,
                     img_metas,
                     benchmark=False,
                     hack_seeds=None,
                     **kwargs):
        """Test without augmentation."""
        # with Timer("Elapsed time in model inference: %f"):
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        if hasattr(self, 'neck') and self.neck:
            output, memory = self.neck(output)
        if self.head:
            seeds, hm = self.bbox_head.forward_test(
                output, kwargs['thr'], hack_seeds=hack_seeds, memory=memory)
        return [seeds, hm]

    def forward_dummy(self, img):
        x = self.backbone(img)
        return x
