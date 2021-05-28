import os
import math
import random
from functools import cmp_to_key

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from .single_stage import SingleStageDetector
from ..builder import DETECTORS
from ..losses import CondLaneLoss


class CondLanePostProcessor(object):

    def __init__(self,
                 mask_size,
                 hm_thr=0.5,
                 min_points=5,
                 hm_downscale=16,
                 mask_downscale=8,
                 use_offset=True,
                 **kwargs):
        self.hm_thr = hm_thr
        self.min_points = min_points
        self.hm_downscale = hm_downscale
        self.mask_downscale = mask_downscale
        self.use_offset = use_offset
        self.horizontal_id = [5]
        # nms 停止线和路沿单独一组
        self.nms_groups = [[1]]
        if 'nms_thr' in kwargs:
            self.nms_thr = kwargs['nms_thr']
        else:
            self.nms_thr = 3
        self.pos = self.compute_locations(
            mask_size, device='cuda:0').repeat(100, 1, 1)

    def nms_seeds_tiny(self, seeds, thr):

        def cal_dis(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        def search_groups(coord, groups, thr):
            for idx_group, group in enumerate(groups):
                for group_point in group:
                    group_point_coord = group_point[1]
                    if cal_dis(coord, group_point_coord) <= thr:
                        return idx_group
            return -1

        def choose_highest_score(group):
            highest_score = -1
            highest_idx = -1
            for idx, _, score in group:
                if score > highest_score:
                    highest_idx = idx
            return highest_idx

        def update_coords(points_info, thr=4):
            groups = []
            keep_idx = []
            for idx, (coord, score) in enumerate(points_info):
                idx_group = search_groups(coord, groups, thr)
                if idx_group < 0:
                    groups.append([(idx, coord, score)])
                else:
                    groups[idx_group].append((idx, coord, score))
            for group in groups:
                choose_idx = choose_highest_score(group)
                if choose_idx >= 0:
                    keep_idx.append(choose_idx)
            return keep_idx

        points = [(item['coord'], item['score']) for item in seeds]
        keep_idxes = update_coords(points, thr=thr)
        update_seeds = [seeds[idx] for idx in keep_idxes]
        return update_seeds

    def compute_locations(self, shape, device):
        pos = torch.arange(
            0, shape[-1], step=1, dtype=torch.float32, device=device)
        pos = pos.reshape((1, 1, -1))
        pos = pos.repeat(shape[0], shape[1], 1)
        return pos

    def lane_post_process_all(self, masks, regs, scores, ranges, downscale,
                              seeds):

        def get_range(ranges):
            max_rows = ranges.shape[1]
            lane_ends = []
            for idx, lane_range in enumerate(ranges):
                min_idx = max_idx = None
                for row_idx, valid in enumerate(lane_range):
                    if valid:
                        min_idx = row_idx - 1
                        break
                for row_idx, valid in enumerate(lane_range[::-1]):
                    if valid:
                        max_idx = len(lane_range) - row_idx
                        break
                if max_idx is not None:
                    max_idx = min(max_rows - 1, max_idx)
                if min_idx is not None:
                    min_idx = max(0, min_idx)
                lane_ends.append([min_idx, max_idx])
            return lane_ends

        lanes = []
        num_ins = masks.size()[0]
        mask_softmax = F.softmax(masks, dim=-1)
        row_pos = torch.sum(
            self.pos[:num_ins] * mask_softmax,
            dim=2).detach().cpu().numpy().astype(np.int32)
        # row_pos = torch.argmax(masks, -1).detach().cpu().numpy()
        ranges = torch.argmax(ranges, 1).detach().cpu().numpy()
        lane_ends = get_range(ranges)
        regs = regs.detach().cpu().numpy()
        num_lanes, height, width = masks.shape
        # with Timer("post process time: %f"):

        for lane_idx in range(num_lanes):
            if lane_ends[lane_idx][0] is None or lane_ends[lane_idx][1] is None:
                continue
            selected_ys = np.arange(lane_ends[lane_idx][0],
                                    lane_ends[lane_idx][1] + 1)
            selected_col_idx = row_pos[lane_idx, :]
            selected_xs = selected_col_idx[selected_ys]
            if self.use_offset:
                selected_regs = regs[lane_idx, selected_ys, selected_xs]
            else:
                selected_regs = 0.5
            selected_xs = np.expand_dims(selected_xs, 1)
            selected_ys = np.expand_dims(selected_ys, 1)
            points = np.concatenate((selected_xs, selected_ys),
                                    1).astype(np.float32)
            points[:, 0] = points[:, 0] + selected_regs
            points *= downscale

            if len(points) > 1:
                lanes.append(
                    dict(
                        id_class=1,
                        points=points,
                        score=scores[lane_idx],
                        seed=seeds[lane_idx]))
        return lanes

    def collect_seeds(self, seeds):
        masks = []
        regs = []
        scores = []
        ranges = []
        for seed in seeds:
            masks.append(seed['mask'])
            regs.append(seed['reg'])
            scores.append(seed['score'])
            ranges.append(seed['range'])
        if len(masks) > 0:
            masks = torch.cat(masks, 0)
            regs = torch.cat(regs, 0)
            ranges = torch.cat(ranges, 0)
            return masks, regs, scores, ranges
        else:
            return None

    def extend_line(self, line, dis=100):
        extended = copy.deepcopy(line)
        start = line[-2]
        end = line[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        norm = math.sqrt(dx**2 + dy**2)
        dx = dx / norm
        dy = dy / norm
        extend_point = [start[0] + dx * dis, start[1] + dy * dis]
        extended.append(extend_point)
        return extended

    def __call__(self, output, downscale):
        lanes = []
        # with Timer("Elapsed time in tiny nms: %f"):
        seeds = self.nms_seeds_tiny(output, self.nms_thr)
        if len(seeds) == 0:
            return [], seeds
        collection = self.collect_seeds(seeds)
        if collection is None:
            return [], seeds
        masks, regs, scores, ranges = collection
        lanes = self.lane_post_process_all(masks, regs, scores, ranges,
                                           downscale, seeds)
        return lanes, seeds


@DETECTORS.register_module
class CondLaneNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_weights={},
                 output_scale=4,
                 num_classes=1):
        super(CondLaneNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=head,
            train_cfg=None,
            test_cfg=None,
            pretrained=pretrained)
        self.num_classes = num_classes
        self.head = head
        if test_cfg is not None and 'out_scale' in test_cfg.keys():
            self.output_scale = test_cfg['out_scale']
        else:
            self.output_scale = 4

        self.loss = CondLaneLoss(loss_weights, num_classes)

    def parse_gt(self, gts, device):
        reg = (torch.from_numpy(gts['reg']).to(device)).unsqueeze(0)
        reg_mask = (torch.from_numpy(gts['reg_mask']).to(device)).unsqueeze(0)
        row = (torch.from_numpy(
            gts['row']).to(device)).unsqueeze(0).unsqueeze(0)
        row_mask = (torch.from_numpy(
            gts['row_mask']).to(device)).unsqueeze(0).unsqueeze(0)
        if 'range' in gts:
            lane_range = (torch.from_numpy(gts['range']).to(device))
        else:
            lane_range = torch.zeros((1, mask.shape[-2]),
                                     dtype=torch.int64).to(device)
        return reg, reg_mask, row, row_mask, lane_range

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
        for idx, m_img in enumerate(gt_masks):
            num = 0
            for m in m_img:
                gts = self.parse_gt(m, device=device)
                reg, reg_mask, row, row_mask, lane_range = gts
                label = m['label']
                num += len(m['points'])
                for p in m['points']:
                    pos = idx * n * hm_h * hm_w + label * hm_h * hm_w + p[
                        1] * hm_w + p[0]
                    # pos = [idx, label, p[1], p[0]]
                    poses.append(pos)
                # m['label'] = torch.from_numpy(np.array(m['label'])).to(device)
                for i in range(len(m['points'])):
                    labels.append(label)
                    regs.append(reg)
                    reg_masks.append(reg_mask)
                    rows.append(row)
                    row_masks.append(row_mask)
                    lane_ranges.append(lane_range)

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
                poses.append(pos)
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
        kwargs.update(gts)

        output = self.backbone(img.type(torch.cuda.FloatTensor))

        output, memory = self.neck(output)

        if self.head:
            outputs = self.bbox_head.forward_train(output, poses, num_ins)
            output = outputs[:4]
            mask_branch, reg_branch = outputs[-1]
            kwargs.update(dict(mask_branch=mask_branch, reg_branch=reg_branch))
        h, w = img_metas[0]['img_shape'][:2]
        losses = self.loss(output, img_metas, **kwargs)
        return losses

    def test_inference(self, img):
        output = self.backbone(img)
        output, memory = self.neck(output)
        seeds, hm = self.bbox_head.forward_test(output, None, 0.5)
        return [seeds, hm]

    def forward_test(self,
                     img,
                     img_metas,
                     benchmark=False,
                     hack_seeds=None,
                     **kwargs):
        """Test without augmentation."""
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output, memory = self.neck(output)
        if self.head:
            seeds, hm = self.bbox_head.forward_test(output, hack_seeds,
                                                    kwargs['thr'])
        return [seeds, hm]

    def forward_dummy(self, img):
        x = self.backbone(img)
        return x
