import os
import math
import copy
import random

import numpy as np
import cv2
import torch
import torch.nn.functional as F


def compute_locations(shape, device):
    pos = torch.arange(
        0, shape[-1], step=1, dtype=torch.float32, device=device)
    pos = pos.reshape((1, 1, -1))
    pos = pos.repeat(shape[0], shape[1], 1)
    return pos


def nms_seeds_tiny(seeds, thr):

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
        self.pos = compute_locations(
            mask_size, device='cuda:0').repeat(100, 1, 1)

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
        seeds = nms_seeds_tiny(output, self.nms_thr)
        if len(seeds) == 0:
            return [], seeds
        collection = self.collect_seeds(seeds)
        if collection is None:
            return [], seeds
        masks, regs, scores, ranges = collection
        lanes = self.lane_post_process_all(masks, regs, scores, ranges,
                                           downscale, seeds)
        return lanes, seeds


class CurvelanesPostProcessor(object):

    def __init__(self,
                 mask_size,
                 hm_thr=0.3,
                 min_points=2,
                 use_offset=True,
                 **kwargs):
        self.hm_thr = hm_thr
        self.min_points = min_points
        self.use_offset = use_offset
        # nms 停止线和路沿单独一组

        self.pos = compute_locations(
            mask_size, device='cuda:0').repeat(100, 1, 1)

    def lane_post_process_all(self, seeds, downscale=4, min_points=2):

        def parser_seeds(seeds):
            masks = []
            regs = []
            ranges = []
            coords = []
            scores = []
            seed_idxes = []
            for idx, seed in enumerate(seeds):
                masks.append(seed['mask'])
                regs.append(seed['reg'])
                ranges.append(seed['range'])
                for i in range(seed['mask'].size()[0]):
                    scores.append(seed['score'])
                    coords.append(seed['coord'])
                    seed_idxes.append(idx)
            if len(masks) > 0:
                masks = torch.cat(masks, 0)
                regs = torch.cat(regs, 0)
                ranges = torch.cat(ranges, 0)
            return masks, regs, ranges, coords, scores, seed_idxes

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

        masks, regs, ranges, coords, scores, seed_idxes = parser_seeds(seeds)

        lanes = []
        if len(scores) == 0:
            return lanes
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
                        seed=seeds[seed_idxes[lane_idx]]))
        return lanes

    def __call__(self, output, downscale):
        lanes = []
        seeds = nms_seeds_tiny(output, 1)
        lanes = self.lane_post_process_all(seeds, downscale)
        return lanes, seeds
