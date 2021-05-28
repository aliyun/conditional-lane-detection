import argparse
import os
import numpy as np
import random
import math
import json

import cv2
import mmcv
import torch
import torch.distributed as dist
import PIL.Image
import PIL.ImageDraw
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils.general_utils import mkdir
from mmdet.models.detectors.condlanenet import CondLaneNet, CondLanePostProcessor

from tools.condlanenet.common import COLORS, parse_lanes


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='seg checkpoint file')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--hm_thr', type=float, default=0.5)
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--show_dst',
        default='./work_dirs/culane/watch',
        help='path to save visualized results.')
    parser.add_argument(
        '--result_dst',
        default='./work_dirs/culane/results',
        help='path to save results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def adjust_result(lanes, crop_bbox, img_shape, tgt_shape=(590, 1640)):

    def in_range(pt, img_shape):
        if pt[0] >= 0 and pt[0] < img_shape[1] and pt[1] >= 0 and pt[
                1] <= img_shape[0]:
            return True
        else:
            return False

    left, top, right, bot = crop_bbox
    h_img, w_img = img_shape[:2]
    crop_width = right - left
    crop_height = bot - top
    ratio_x = crop_width / w_img
    ratio_y = crop_height / h_img
    offset_x = (tgt_shape[1] - crop_width) / 2
    offset_y = top

    results = []
    if lanes is not None:
        for key in range(len(lanes)):
            pts = []
            for pt in lanes[key]['points']:
                pt[0] = float(pt[0] * ratio_x + offset_x)
                pt[1] = float(pt[1] * ratio_y + offset_y)
                pts.append(pt)
            if len(pts) > 1:
                results.append(pts)
    return results


def adjust_point(hm_points,
                 downscale,
                 crop_bbox,
                 img_shape,
                 tgt_shape=(590, 1640)):
    left, top, right, bot = crop_bbox
    h_img, w_img = img_shape[:2]
    crop_width = right - left
    crop_height = bot - top
    ratio_x = crop_width / w_img
    ratio_y = crop_height / h_img
    offset_x = (tgt_shape[1] - crop_width) / 2
    offset_y = top
    coord_x = float((hm_points[0] + 0.5) * downscale * ratio_x + offset_x)
    coord_y = float((hm_points[1] + 0.5) * downscale * ratio_y + offset_y)
    coord_x = max(0, coord_x)
    coord_x = min(coord_x, tgt_shape[1])
    coord_y = max(0, coord_y)
    coord_y = min(coord_y, tgt_shape[0])
    return [coord_x, coord_y]


def out_result(lanes, dst=None):
    if dst is not None:
        with open(dst, 'w') as f:
            for lane in lanes:
                for idx, p in enumerate(lane):
                    if idx == len(lane) - 1:
                        print('{:.2f} '.format(p[0]), end='', file=f)
                        print('{:.2f}'.format(p[1]), file=f)
                    else:
                        print('{:.2f} '.format(p[0]), end='', file=f)
                        print('{:.2f} '.format(p[1]), end='', file=f)


def vis_one(results, filename, width=9):
    img = cv2.imread(filename)
    img_gt = cv2.imread(filename)
    img_pil = PIL.Image.fromarray(img)
    img_gt_pil = PIL.Image.fromarray(img_gt)
    num_failed = 0
    preds, annos = parse_lanes(results, filename, (590, 1640))
    for idx, anno_lane in enumerate(annos):
        PIL.ImageDraw.Draw(img_gt_pil).line(
            xy=anno_lane, fill=COLORS[idx + 1], width=width)
    for idx, pred_lane in enumerate(preds):
        PIL.ImageDraw.Draw(img_pil).line(
            xy=pred_lane, fill=COLORS[idx + 1], width=width)

    img = np.array(img_pil, dtype=np.uint8)
    img_gt = np.array(img_gt_pil, dtype=np.uint8)

    return img, img_gt, num_failed


def single_gpu_test(seg_model,
                    data_loader,
                    show=None,
                    hm_thr=0.3,
                    result_dst=None,
                    crop_bbox=[0, 270, 1640, 590],
                    nms_thr=4,
                    mask_size=(1, 40, 100)):
    seg_model.eval()
    dataset = data_loader.dataset
    post_processor = CondLanePostProcessor(
        mask_size=mask_size, hm_thr=hm_thr, use_offset=True, nms_thr=nms_thr)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            sub_name = data['img_metas'].data[0][0]['sub_img_name']
            img_shape = data['img_metas'].data[0][0]['img_shape']
            sub_dst_name = sub_name.replace('.jpg', '.lines.txt')
            dst_dir = result_dst + sub_dst_name
            dst_folder = os.path.split(dst_dir)[0]
            mkdir(dst_folder)
            seeds, hm = seg_model(
                return_loss=False, rescale=False, thr=hm_thr, **data)
            downscale = data['img_metas'].data[0][0]['down_scale']
            lanes, seeds = post_processor(seeds, downscale)
            result = adjust_result(
                lanes=lanes,
                crop_bbox=crop_bbox,
                img_shape=img_shape,
                tgt_shape=(590, 1640))
            out_result(result, dst=dst_dir)

        if show is not None and show:
            filename = data['img_metas'].data[0][0]['filename']
            img_vis, img_gt_vis, num_failed = vis_one(result, filename)
            basename = '{}_'.format(num_failed) + sub_name[1:].replace(
                '/', '.')
            dst_show_dir = os.path.join(show, basename)
            mkdir(show)
            cv2.imwrite(dst_show_dir, img_vis)
            dst_gt_dir = os.path.join(show, basename + '.gt.jpg')
            mkdir(show)
            cv2.imwrite(dst_gt_dir, img_gt_vis)

        batch_size = data['img'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()


class DateEnconding(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, np.float32):
            return float(o)


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    if not args.show:
        show_dst = None
    else:
        show_dst = args.show_dst
    if args.show is not None and args.show:
        mkdir(args.show_dst)

    single_gpu_test(
        model,
        data_loader,
        show=show_dst,
        hm_thr=args.hm_thr,
        result_dst=args.result_dst,
        crop_bbox=cfg.crop_bbox,
        nms_thr=cfg.nms_thr,
        mask_size=cfg.mask_size)


if __name__ == '__main__':
    main()