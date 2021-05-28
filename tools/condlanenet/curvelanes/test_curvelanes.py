import argparse
import os
import numpy as np
import random
import math
import json
import copy
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
from tools.condlanenet.post_process import CurvelanesPostProcessor
from tools.condlanenet.lane_metric import LaneMetricCore
from tools.condlanenet.common import convert_coords_formal, parse_anno, COLORS


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='seg checkpoint file')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--hm_thr', type=float, default=0.5)
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--show_dst',
        default='./work_dirs/curvelanes/watch',
        help='path to save visualized results.')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='whether to compare pred and gt')
    parser.add_argument('--eval_width', type=float, default=224)
    parser.add_argument('--eval_height', type=float, default=224)
    parser.add_argument('--lane_width', type=float, default=5)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def adjust_result(lanes,
                  crop_offset,
                  crop_shape,
                  img_shape,
                  tgt_shape=(590, 1640)):
    h_img, w_img = img_shape[:2]
    ratio_x = crop_shape[1] / w_img
    ratio_y = crop_shape[0] / h_img
    offset_x, offset_y = crop_offset

    results = []
    if lanes is not None:
        for key in range(len(lanes)):
            pts = []
            for pt in lanes[key]['points']:
                pt[0] = float(pt[0] * ratio_x + offset_x)
                pt[1] = float(pt[1] * ratio_y + offset_y)
                pts.append(tuple(pt))
            if len(pts) > 1:
                results.append(pts)
    return results


def vis_one_for_paper(results,
                      filename,
                      result_record,
                      ori_shape,
                      lane_width=11,
                      draw_gt=True):
    pr_list = result_record['pr_list']
    gt_list = result_record['gt_list']
    img = cv2.imread(filename)
    img_ori = copy.deepcopy(img)
    img_gt = copy.deepcopy(img)
    img_pil = PIL.Image.fromarray(img)
    img_gt_pil = PIL.Image.fromarray(img_gt)
    num_failed = pr_list.count(False) + gt_list.count(False)

    annos = parse_anno(filename, formal=False)
    if draw_gt:
        for idx, (anno_lane, _) in enumerate(zip(annos, gt_list)):
            PIL.ImageDraw.Draw(img_gt_pil).line(
                xy=anno_lane, fill=COLORS[idx + 1], width=lane_width)
    for idx, (pred_lane, _) in enumerate(zip(results, pr_list)):
        PIL.ImageDraw.Draw(img_pil).line(
            xy=pred_lane, fill=COLORS[idx + 1], width=lane_width)

    img = np.array(img_pil, dtype=np.uint8)
    img_gt = np.array(img_gt_pil, dtype=np.uint8)
    return img, img_gt, num_failed, img_ori


def single_gpu_test(seg_model,
                    data_loader,
                    show=None,
                    hm_thr=0.3,
                    evaluate=True,
                    eval_width=224,
                    eval_height=224,
                    lane_width=5,
                    mask_size=(1, 40, 100)):
    seg_model.eval()
    dataset = data_loader.dataset
    post_processor = CurvelanesPostProcessor(
        mask_size=mask_size, hm_thr=hm_thr)
    evaluator = LaneMetricCore(
        eval_width=eval_width,
        eval_height=eval_height,
        iou_thresh=0.5,
        lane_width=lane_width)
    prog_bar = mmcv.ProgressBar(len(dataset))
    hm_tp, hm_fp, hm_fn = 0, 0, 0
    out_seeds = []
    for i, data in enumerate(data_loader):

        with torch.no_grad():
            filename = data['img_metas'].data[0][0]['filename']
            sub_name = data['img_metas'].data[0][0]['sub_img_name']
            img_shape = data['img_metas'].data[0][0]['img_shape']
            ori_shape = data['img_metas'].data[0][0]['ori_shape']
            crop_offset = data['img_metas'].data[0][0]['crop_offset']
            crop_shape = data['img_metas'].data[0][0]['crop_shape']

            seeds, hm = seg_model(
                return_loss=False, rescale=False, thr=hm_thr, **data)
            downscale = data['img_metas'].data[0][0]['down_scale']
            lanes, seeds = post_processor(seeds, downscale)

            result = adjust_result(
                lanes=lanes,
                crop_offset=crop_offset,
                crop_shape=crop_shape,
                img_shape=img_shape,
                tgt_shape=ori_shape)
            if evaluate:
                pred = convert_coords_formal(result)
                anno = parse_anno(filename)
                gt_wh = dict(height=ori_shape[0], width=ori_shape[1])
                predict_spec = dict(Lines=pred, Shape=gt_wh)
                target_spec = dict(Lines=anno, Shape=gt_wh)
                evaluator(target_spec, predict_spec)

        if show is not None and show:

            filename = data['img_metas'].data[0][0]['filename']

            img_vis, img_gt_vis, num_failed, img_ori = vis_one_for_paper(
                result,
                filename,
                result_record=evaluator.result_record[-1],
                ori_shape=ori_shape,
                draw_gt=True,
                lane_width=13)
            
            basename = sub_name.replace('/', '.')
            dst_show_dir = os.path.join(show, basename)
            mkdir(show)
            cv2.imwrite(dst_show_dir, img_vis)
            dst_show_gt_dir = os.path.join(show, basename + '.gt.jpg')
            cv2.imwrite(dst_show_gt_dir, img_gt_vis)

        if i % 100 == 0:
            print(evaluator.summary())

        batch_size = data['img'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    print(evaluator.summary())


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
    show_dst = None
    if args.show:
        show_dst = args.show_dst
        mkdir(args.show_dst)

    single_gpu_test(
        seg_model=model,
        data_loader=data_loader,
        show=show_dst,
        hm_thr=args.hm_thr,
        evaluate=args.evaluate,
        eval_width=args.eval_width,
        eval_height=args.eval_height,
        lane_width=args.lane_width,
        mask_size=cfg.mask_size)


if __name__ == '__main__':
    main()