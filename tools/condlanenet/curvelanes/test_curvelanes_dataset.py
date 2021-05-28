import argparse
import os

import mmcv
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils.general_utils import mkdir

from tools.condlanenet.common import COLORS


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', required=True,
                        help='test config file path')
    parser.add_argument('--show', required=True, help='show results')
    parser.add_argument('--max_show_num', type=int, default=50, help='show results')
    args = parser.parse_args()
    return args

def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros([h, w, 3], dtype=np.uint8)
    for i in range(np.max(mask)+1):
        rgb[mask == i] = COLORS[i]
    return rgb

def vis_one(data):
    # image
    img = data['img'].data[0].detach().cpu().numpy()[0, :, :, :]
    norm_cfg = data['img_metas'].data[0][0]['img_norm_cfg']
    downscale = data['img_metas'].data[0][0]['down_scale']
    hm_downscale = data['img_metas'].data[0][0]['hm_down_scale']
    img = img.transpose(1, 2, 0)
    img = (img * norm_cfg['std']) + norm_cfg['mean']
    img = img.astype(np.uint8)
    # hm
    gt_hm = data['gt_hm'].data[0].detach().cpu().numpy()[
        0, :, :, :] * 255
    vis_hm = np.zeros_like(gt_hm[0])
    for i in range(gt_hm.shape[0]):
        vis_hm += gt_hm[i, :, :]

    gt_masks = data['img_metas'].data[0][0]['gt_masks']
    vis_img = np.zeros(img.shape, np.uint8)
    vis_img[:] = img[:]
    for i, gt_info in enumerate(gt_masks):
        points = gt_info['points']
        mask_infos = gt_info['gt_masks']
        for color_idx, mask_info in enumerate(mask_infos):
            row = mask_info['row']
            row_range = mask_info['range']
            for coord_y, (coord_x, valid) in enumerate(zip(row, row_range[0])):
                if valid:
                    coord_y *= downscale
                    coord_x *= downscale
                    coord_x = int(coord_x)
                    coord_y = int(coord_y)
                    cv2.circle(vis_img, (coord_x, coord_y), 3, color=COLORS[color_idx+1], thickness=-1)
            points = mask_info['points']
            for p in points:
                cv2.circle(vis_img, (hm_downscale*p[0], hm_downscale*p[1]), 3, COLORS[1], -1)
                cv2.circle(vis_img, (hm_downscale*p[0], hm_downscale*p[1]), 1, (0,0,0), -1)
            img = vis_img
    return img, vis_hm

def main():
    args = parse_args()
    mkdir(args.show)
    # build the dataloader
    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data['workers_per_gpu'],
        dist=False,
        shuffle=False)
    for index, data in tqdm(enumerate(data_loader)):
        file_name = data['img_metas'].data[0][0]['filename']
        save_name = os.path.splitext(os.path.basename(file_name))[0]

        print(index, file_name)
        vis_img, vis_hm = vis_one(data)
        vis_img_dir = os.path.join(args.show, '{}_img.png'.format(save_name))
        vis_hm_dir = os.path.join(args.show, '{}_hm.png'.format(save_name))
        cv2.imwrite(vis_img_dir, vis_img)
        cv2.imwrite(vis_hm_dir, vis_hm)
        if index >= args.max_show_num:
            break


if __name__ == '__main__':
    main()