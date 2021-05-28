import argparse
import os
import cv2
import numpy as np
import mmcv
import torch

from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.utils.general_utils import Timer
from mmdet.models.detectors.condlanenet import CondLanePostProcessor

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

SIZE = (800, 320)

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'checkpoint', default=None, help='test config file path')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    model = build_detector(cfg.model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    img = cv2.imread('./tools/condlanenet/test.jpg')
    img = img[270:, ...]
    img = cv2.resize(img, SIZE)
    mean = np.array([75.3, 76.6, 77.6])
    std = np.array([50.5, 53.8, 54.3])
    img = mmcv.imnormalize(img, mean, std, False)
    x = torch.unsqueeze(torch.from_numpy(img).permute(2, 0, 1), 0)
    model = model.cuda().eval()
    x = x.cuda()

    post_processor = CondLanePostProcessor(mask_size=(1, 40, 100), hm_thr=0.5, seg_thr=0.5)
    # warm up
    for i in range(1000):
        seeds, _ = model.test_inference(x)
        post_processor(seeds, 4)

    with Timer("Elapsed time in all model infernece: %f"):
        for i in range(1000):
            seeds, _ = model.test_inference(x)
            lanes, seeds = post_processor(seeds, 4)


if __name__ == '__main__':
    main()