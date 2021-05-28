import os
import random

import cv2
import numpy as np
from mmdet.utils.general_utils import mkdir, getPathList, path_join

from .custom import CustomDataset
from .pipelines import Compose
from .builder import DATASETS
from .culane_dataset import CulaneDataset


@DATASETS.register_module
class CurvelanesDataset(CulaneDataset):

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        imgname = path_join(self.img_prefix, img_info)
        sub_img_name = img_info
        img_tmp = cv2.imread(imgname)
        ori_shape = img_tmp.shape
        if ori_shape == (1440, 2560, 3):
            img = np.zeros((800, 2560, 3), np.uint8)
            img[:800, :, :] = img_tmp[640:, ...]
            offset_x = 0
            offset_y = -640
        elif ori_shape == (660, 1570, 3):
            img = np.zeros((480, 1570, 3), np.uint8)
            img[:480, :, :] = img_tmp[180:, ...]
            offset_x = 0
            offset_y = -180
        elif ori_shape == (720, 1280, 3):
            img = np.zeros((352, 1280, 3), np.uint8)
            img[:352, :, :] = img_tmp[368:, ...]
            offset_x = 0
            offset_y = -368
        else:
            return None
        img_shape = img.shape
        kps, id_classes, id_instances = self.load_labels(
            idx, offset_x, offset_y)
        results = dict(
            filename=imgname,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=kps,
            id_classes=id_classes,
            id_instances=id_instances,
            img_shape=img_shape,
            ori_shape=ori_shape)

        return self.pipeline(results)

    def prepare_test_img(self, idx):
        imgname = path_join(self.img_prefix, self.img_infos[idx])
        sub_img_name = self.img_infos[idx]
        img_tmp = cv2.imread(imgname)
        ori_shape = img_tmp.shape

        if ori_shape == (1440, 2560, 3):
            img = np.zeros((800, 2560, 3), np.uint8)
            img[:800, :, :] = img_tmp[640:, ...]
            crop_shape = (800, 2560, 3)
            crop_offset = [0, 640]
        elif ori_shape == (660, 1570, 3):
            img = np.zeros((480, 1570, 3), np.uint8)
            crop_shape = (480, 1570, 3)
            img[:480, :, :] = img_tmp[180:, ...]
            crop_offset = [0, 180]
        elif ori_shape == (720, 1280, 3):
            img = np.zeros((352, 1280, 3), np.uint8)
            img[:352, :, :] = img_tmp[368:, ...]
            crop_shape = (352, 1280, 3)
            crop_offset = [0, 368]

        else:
            return None

        results = dict(
            filename=imgname,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=[],
            id_classes=[],
            id_instances=[],
            img_shape=crop_shape,
            ori_shape=ori_shape,
            crop_offset=crop_offset,
            crop_shape=crop_shape)
        return self.pipeline(results)