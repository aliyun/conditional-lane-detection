import glob
import os
import json

import cv2
import numpy as np
from mmdet.utils.general_utils import mkdir, getPathList, path_join

from .custom import CustomDataset
from .pipelines import Compose
from .builder import DATASETS


@DATASETS.register_module
class TuSimpleDataset(CustomDataset):

    def __init__(self,
                 data_root,
                 data_list,
                 pipeline,
                 test_mode=False,
                 test_suffix='png'):
        self.img_prefix = data_root
        self.test_suffix = test_suffix
        self.test_mode = test_mode
        
        # read image list
        self.img_infos = self.parser_datalist(data_list)
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        self.pipeline = Compose(pipeline)
    
    def parser_datalist(self, data_list):
        img_infos = []
        for anno_file in data_list:
            json_gt = [json.loads(line) for line in open(anno_file)]
            img_infos += json_gt
        return img_infos
    
    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.img_infos)

    def prepare_train_img(self, idx):
        sub_img_name = self.img_infos[idx]['raw_file']
        imgname = path_join(self.img_prefix, sub_img_name)
        img = cv2.imread(imgname)
        ori_shape = img.shape
        offset_x = 0
        offset_y = 0
        img_shape = img.shape
        kps, id_classes, id_instances = self.load_labels(idx, offset_x, offset_y)
        results = dict(filename=imgname,
                       sub_img_name=sub_img_name,
                       img=img,
                       gt_points=kps,
                       id_classes=id_classes,
                       id_instances=id_instances,
                       img_shape=img_shape,
                       ori_shape=ori_shape)
        return self.pipeline(results)


    def prepare_test_img(self, idx):
        sub_img_name = self.img_infos[idx]['raw_file']
        imgname = path_join(self.img_prefix, sub_img_name)
        h_samples = self.img_infos[idx]['h_samples']
        img = cv2.imread(imgname)
        ori_shape = img.shape
        results = dict(filename=imgname,
                       sub_img_name=sub_img_name,
                       img=img,
                       gt_points=[],
                       id_classes=[],
                       id_instances=[],
                       img_shape=ori_shape,
                       ori_shape=ori_shape,
                       crop_offset=0,
                       h_samples=h_samples,
                       img_info=self.img_infos[idx])
        return self.pipeline(results)

    def load_labels(self, idx, offset_x, offset_y):
        shapes = []
        for lane in self.img_infos[idx]['lanes']:
            coords = []
            for coord_x, coord_y in zip(lane, self.img_infos[idx]['h_samples']):
                if coord_x >= 0:
                    coord_x = float(coord_x)
                    coord_y = float(coord_y)
                    coord_x += offset_x
                    coord_y += offset_y
                    coords.append(coord_x)
                    coords.append(coord_y)
            if len(coords) > 3:
                shapes.append(coords)
        id_classes = [1 for i in range(len(shapes))]
        id_instances = [i+1 for i in range(len(shapes))]
        return shapes, id_classes, id_instances
