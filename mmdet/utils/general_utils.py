import os
import time
import numpy as np
import cv2


def getPathList(path, suffix='png'):
    if (path[-1] != '/') & (path[-1] != '\\'):
        path = path + '/'
    pathlist = list()
    g = os.walk(path)
    for p, d, filelist in g:
        for filename in filelist:
            if filename.endswith(suffix):
                pathlist.append(os.path.join(p, filename))
    return pathlist

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    if not os.path.isdir(path):
        os.mkdir(path)

def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = (mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb))
        img = (np.clip(img * 255, a_min=0, a_max=255)).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs

def path_join(root, name):
    if root == '':
        return name
    if name[0] == '/':
        return os.path.join(root, name[1:])
    else:
        return os.path.join(root, name)



class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))