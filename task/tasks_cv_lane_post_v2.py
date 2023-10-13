from functools import partial
import copy
import os.path as osp
import os
import cv2
import time
from scipy.spatial.distance import cdist
import sys
import pickle
import argparse
import numpy as np
from mmdet.models.detectors.condlanenet import CondLanePostProcessor
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.utils.general_utils import Timer
from mmdet.models.detectors.condlanenet import CondLanePostProcessor
from tools.condlanenet.common import tusimple_convert_formal, COLORS
import warnings
warnings.filterwarnings("ignore")
import mmcv
try:
    import torch
except ImportError as e:
    torch = e 
import PIL
import shutil
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', default='configs/condlanenet/tusimple/tusimple_small_test.py', 
                        help='test config file path')
    parser.add_argument(
        '--checkpoint', default=None, help='test config file path')
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

def project_points2image(points, lidar2img, img_width, img_height, use_weight=False, image=None, init_delta=0 ,init_pit=0, init_yaw=0, init_roll=0):
    """Project points 2 image, and return unique coord in image

    Args:
        points : N*3
    """
    pitch_delta = init_pit / 180 * 3.1415
    yaw_delta = init_yaw / 180 * 3.1415
    roll_delta = init_roll / 180 * 3.1415
    pitch_mat = _pitch_mat(pitch_delta)
    yaw_mat = _yaw_mat(yaw_delta)
    roll_mat = _roll_mat(roll_delta)
    
    image_raw = copy.deepcopy(image)


    pad = np.ones_like(points[:, :1])
    points = np.concatenate([points, pad], axis=-1)
    points = points.dot(pitch_mat.T)
    points = points.dot(roll_mat.T)
    points = points.dot(yaw_mat.T)
    img_points = points.dot(lidar2img.T)
    depth_mask = img_points[:, 2] > 0
    img_points = img_points[depth_mask]
    lidar_points = points[depth_mask]
    img_points[..., :3] /= img_points[..., 2:3]
    img_points = img_points[..., :2]
    img_points = img_points.astype(np.int32)

    mask1 = img_points[:, 0] > 0
    mask2 = img_points[:, 0] < img_width
    mask3 = img_points[:, 1] > 0
    mask4 = img_points[:, 1] < img_height
    mask = mask1 & mask2 & mask3 & mask4

    img_points = img_points[mask]
    lidar_points = lidar_points[mask]
    cld_weight = lidar_points[:,0].copy()
    cld_weight = cld_weight.reshape(-1,1)
    cld_weight = cld_weight * 0.1
    if use_weight:
        return img_points, lidar_points, cld_weight
    else:
        return img_points, lidar_points
    
def adjust_result(lanes, crop_bbox, img_shape):
    h_img, w_img = img_shape[:2]
    ratio_x = (crop_bbox[2] - crop_bbox[0]) / w_img
    ratio_y = (crop_bbox[3] - crop_bbox[1]) / h_img
    offset_x, offset_y = crop_bbox[:2]

    results = []
    scores = []
    if lanes is not None:
        for key in range(len(lanes)):
            pts = []
            for pt in lanes[key]['points']:
                pt[0] = float(pt[0] * ratio_x + offset_x)
                pt[1] = float(pt[1] * ratio_y + offset_y)
                pts.append(tuple(pt))
            if len(pts) > 1:
                results.append(pts)
                scores.append(lanes[key]['score'])
    return results, scores

def cal_delta(lidar2img, img_width=1920, img_height=1080, step=0.1, info=None, thre=[-2,2]):
    '''
    
    '''
    delta_range = thre
    min_cost = 10000
    min_delta = []
    init_roll = 0
    for pitch_row in range(int((delta_range[1] - delta_range[0]) / step)):
        init_pit = pitch_row * step + delta_range[0]
        for yaw_row in range(int((delta_range[1] - delta_range[0]) / step)):
            init_yaw = yaw_row * step + delta_range[0]
            tmp_res = []
            for index, value in enumerate(info):
                img_points, _, cld_wight = project_points2image(
                        points=value['lidar_points'], lidar2img=lidar2img, img_width=1920, img_height=1080, 
                        use_weight=True, init_pit=init_pit, init_yaw=init_yaw, init_roll=init_roll)
                
                cost_sum = cdist(value['line_tensor'], img_points, metric='euclidean')
                cost_min = np.min(cost_sum,axis=0)
                cost_min = cost_min * cld_wight
                cost_ave = np.average(cost_min)
                if cost_ave > 40:
                    break
                tmp_res.append(cost_ave)
            if len(tmp_res) == len(info):
                ave_cost = sum(tmp_res) / max(len(tmp_res), 10e-6)
                if ave_cost < min_cost:
                    min_delta = [round(init_pit, 2), round(init_yaw, 2), round(init_roll, 2)]              
                    min_cost = ave_cost
                    per_line_cost = tmp_res.copy()
                        
    
    if min_delta:
        for idx, value in enumerate(info):
            points, _ = project_points2image(
                                points=value['lidar_points'], lidar2img=lidar2img, img_width=1920, img_height=1080, 
                                init_pit=min_delta[0], init_yaw=min_delta[1], init_roll=min_delta[2])
            value['cost'] = per_line_cost[idx]
            value['fix_points'] = points
   
    return info, min_delta

def com_dis(cv_res, cloud_res, lidarpoints=None, lidar2img=None, img_width=None, img_height=None, thre=200, use_thr=False, H=1920, W=1080, info=None, file_path=None):
    '''
    Args:
        cv_res (list): [[22,2] * num_lanes]
        cloud_res (np.ndarray): num_cloud*2
    '''
    result_list = []
    line_tensor = None
    cv_tensor = None
    cost = []
    N = len(cv_res)
    maxpoints_perlane = max(len(sublist) for sublist in cv_res)
    for i in range(N):
        num_missing_point = maxpoints_perlane - len(cv_res[i])
        if num_missing_point != 0:
            cv_res[i] = cv_res[i] + [cv_res[i][-1]] * num_missing_point

    point_cloud = torch.from_numpy(cloud_res)     
    lines = torch.tensor(cv_res).permute(1,2,0)   

   
    expanded_point_cloud = point_cloud.unsqueeze(1).unsqueeze(-1) 
    expanded_lines = lines.unsqueeze(0) 

    distances = torch.norm(expanded_point_cloud - expanded_lines, dim=2) 
    distances = distances.view(distances.shape[0], -1) 
    nearest_cloud_dis, nearest_cloud_idx = torch.topk(-distances, 1, dim=1)    
    nearest_cloud_dis = -nearest_cloud_dis
    nearest_points_idx = nearest_cloud_idx // N
    nearest_cloud_idx = nearest_cloud_idx % N
    nearest_line_points = lines[nearest_points_idx, :, nearest_cloud_idx]
    nearest_line_points = nearest_line_points.squeeze(1)
    nearest_cloud_res = torch.cat((nearest_cloud_idx, point_cloud, nearest_cloud_dis, nearest_line_points), dim=1)
    
    y_coords = nearest_cloud_res[:,2]
    z = y_coords // 5
    z = (z - 225) / (-25)
    adjust_cloud_dis = nearest_cloud_res[:,3] * z
    adjust_cloud_dis = adjust_cloud_dis.unsqueeze(1)
    nearest_cloud_res = torch.cat((nearest_cloud_res, adjust_cloud_dis, torch.from_numpy(lidarpoints)), dim=1)
    if use_thr:
        indics_to_keep = torch.where(nearest_cloud_res[:,6] < thre)[0]
        nearest_cloud_res = nearest_cloud_res[indics_to_keep]

    
    
    for line_idx in range(N):
        line_tensor = None
        tmp_res = {}
        line_points = lines[:, :, line_idx].tolist()
        sorted(line_points, key=lambda s:s[1])
        tmp_N = len(line_points) // 2
        if tmp_N < 1:
            pass
        tmp_l = len(line_points) % 2
        for t_p in range(tmp_N-1):
            w = np.array([line_points[t_p * 2][0], line_points[t_p * 2 + 1][0], line_points[t_p * 2 + 2][0]])
            h = np.array([line_points[t_p * 2][1], line_points[t_p * 2 + 1][1], line_points[t_p * 2 + 2][1]])
            confficients = np.polyfit(h,w,3)
            fitted_function = np.poly1d(confficients)
            new_pts = []
            for i in range(int(h[0]), int(h[-1])):
                fit_w = fitted_function(i)
                new_pts.append([fit_w, i])
            if line_tensor is not None:
                line_tensor = torch.cat((line_tensor, torch.tensor(new_pts)), dim=0)
            else:
                line_tensor = torch.tensor(new_pts)
        if tmp_l == 0:
            w = np.array([line_points[(tmp_N-2) * 2 + 1][0], line_points[(tmp_N-2) * 2 + 2][0], line_points[(tmp_N-1) * 2 + 1][0]])
            h = np.array([line_points[(tmp_N-2) * 2 + 1][1], line_points[(tmp_N-2) * 2 + 2][1], line_points[(tmp_N-1) * 2 + 1][1]])
            confficients = np.polyfit(h,w,3)
            fitted_function = np.poly1d(confficients)
            new_pts = []
            for i in range(int(h[1]), int(h[-1])):
                fit_w = fitted_function(i)
                new_pts.append([i, fit_w])
            line_tensor = torch.cat((line_tensor, torch.tensor(new_pts)), dim=0)
        target_y = 800
        line_dis = np.abs(np.array(line_tensor)[:,1] - target_y)
        min_dis_index = np.argmin(line_dis)
        line_cam_point = line_tensor[min_dis_index].tolist()

        if cv_tensor is None:
            cv_tensor = line_tensor
        else:
            cv_tensor = torch.cat((cv_tensor, line_tensor), dim=0)
        matching_indices = (nearest_cloud_res[:, 0] == line_idx)
        cloud_coords_keep = nearest_cloud_res[matching_indices, 1:3][:,1] > 648   
        cloud_coords = nearest_cloud_res[matching_indices, 1:3][cloud_coords_keep].tolist()    
        lidar_points = nearest_cloud_res[matching_indices, 7:10][cloud_coords_keep]   
        if len(cloud_coords) < 15:
            cloud_coords.clear()
        if not torch.any(torch.tensor(cloud_coords)) or not torch.any(line_tensor):
            continue
        
        cld_dis = np.abs(np.array(cloud_coords)[:,1] - target_y)
        min_dis_index = np.argmin(cld_dis)
        cld_cam_point = cloud_coords[min_dis_index]
        tmp_res['cld_cam_point'] = cld_cam_point
        tmp_res['line_cam_point'] = line_cam_point
        instance_cld_dis = torch.cdist(line_tensor.float(), torch.tensor(cloud_coords))   
        cloud_dis = nearest_cloud_res[matching_indices, 3].tolist()
        line_point = nearest_cloud_res[matching_indices, 4:6].tolist()
        cloud_real_dis = nearest_cloud_res[matching_indices, 6].tolist()
        tmp_res['file_path'] = file_path
        tmp_res['camera_type'] = file_path.split('/')[-3]
        tmp_res['line_points'] = line_points
        tmp_res['cloud_coords'] = cloud_coords
        tmp_res['cloud_dis'] = cloud_dis
        tmp_res['line_point'] = line_point
        tmp_res['cloud_real_dis'] = cloud_real_dis
        tmp_res['line_tensor'] = line_tensor
        tmp_res['lidar_points'] = lidar_points
        tmp_res['cost'] = None
        result_list.append(tmp_res)

    if not result_list:
        delta_info = delta = None
    else:
        delta_info, delta = cal_delta(lidar2img, img_width=img_width, img_height=img_height, info=result_list)


    return delta_info, delta

def _yaw_mat(delta):
    yaw_mat = np.array([[np.cos(delta), -np.sin(delta), 0, 0],
                        [np.sin(delta), np.cos(delta), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    return yaw_mat

def _pitch_mat(delta):
    pitch_mat = np.array([
        [np.cos(delta), 0, -np.sin(delta), 0],
        [0, 1, 0, 0],
        [np.sin(delta), 0, np.cos(delta), 0],
        [0, 0, 0, 1]
    ])
    return pitch_mat

def _roll_mat(delta):
    roll_mat = np.array([
                            [1, 0, 0, 0],
                            [0, np.cos(delta), -np.sin(delta), 0],
                            [0, np.sin(delta), np.cos(delta), 0],
                            [0, 0, 0, 1],
    ])
    return roll_mat

def fix_one(info, img, cost=None, lane_width=7):
    img_pil = PIL.Image.fromarray(img)
    font = ImageFont.truetype("Arial.ttf",size=30)
    for idx in range(len(info)):
        lane_tuple = [tuple(p) for p in info[idx]['line_points']]
        PIL.ImageDraw.Draw(img_pil).line(
            xy=lane_tuple, fill=COLORS[idx + 1], width=lane_width)
        point_tuple = [tuple(p) for p in info[idx]['fix_points']]
        for i, point in enumerate(point_tuple):
            center = point
            radius = 2
            left_top = (center[0] - radius, center[1] - radius)
            right_bottom = (center[0] + radius, center[1] + radius)
            PIL.ImageDraw.Draw(img_pil).ellipse([left_top, right_bottom], fill=COLORS[idx + 1])
        line_cost = info[idx]['cost']
        text = f"loss:{line_cost:.3f}"
        text_position = (point_tuple[0][0] + 20, lane_tuple[0][1] - 20)
        PIL.ImageDraw.Draw(img_pil).text(text_position, text, fill=COLORS[idx + 1], font=font)
    delta_info = 'pitch_delta:'+str(cost[0])+ ', yaw_delta:'+str(cost[1]) + ', roll_delta:'+str(cost[2])
    delt_position = (50,50)
    PIL.ImageDraw.Draw(img_pil).text(delt_position, delta_info, fill=COLORS[20], font=font)
    img = np.array(img_pil, dtype=np.uint8)
    return img

def vis_one(result, img):
    img_pil = PIL.Image.fromarray(img)
    for i in range(len(result)):
        point_tuple = [tuple(p) for p in result[i]['cloud_coords']]
        for point in point_tuple:
            center = point
            radius = 2
            left_top = (center[0] - radius, center[1] - radius)
            right_bottom = (center[0] + radius, center[1] + radius)
            PIL.ImageDraw.Draw(img_pil).ellipse([left_top, right_bottom], fill=COLORS[i + 1])
    img = np.array(img_pil, dtype=np.uint8)
    return img


def deal_lane_cv2cld(
        model_path: str,
        model_cfg: str,
        cld_pkl_path: str,
        output_dir: str,
        depth_thr: float = 30.0,
        predict_thre: float = 0.5,
        lidar_infer_results: str = None,
)-> str:
    '''Build inference command.
    Args:


    '''

    ckpt_file = model_path
    cfg_file = model_cfg
    cfg = mmcv.Config.fromfile(cfg_file)
    useless_lane = 0
    cfg.model.pretrained = None
    model = build_detector(cfg.model)
    download_img_info_path = osp.join(output_dir, 'download_img_info.pkl')
    with open(download_img_info_path, 'rb') as img_info:
        download_img_info = pickle.load(img_info)
    if ckpt_file:
        load_checkpoint(model, ckpt_file)
    if lidar_infer_results:
        cld_pkl_path = lidar_infer_results['result_file']
    with open(cld_pkl_path, 'rb') as f:
        cld_data = pickle.load(f)
    for i in range(len(cld_data)):
        file_fn_path = str(cld_data[i]['img_filenames'][1])
        cam_type, frame_id = file_fn_path.split('/')[-3:-1]
        pandar_flu2imgs_fn = cld_data[i]['pandar_flu2imgs'][1]
        point_coords = cld_data[i]['point_coords']
        prediction_cls = cld_data[i]['prediction_cls_name']

        line_mask = prediction_cls == 'line'
        depth_mask = point_coords[:,0] < depth_thr
        filter_mask = line_mask & depth_mask
        point_coords_filtered = point_coords[filter_mask]

        cam_info_i = [index for index in range(len(download_img_info)) if str(download_img_info[index]['id']) == frame_id][0]
        img_fn_path = download_img_info[cam_info_i]['camera_front_narrow_storage_url']
        sub_name = img_fn_path.split('/')[-1]
        fn_img = cv2.imread(img_fn_path)
        H, W, _ = fn_img.shape
        if H == 2160 and W == 3840:
            img_scale = 0.5
            fn_img = cv2.resize(fn_img, (0,0), fx=img_scale, fy=img_scale)
        cld_points, lidar_points = project_points2image(
            points=point_coords_filtered, lidar2img=pandar_flu2imgs_fn, 
            img_width=W, img_height=H, image=fn_img)
        img = fn_img[160:,...]
        img = cv2.resize(img, (800,320))
        mean = np.array([75.3, 76.6, 77.6])
        std = np.array([50.5, 53.8, 54.3])
        img = mmcv.imnormalize(img, mean, std, False)
        img = torch.unsqueeze(torch.from_numpy(img).permute(2, 0, 1), 0)
        model.cuda().eval()
        img = img.cuda()
        crop_bbox = (0, 160, 1920, 1080)
        post_processor = CondLanePostProcessor(mask_size=(1, 40, 100), hm_thr=0.5, seg_thr=0.5)
        seeds, _ = model.test_inference(img=img, thr=predict_thre)
        lanes, seeds = post_processor(seeds, 8)
        tmp_result, scores = adjust_result(lanes=lanes, crop_bbox=crop_bbox, img_shape=(320,800,3))
        result = []
        for index in range(len(tmp_result)):
            w = np.array([tmp_result[index][i][0] for i in range(len(tmp_result[index]))])
            h = np.array([tmp_result[index][i][1] for i in range(len(tmp_result[index]))])
            confficients = np.polyfit(w, h, 2)
            estimated_h = np.polyval(confficients, w)
            loss = np.sqrt(np.mean((estimated_h - h) ** 2))
            if loss < 30:
                result.append(tmp_result[index])

        if len(result) == 0 and len(tmp_result) > len(result):
            useless_lane += 1

        if len(result) > 1 and len(cld_points):
            res_dis, cost = com_dis(
                result, cld_points, lidarpoints=lidar_points, lidar2img=pandar_flu2imgs_fn, 
                img_width=W, img_height=H, use_thr=True, thre=500, file_path=img_fn_path)
            if not res_dis or 'fix_points' not in res_dis[0]:
                continue
            if len(res_dis) > 1:
                img_vis = vis_one(res_dis, fn_img)
                img_fix = fix_one(res_dis, fn_img, cost)
                res_info = sub_name+ ', pitch_delta:'+str(cost[0])+ ', yaw_delta:'+str(cost[1]) + ', roll_delta:'+str(cost[2]) + '\n'
                # debug
                save_path = osp.join(output_dir,"vis_output" ,sub_name)
                fix_subname = sub_name.replace('.jpg', '_fix.jpg')
                fix_save_path = osp.join(output_dir, "fix_output" , fix_subname)
                save_info_path = osp.join(output_dir, "res_info")
                os.makedirs(save_info_path, exist_ok=True)
                with open(osp.join(save_info_path, "res_info.txt"), 'a') as res_:
                    res_.write(res_info)
                res_.close()
                os.makedirs(osp.dirname(save_path), exist_ok=True)
                os.makedirs(osp.dirname(fix_save_path), exist_ok=True)
                cv2.imwrite(save_path, img_vis)
                cv2.imwrite(fix_save_path, img_fix)
                cache_dir = osp.dirname(img_fn_path)
            else:
                pass    
        else:
            continue    
        
if __name__ == "__main__":
    model_path = sys.argv[1]
    model_cfg = sys.argv[2]
    cld_pkl_path = sys.argv[3]
    output_dir = sys.argv[4]
    depth_thr = float(sys.argv[5])
    predict_thre = float(sys.argv[6])
    deal_lane_cv2cld(model_path, model_cfg, cld_pkl_path, output_dir,
                     depth_thr, predict_thre)
