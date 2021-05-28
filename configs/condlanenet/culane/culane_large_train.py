"""
    config file of the large version of CondLaneNet for culane
"""
# global settings
dataset_type = 'CulaneDataset'
data_root = "/disk1/zhouyang/dataset/culane"
mask_down_scale = 4
hm_down_scale = 16
num_lane_classes = 1
line_width = 3
radius = 6
nms_thr = 4
batch_size = 4
img_norm_cfg = dict(
    mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3], to_rgb=False)
ori_scale = (1640, 590)  # for culane
crop_bbox = [0, 270, 1640, 590]
img_scale = (800, 320)
mask_size = (1, 80, 200)

train_cfg = dict(out_scale=mask_down_scale)
test_cfg = dict(out_scale=mask_down_scale)

# model settings
model = dict(
    type='CondLaneNet',
    pretrained='torchvision://resnet101',
    train_cfg=train_cfg,
    test_cfg=test_cfg,
    num_classes=1,
    backbone=dict(
        type='ResNet',
        depth=101,
        strides=(1, 2, 2, 2),
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='TransConvFPN',
        in_channels=[256, 512, 1024, 256],
        out_channels=64,
        num_outs=4,
        trans_idx=-1,
        trans_cfg=dict(
            in_dim=2048,
            attn_in_dims=[2048, 256],
            attn_out_dims=[256, 256],
            strides=[1, 1],
            ratios=[4, 4],
            pos_shape=(batch_size, 10, 25),
        ),
    ),
    head=dict(
        type='CondLaneHead',
        heads=dict(hm=num_lane_classes),
        in_channels=(64, ),
        num_classes=num_lane_classes,
        head_channels=64,
        head_layers=1,
        disable_coords=False,
        branch_in_channels=64,
        branch_channels=64,
        branch_out_channels=64,
        reg_branch_channels=64,
        branch_num_conv=1,
        hm_idx=2,
        mask_idx=0,
        compute_locations_pre=True,
        location_configs=dict(size=(batch_size, 1, 80, 200), device='cuda:0')),
    loss_weights=dict(
        hm_weight=1,
        kps_weight=0.4,
        row_weight=1.,
        range_weight=1.,
    ),
)

train_compose = dict(bboxes=False, keypoints=True, masks=False)

# data pipeline settings
train_al_pipeline = [
    dict(type='Compose', params=train_compose),
    dict(
        type='Crop',
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=(-10, 10),
                sat_shift_limit=(-15, 15),
                val_shift_limit=(-10, 10),
                p=1.0),
        ],
        p=0.7),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.2),
    dict(type='RandomBrightness', limit=0.2, p=0.6),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.1,
        scale_limit=(-0.2, 0.2),
        rotate_limit=10,
        border_mode=0,
        p=0.6),
    dict(
        type='RandomResizedCrop',
        height=img_scale[1],
        width=img_scale[0],
        scale=(0.8, 1.2),
        ratio=(1.7, 2.7),
        p=0.6),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
]

val_al_pipeline = [
    dict(type='Compose', params=train_compose),
    dict(
        type='Crop',
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
]

train_pipeline = [
    dict(type='albumentation', pipelines=train_al_pipeline),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectLane',
        down_scale=mask_down_scale,
        hm_down_scale=hm_down_scale,
        max_mask_sample=5,
        line_width=line_width,
        radius=radius,
        keys=['img', 'gt_hm'],
        meta_keys=[
            'filename', 'sub_img_name', 'gt_masks', 'mask_shape', 'hm_shape',
            'ori_shape', 'img_shape', 'down_scale', 'hm_down_scale',
            'img_norm_cfg', 'gt_points'
        ]),
]

val_pipeline = [
    dict(type='albumentation', pipelines=val_al_pipeline),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectLane',
        down_scale=mask_down_scale,
        hm_down_scale=hm_down_scale,
        radius=radius,
        keys=['img', 'gt_hm'],
        meta_keys=[
            'filename', 'sub_img_name', 'gt_masks', 'mask_shape', 'hm_shape',
            'ori_shape', 'img_shape', 'down_scale', 'hm_down_scale',
            'img_norm_cfg', 'gt_points'
        ]),
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + '/list/train.txt',
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + '/list/test.txt',
        pipeline=val_pipeline,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + '/list/test.txt',
        test_suffix='.jpg',
        pipeline=val_pipeline,
        test_mode=True,
    ))

# optimizer
optimizer = dict(type='Adam', lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[8, 14])

# runtime settings
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

total_epochs = 16
device_ids = "0,1"
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/exps/culane/large'
load_from = None
resume_from = None
workflow = [('train', 200), ('val', 1)]