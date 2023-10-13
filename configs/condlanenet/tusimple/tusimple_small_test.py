"""
    config file of the small version of CondLaneNet for tusimple
"""
# global settings
dataset_type = 'TuSimpleDataset'
data_root = ""
test_mode = False
mask_down_scale = 8
hm_down_scale = 16
mask_size = (1, 40, 100)
line_width = 3
radius = 6
nms_thr = 2
num_lane_classes = 1
batch_size = 1
img_norm_cfg = dict(
    mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3], to_rgb=False)
img_scale = (800, 320)
ori_scale = (1280, 720)  # for tusimple
crop_bbox = [0, 160, 1280, 720]
train_cfg = dict(out_scale=mask_down_scale)
test_cfg = dict(out_scale=mask_down_scale)

# model settings
model = dict(
    type='CondLaneNet',
    pretrained='torchvision://resnet18',
    train_cfg=train_cfg,
    test_cfg=test_cfg,
    num_classes=num_lane_classes,
    backbone=dict(
        type='ResNet',
        depth=18,
        strides=(1, 2, 2, 2),
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='TransConvFPN',
        in_channels=[128, 256, 64],
        out_channels=64,
        num_outs=4,
        trans_idx=-1,
        trans_cfg=dict(
            in_dim=512,
            attn_in_dims=[512, 64],
            attn_out_dims=[64, 64],
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
        hm_idx=1,
        mask_idx=0,
        compute_locations_pre=True,
        location_configs=dict(size=(batch_size, 1, 40, 100), device='cuda:0')),
    loss_weights=dict(
        hm_weight=1,
        kps_weight=0.4,
        row_weight=1.,
        range_weight=1.,
    ),
)

train_compose = dict(bboxes=False, keypoints=True, masks=False)

#data pipeline settings
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
            'img_norm_cfg', 'gt_points', 'h_samples', 'img_info'
        ]),
]

data = dict(
    samples_per_gpu=
    batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=[
            data_root + '/label_data_0313.json',
            data_root + '/label_data_0531.json',
            data_root + '/label_data_0601.json'
        ],
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=[data_root + '/test_baseline.json'],
        pipeline=val_pipeline,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=[data_root + '/test_label.json'],
        test_suffix='.jpg',
        pipeline=val_pipeline,
        test_mode=True,
    ))

# optimizer
optimizer = dict(type='Adam', lr=2.5e-4, betas=(0.9, 0.999), eps=1e-8)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=10,
    warmup_ratio=1.0 / 3,
    step=[40, 60])

# runtime settings
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

total_epochs = 70
device_ids = "0,1"
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/tusimple/small'
load_from = None
resume_from = None
workflow = [('train', 200), ('val', 1)]
