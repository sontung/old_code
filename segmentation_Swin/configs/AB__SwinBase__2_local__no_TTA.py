# section model
norm_cfg = dict(type='BN', requires_grad=True)
num_classes = 5

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# section dataset
img_scale = (1920, 1080)
crop_size = (512, 512)
flip_ratio = 0.25

data_type = 'AirbagHBLab'
data_root = '/media/maihai/Data/0_haims_datasets/Airbag_datasets/datasets/1_mmsegment_format/ab_abr_head_ear'
img_dir   = '/media/maihai/Data/0_haims_datasets/Airbag_datasets/datasets/1_mmsegment_format/ab_abr_head_ear/img_dir'
ann_dir   = '/media/maihai/Data/0_haims_datasets/Airbag_datasets/datasets/1_mmsegment_format/ab_abr_head_ear/ann_dir'

train = '/media/maihai/Data/0_haims_datasets/Airbag_datasets/datasets/1_mmsegment_format/ab_abr_head_ear/train.txt'
val   = '/media/maihai/Data/0_haims_datasets/Airbag_datasets/datasets/1_mmsegment_format/ab_abr_head_ear/val.txt'
test  = '/media/maihai/Data/0_haims_datasets/Airbag_datasets/datasets/1_mmsegment_format/ab_abr_head_ear/test_phase_2.txt'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=data_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', flip_ratio=flip_ratio),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        split=train),
    val=dict(
        type=data_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=img_scale,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split=val),
    test=dict(
        type=data_type,
        data_root=data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=img_scale,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    # dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split=test))

# section misc
log_config = dict(interval=10, hooks=[dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoints/upernet_swin_base_patch4_window7_512x512.pth'
resume_from = '/media/maihai/Data/0_haims_datasets/Airbag_datasets/ckpt_and_results/2_2/latest.pth'
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=16000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=1000, metric='mIoU')
work_dir = '/media/maihai/Data/0_haims_datasets/Airbag_datasets/ckpt_and_results/2_no_TTA'
seed = 0
gpu_ids = range(0, 1)