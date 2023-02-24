_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',  # get the base Faster R-CNN model
    '../_base_/datasets/ssdd_detection.py',     # get the base config of the Ship SAR Detection Dataset
    '../_base_/schedules/schedule_ssdd.py',
    '../_base_/default_runtime.py'
]

# ssdd dataset settings
classes = ('ship',)  # only one classe in the dataset
dataset_type = 'CocoDataset'  # dataset is in the COCO format
data_root = 'data/ssdd/'  # root dir of the dataset
img_norm_cfg = dict(
    mean=[41.106, 41.106, 41.106], std=[43.288, 43.288, 43.288], to_rgb=False)       # normalisation of images

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(500, 350), keep_ratio=False),  # resize images
    # do a random flip with the probability flip_ratio
    dict(type='RandomFlip', flip_ratio=[0.25, 0.25, 0.25], direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='Normalize', **img_norm_cfg),         # normalization with the find values
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(500, 350),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field 'classes'
        classes=classes,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/train',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field 'classes'
        classes=classes,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/test',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field 'classes'
        classes=classes,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/test',
        pipeline=test_pipeline,
    ))

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            # explicitly over-write all the `num_classes` field from default 80 to 1.
            num_classes=1,
        ),
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms=dict(
                type='nms',
                iou_threshold=0.5,
            )
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms=dict(
                type='nms',
                iou_threshold=0.5,
            )
        ),
    )
)
