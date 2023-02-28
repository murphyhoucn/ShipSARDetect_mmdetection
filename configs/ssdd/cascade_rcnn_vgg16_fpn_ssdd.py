_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',  # get the base Faster R-CNN model
    '../_base_/datasets/ssdd_detection.py',  # get the base config of the Ship SAR Detection Dataset
    '../_base_/schedules/schedule_ssdd.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='VGG',
        depth=16,
        with_bn=False,
        num_stages=5,
        dilations=(1, 1, 1, 1, 1),
        out_indices=(1, 2, 3, 4),
        frozen_stages=-1,
        bn_eval=False,
        bn_frozen=False,
        ceil_mode=False,
        with_last_pool=False,
    ),
    neck=dict(
        in_channels=[128, 256, 512, 512],
    ),
    rpn_head=dict(
        type='RPNHead',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8, 32, 64, 96],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64])),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=1,
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=1,
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=1,
            ),
        ]
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms=dict(type='nms', iou_threshold=0.5))),

    test_cfg=dict(
        rpn=dict(
            nms=dict(type='nms', iou_threshold=0.5)),
        rcnn=dict(
            nms=dict(type='nms', iou_threshold=0.5))))

