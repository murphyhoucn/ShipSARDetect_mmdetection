_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',  # get the base Faster R-CNN model
    '../_base_/datasets/ssdd_detection.py',  # get the base config of the Ship SAR Detection Dataset
    '../_base_/schedules/schedule_ssdd.py',
    '../_base_/default_runtime.py'
]


model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=672,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
    ),
    neck=dict(
        in_channels=[96, 192, 384, 768]
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
