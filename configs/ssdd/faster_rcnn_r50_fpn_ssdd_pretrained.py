_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',  # get the base Faster R-CNN model
    '../_base_/datasets/ssdd_detection.py',     # get the base config of the Ship SAR Detection Dataset
    '../_base_/schedules/schedule_ssdd.py',
    '../_base_/default_runtime.py'
]


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
