_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',  # get the base Faster R-CNN model
    '../_base_/datasets/ssdd_detection.py',     # get the base config of the Ship SAR Detection Dataset
    '../_base_/schedules/schedule_ssdd.py',
    '../_base_/default_runtime.py'
]

# optimizer = dict(lr=0.001, momentum=0.9, weight_decay=0.0005)      # set the optimizer Learning Rate
# runner = dict(type='EpochBasedRunner', max_epochs=50)       # set the number of epoch

data = dict(
    samples_per_gpu=2,      # batch size
    workers_per_gpu=2,     #
)


model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
    ),
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
                iou_threshold=0.7,
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
