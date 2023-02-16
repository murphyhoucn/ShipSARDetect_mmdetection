_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',  # get the base Faster R-CNN model
    '../_base_/datasets/ssdd_detection.py',  # get the base config of the Ship SAR Detection Dataset
    '../_base_/schedules/schedule_ssdd.py',
    '../_base_/default_runtime.py'
]

# optimizer = dict(lr=0.001, momentum=0., weight_decay=0.)  # set the optimizer Learning Rate
# runner = dict(type='EpochBasedRunner', max_epochs=50)  # set the number of epoch

data = dict(
    samples_per_gpu=4,      # batch size
    workers_per_gpu=2,     # number of CPU core to use
)

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
    )
)
