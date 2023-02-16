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
