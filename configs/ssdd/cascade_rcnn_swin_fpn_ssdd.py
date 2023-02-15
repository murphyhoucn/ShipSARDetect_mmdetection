_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',  # get the base Faster R-CNN model
    '../_base_/datasets/ssdd_detection.py',  # get the base config of the Ship SAR Detection Dataset
    '../_base_/schedules/schedule_ssdd.py',
    '../_base_/default_runtime.py'
]

# optimizer = dict(lr=0.001, momentum=0., weight_decay=0.)  # set the optimizer Learning Rate
# runner = dict(type='EpochBasedRunner', max_epochs=50)  # set the number of epoch

data = dict(
    samples_per_gpu=2,      # batch size
    workers_per_gpu=2,     # number of CPU core to use
)

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
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
