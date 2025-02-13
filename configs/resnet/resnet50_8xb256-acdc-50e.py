_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/acdc_bs64.py',
    '../_base_/schedules/acdc_bs256.py',
    '../_base_/default_runtime.py'
]

# Model settings
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        drop_path_rate=0.05,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth',
            prefix='backbone',
        )
    ),
    head=dict(
        num_classes=5,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original',
        )),
    # train_cfg=dict(augments=[
    #     dict(type='BatchMixup', alpha=0.2, num_classes=1000, prob=0.5),
    #     dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    # ])
)

# Dataset settings
sampler = dict(type='RepeatAugSampler')

# Schedule settings
runner = dict(max_epochs=50)
optimizer = dict(
    weight_decay=0.01,
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
)
checkpoint_config = dict(interval=9999)  # no checkpointing
evaluation = dict(save_best="accuracy")
