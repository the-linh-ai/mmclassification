_base_ = ['./pipelines/rand_aug.py']

# dataset settings
dataset_type = 'ACDCDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=448),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies={{_base_.rand_increasing_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=7,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(448, -1)),
    dict(type='CenterCrop', crop_size=448),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='data/acdc/rgb_anon/',
        split='train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/acdc/rgb_anon/',
        split='val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='data/acdc/rgb_anon/',
        split='test',
        pipeline=test_pipeline))

evaluation = dict(
    interval=1,
    metric=['accuracy', 'precision', 'recall', 'f1_score', 'support',
            'confusion_matrix'],
    metric_options=dict(topk=1),
    gpu_collect=True)
