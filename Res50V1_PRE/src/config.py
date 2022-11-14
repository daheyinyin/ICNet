
# ============================================================================
"""
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed
# config optimizer for resnet50, imagenet2012. Momentum is default, Thor is optional.
cfg = ed({
    'optimizer': 'Momentum',
    })

config1 = ed({
    "class_num": 1001,
    "batch_size": 256,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 150,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 5,
    "decay_mode": "linear",
    "save_checkpoint_path": "./checkpoints",
    "hold_epochs": 0,
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0.8,
    "lr_end": 0.0
})
