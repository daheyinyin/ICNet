"""train ICNet and get checkpoint files."""
import os
import sys
import logging
import argparse
import ast
import yaml
import mindspore.nn as nn
from mindspore import Model
from mindspore import context
from mindspore import set_seed
from mindspore.context import ParallelMode
from mindspore.communication import init
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import TimeMonitor
from mindspore.communication.management import get_group_size, get_rank


parser = argparse.ArgumentParser(description="ICNet Evaluation")
parser.add_argument("--project_path", type=str, help="project_path")
parser.add_argument('--device_target', type=str, default='Ascend',
                    help='device target, Ascend or GPU (Default: Ascend)')
parser.add_argument('--device_id', type=int, default=0, help='device id')
parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                    help="Run distribute, default is false.")
args_opt = parser.parse_args()

def train_net():
    """train"""
    set_seed(1234)

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, save_graphs=False)
    if args_opt.run_distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          device_num=device_num,
                                          gradients_mean=True)
        if args_opt.device_target == 'Ascend':
            context.set_context(device_id=args_opt.device_id)
    else:
        device_num = 1
        rank_id = 0
        context.set_context(device_id=args_opt.device_id)

    prefix = 'cityscapes-2975.mindrecord'
    mindrecord_dir = cfg['train']["mindrecord_dir"]
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    dataset = create_icnet_dataset(mindrecord_file, batch_size=cfg['train']["train_batch_size_percard"],
                                   device_num=device_num, rank_id=rank_id)

    train_data_size = dataset.get_dataset_size()
    print("data_size", train_data_size)
    epoch = cfg["train"]["epochs"]
    if device_num == 1 or args_opt.device_target == 'GPU':
        network = ICNetdc(pretrained_path=cfg["train"]["pretrained_model_path"], norm_layer=nn.BatchNorm2d)
    else:
        network = ICNetdc(pretrained_path=cfg["train"]["pretrained_model_path"])

    iters_per_epoch = train_data_size
    total_train_steps = iters_per_epoch * epoch
    base_lr = cfg["optimizer"]["init_lr"]
    iter_lr = poly_lr(base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    optim = nn.SGD(params=network.trainable_params(), learning_rate=iter_lr, momentum=cfg["optimizer"]["momentum"],
                   weight_decay=cfg["optimizer"]["weight_decay"])

    model = Model(network, optimizer=optim, metrics=None)

    config_ck_train = CheckpointConfig(save_checkpoint_steps=iters_per_epoch * cfg["train"]["save_checkpoint_epochs"],
                                       keep_checkpoint_max=cfg["train"]["keep_checkpoint_max"])
    ckpoint_cb_train = ModelCheckpoint(prefix='ICNet',
                                       directory=os.path.join(args_opt.project_path, 'ckpt' + str(rank_id)),
                                       config=config_ck_train)
    time_cb_train = TimeMonitor(data_size=dataset.get_dataset_size())
    loss_cb_train = LossMonitor()
    print("train begins------------------------------")
    model.train(epoch=epoch, train_dataset=dataset, callbacks=[ckpoint_cb_train, loss_cb_train, time_cb_train],
                dataset_sink_mode=True)
    print("End of the training------------------------------")

if __name__ == '__main__':
    # Set config file
    sys.path.append(args_opt.project_path)
    from src.cityscapes_mindrecord import create_icnet_dataset
    from src.models.icnet_dc import ICNetdc
    from src.lr_scheduler import poly_lr
    config_file = "src/model_utils/icnet.yaml"
    config_path = os.path.join(args_opt.project_path, config_file)
    with open(config_path, "r") as yaml_file:
        cfg = yaml.safe_load(yaml_file.read())
    logging.basicConfig(level=logging.INFO)
    train_net()
