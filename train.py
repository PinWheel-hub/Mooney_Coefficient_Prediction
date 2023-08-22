import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from nets.net import NyNetwork
from torch.utils.data import DataLoader

from utils.callbacks import LossHistory
from utils.dataloader import MenniDataset
from utils.utils import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda = True
    model_path = ""
    Init_Epoch = 0
    End_Epoch = 100
    batch_size = 64
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01

    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   weight_decay    权值衰减，可防止过拟合
    # ------------------------------------------------------------------#
    optimizer_type = "adam"
    weight_decay = 5e-4

    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    # ------------------------------------------------------------------#
    lr_decay_type = "cos"

    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    # ------------------------------------------------------------------#
    save_period = 1

    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------------------#
    num_workers = 4
    model = NyNetwork()
    weights_init(model)

    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    loss_fn = nn.MSELoss()
    loss_history = LossHistory("logs/", model)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    train_json = 'dataset/my_train.txt'
    val_json = 'dataset/my_val.txt'

    import json
    with open(train_json, 'r') as f:
        num_train = sum(1 for line in f)
    with open(val_json, 'r') as f:
        num_val = sum(1 for line in f)

    # -------------------------------------------------------------------#
    #   判断当前batch_size与64的差别，自适应调整学习率
    # -------------------------------------------------------------------#
    nbs = 64
    Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
    Min_lr_fit = max(batch_size / nbs * Min_lr, 1e-6)

    # ---------------------------------------#
    #   根据optimizer_type选择优化器
    # ---------------------------------------#
    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit)
    }[optimizer_type]

    # ---------------------------------------#
    #   判断每一个世代的长度
    # ---------------------------------------#
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
    
    # ---------------------------------------#
    #   获得学习率下降的公式
    # ---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, End_Epoch)

    # ---------------------------------------#
    #   构建数据集加载器。
    # ---------------------------------------#
    train_dataset = MenniDataset(train_json, epoch_length=End_Epoch, train=True)
    val_dataset = MenniDataset(val_json, epoch_length=End_Epoch, train=False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True)

    # ---------------------------------------#
    #   开始模型训练
    # ---------------------------------------#
    for epoch in range(Init_Epoch, End_Epoch):

        gen.dataset.epoch_now = epoch
        gen_val.dataset.epoch_now = epoch

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(model_train, model, loss_fn, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen,
                      gen_val, End_Epoch, Cuda, save_period)
