# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'hrnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.EXTRA = CN(new_allowed=True)

# DATASET related params
_C.DATA = CN()
_C.DATA.data_root = ''
_C.DATA.train_list = ''
_C.DATA.test_list = ''
_C.DATA.val_list = ''
_C.DATA.classes = 2
_C.DATA.model_name = 'DLA34MTLFusion'

# training
_C.TRAIN = CN()

_C.TRAIN.mtl = True
_C.TRAIN.dataset = 'change'
_C.TRAIN.arch = 'linknet34'
_C.TRAIN.normalize_type = 'Mean'
_C.TRAIN.mean = [74.67876893, 75.08650789, 75.56897547]
_C.TRAIN.std = [46.56804672, 48.34922229, 50.03066040]
_C.TRAIN.train_h = 256
_C.TRAIN.train_w = 256
_C.TRAIN.ignore_label = 255
_C.TRAIN.aux_weight = 4
_C.TRAIN.train_gpu = [0]
_C.TRAIN.workers = 16
_C.TRAIN.batch_size = 32
_C.TRAIN.batch_size_val = 4
_C.TRAIN.base_lr = 0.01
_C.TRAIN.epochs = 150
_C.TRAIN.start_epoch = 0
_C.TRAIN.power = 0.9
_C.TRAIN.momentum = 0.9
_C.TRAIN.weight_decay = 0.0005
_C.TRAIN.manual_seed = 7
_C.TRAIN.print_freq = 500
_C.TRAIN.save_freq = 1
_C.TRAIN.save_path = 'exp/spacenet/dla34mtl/model'
_C.TRAIN.resume = ''
_C.TRAIN.valuate = True
_C.TRAIN.lr_step = 0.1
_C.TRAIN.milestones = [60, 90, 110, 130]
_C.TRAIN.cls_dim = (32, 32, 36)
_C.TRAIN.loss = 'bcedice'
_C.TRAIN.optimizer = 'Adam'
_C.TRAIN.weight = ''
_C.TRAIN.episode = 0
_C.TRAIN.threshold = 0.0

# testing
_C.TEST = CN()
_C.TEST.pad = 11
_C.TEST.split = 'val'
_C.TEST.val_h = 256
_C.TEST.val_w = 256
_C.TEST.test_h = 256
_C.TEST.test_w = 256
_C.TEST.has_prediction = False
_C.TEST.save_folder = 'exp/spacenet/dla34mtl/result/'
_C.TEST.batch_size_test = 16
_C.TEST.test_weight = ''



def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)