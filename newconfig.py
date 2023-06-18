# -*- coding = utf-8 -*-
# @Time : 2023/6/17 23:24
# @Author : Happiness
# @File : newconfig.py
# @Software : PyCharm




#####导入工具包

import numpy as np
from PIL import Image

import os.path as osp
from tqdm import tqdm

import mmcv
import mmengine
import matplotlib.pyplot as plt




####载入config配置文件

from mmengine import Config
cfg = Config.fromfile('../../configs/pspnet/pspnet_r50-d8_4xb2-40k_DubaiDataset.py')






#####修改config配置文件
cfg.norm_cfg = dict(type='BN', requires_grad=True) # 只使用GPU时，BN取代SyncBN
cfg.crop_size = (256, 256)
cfg.model.data_preprocessor.size = cfg.crop_size
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head

# 模型 decode/auxiliary 输出头，指定为类别个数
cfg.model.decode_head.num_classes = 6
cfg.model.auxiliary_head.num_classes = 6

cfg.train_dataloader.batch_size = 8


###3修改成绝对路径

cfg.train_dataloader.dataset.data_root='D:/0.dive into pytorch/openmmlab/mmsegmentation/data/Dubai-dataset/'
cfg.val_dataloader.dataset.data_root='D:/0.dive into pytorch/openmmlab/mmsegmentation/data/Dubai-dataset/'
cfg.test_dataloader.dataset.data_root='D:/0.dive into pytorch/openmmlab/mmsegmentation/data/Dubai-dataset/'



cfg.test_dataloader = cfg.val_dataloader

# 结果保存目录
cfg.work_dir = './work_dirs/DubaiDataset'

# 训练迭代次数
cfg.train_cfg.max_iters = 3000
# 评估模型间隔
cfg.train_cfg.val_interval = 400
# 日志记录间隔
cfg.default_hooks.logger.interval = 100
# 模型权重保存间隔
cfg.default_hooks.checkpoint.interval = 1500

# 随机数种子
cfg['randomness'] = dict(seed=0)



####查看完整config配置文件

# print(cfg.pretty_text)



#3####保存config配置文件



cfg.dump('pspnet-DubaiDataset.py')