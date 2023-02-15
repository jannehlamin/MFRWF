import torch
from torch import nn

from Models.config import config, update_config
from ..extensions.sync_batchnorm import SynchronizedBatchNorm2d

BatchNorm2d = SynchronizedBatchNorm2d

if torch.__version__.startswith('0'):
    # from .sync_bn.inplace_abn.bn import InPlaceABNSync
    if len(config.GPUS) > 1:
        BatchNorm2d = BatchNorm2d
    else:
        BatchNorm2d = nn.BatchNorm2d
    BatchNorm2d_class = BatchNorm2d
    relu_inplace = False
else:
    if len(config.GPUS) > 1:
        BatchNorm2d_class = BatchNorm2d
    else:
        BatchNorm2d_class = BatchNorm2d = nn.BatchNorm2d
    relu_inplace = True