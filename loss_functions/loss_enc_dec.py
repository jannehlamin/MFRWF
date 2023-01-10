import torch
import torch as t
import torch.nn as nn
from torch.nn import functional as F
from loss_functions.losses.dice_loss import DiceLoss


class SegmentationLosses(object):
    def __init__(self, n_classes=3, weight=None, size_average=True, batch_average=True, ignore_index=0, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.nclasses = n_classes

    def build_loss(self, mode='ce'):
        """Choices: ['dice', 'ce' or 'focal', 'log']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice':
            return self.dice_loss
        elif mode == 'log':
            return self.log_cosh_dice_loss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n
        return loss

    def dice_loss(self, logit, target):
        criterion = self.dice_loss_llJ
        if self.cuda:
           criterion = criterion.cuda()
        loss = criterion(logit, target.long())
        return loss

    def log_cosh_dice_loss(self, y_pred, y_true):

        x = self.dice_loss_llJ(y_pred, y_true.long())
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)

    def dice_loss_llJ(self, y_pred, y_true):
        Dice = DiceLoss(class_weight=self.weight, ignore_index=self.ignore_index)
        loss = Dice(y_pred, y_true)
        return loss


if __name__ == "__main__":

    loss = SegmentationLosses(cuda=False)
    a = torch.rand(2, 3, 512, 512).cuda()
    b = torch.rand(2, 512, 512).cuda()
    # print(loss.dice_loss(a, b).item())
    print(loss.log_cosh_dice_loss(a, b).item())
    # print(loss.dice_loss_n(a, b)
    # print(F.one_hot(b, num_classes=3).permute(0, 3, 1, 2).shape)
    # print(loss.dice_loss(a, b.long()).item())
    # true_1_hot = torch.eye(3)[b.squeeze(1)]
    # b = true_1_hot.permute(0, 3, 1, 2).float()
    # print(b.shape)
