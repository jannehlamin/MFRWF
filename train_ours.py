import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from dataloaders.data_util import make_data_loader_nostream
from models.backbone.extensions.sync_batchnorm import patch_replication_callback
from models.backbone.utils_1.utils import FullModel_nostream
from models.nets.hybrid_baseline import OHybridCR
from mypath import Path
from datetime import datetime
# -------------------------------------------------------------------
from loss_functions.calculate_weights import calculate_weigths_labels
from loss_functions.loss import SegmentationLosses
from loss_functions.saver import Saver
from loss_functions.summaries import TensorboardSummary
from loss_functions.lr_scheduler import LR_Scheduler
from loss_functions.metrics import Evaluator
torch.cuda.set_device(0)

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader_nostream(args, **kwargs)

        distributed = args.local_rank >= 0
        if distributed:
            device = torch.device('cuda:{}'.format(args.local_rank))
            torch.cuda.set_device(device)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://",
            )

        model = OHybridCR(args, self.nclass, backbone=args.backbone).cuda()
        # optimizer
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': args.lr}]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9,
                                    weight_decay=0.0001, nesterov=False)

        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.pt')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

        self.criterion = SegmentationLosses(n_classes=self.nclass, weight=weight, cuda=args.cuda).build_loss(
            mode=args.loss_type)
        # if args.backbone == "hrnet":
        model = FullModel_nostream(model, self.criterion).cuda()
        from torchinfo import summary
        summary(model)
        self.model, self.optimizer = model, optimizer

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        self.evaluator.reset()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.long().cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            loss, output = self.model(image, target)

            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            # print(target.shape, output[1].shape)
            self.evaluator.add_batch(target.detach().cpu().numpy(), output[0])

            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train total_loss per epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        iou = self.evaluator.Intersection_over_Union()
        precision = self.evaluator.precision_macro_average()
        recall = self.evaluator.recall_macro_average()
        f1score = self.evaluator.getF1Score()

        train_result = {"acc": Acc, "acc_clas": Acc_class, "iou": iou, "miou": mIoU, "fwiou": FWIoU,
                       "prec": precision, "rec": recall, "f1sco": f1score}

        return train_loss, train_result

    def log_experimental_Data(self, filename, data):
        with open(filename, "w") as metric:
            result = []
            for i in range(0, len(data)):
                result.append(str(data[i]["acc"]) + str(",") + str(data[i]["acc_clas"]) + str(",") + str(data[i]["iou"]) + str(",") +
                              str(data[i]["miou"]) + str(",") + str(data[i]["fwiou"]) +str(",") + str(data[i]["prec"]) +
                              str(",") + str(data[i]["rec"]) + str(",") + str(data[i]["f1sco"]))

            metric.writelines("% s\n" % d for d in result)
            metric.close()

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.long().cuda()
            with torch.no_grad():
                loss, output = self.model(image, target)

            loss = loss.mean()
            test_loss += loss.item()

            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            target = target.cpu().numpy()
            self.evaluator.add_batch(target, output[0])

        # Fast test during the training
        # print(self.evaluator.view_confusion_matrics())
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        iou = self.evaluator.Intersection_over_Union()
        precision = self.evaluator.precision_macro_average()
        recall = self.evaluator.recall_macro_average()
        f1score = self.evaluator.getF1Score()

        val_result = {"acc": Acc, "acc_clas": Acc_class, "iou": iou, "miou": mIoU, "fwiou": FWIoU,
                       "prec": precision, "rec": recall, "f1sco": f1score}

        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Mean Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        return test_loss, val_result


# python train.py --cfg=test.yaml --backbone='regnet' --epoch=1
def main(lr):
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--backbone', type=str, default='ours_l34rw_fully', choices=['baseline','ours_l34rw_partial_weight', 'ours_l34rw_fully',
                'ours_l34rw_partial_cwffd', 'ours_r34rw_fully', 'ours_r50rw_fully', 'ours_l34rw_partial_decoder'], help='Backone name (default: hrnet)')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--mstream', type=str, default='yes', choices=['yes', 'no'], help='loss func type (default: yes )'),
    parser.add_argument('--r2n', type=str, default='no', choices=['yes', 'no'], help='loss func type (default: ce)')
    parser.add_argument('opts', help='Modify config options using the command-line', default=None,
                        nargs=argparse.REMAINDER)
    # ----------------------------------Dataset and the loss function--------------------------------------------------
    parser.add_argument('--dataset', type=str, default='cweeds', choices=['cweeds', 'bweeds', 'rweeds'],
                        help='dataset name (default: cweeds)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=64,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=64,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=True,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='log',
                        choices=['dice', 'ce', 'focal', 'log'],
                        help='loss func type (default: ce)')
    # --------------------------------model training parameter----------------------------------------------------------
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=5,
                        metavar='N', help='input batch size for \
                                    training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                    testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # ---------------------------------------- optimizer params --------------------------------------------------------
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    # -------------------------------------- checking point ------------------------------------------------------------
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # ------------------------------------- fine-tuning pre-trained models ---------------------------------------------
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # ------------------------------------- evaluation option ----------------------------------------------------------
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # ----------------------------------------------------------------------------------
    args = parser.parse_args()
    # update_config(config, args)

    # training process
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 10,
            'cityscapes': 200,
            'pascal': 50,
            'weeds': 2,
            'bonirob': 1,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'weeds': lr,
        }

        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    train = []
    val = []
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        _, t_metric = trainer.training(epoch)
        train.append(t_metric)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            v_loss, v_metric = trainer.validation(epoch)
            val.append(v_metric)
    # train and val metrics
    trainer.log_experimental_Data(str(args.dataset)+str(datetime.now().time())+"_train.txt", train)
    trainer.log_experimental_Data(str(args.dataset)+str(datetime.now().time())+"_val.txt", val)

    trainer.writer.close()
    return args


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    lrs = [0.1, 0.1, 0.1]
    for i in range(len(lrs)):
        print("Starting LR .......", lrs[i])
        main(lrs[i])
        directory = ROOT_DIR + "/experiments/"
        print("End......", lrs[i])
        print("Processing the best model for lr=", i, ".............")
        os.rename(directory + "rweeds", directory + str(i) + "test_ours")

