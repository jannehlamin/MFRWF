import argparse
import os.path as osp
import numpy as np
from tqdm import tqdm
import warnings
# Architecture of different dataset
from models.backbone.extensions.sync_batchnorm import patch_replication_callback
from dataloaders.data_util import make_data_loader_nostream
from dataloaders.data_util.utils import decode_segmap
import torch
from PIL import Image
from models.backbone.utils_1.utils import FullModel_nostream
import torch.autograd.profiler as profiler
from models.nets.hybrid_ocr_nostream import OHybridCR
from loss_functions.loss import SegmentationLosses
from loss_functions.metrics_test import Evaluator
from torchvision import transforms
import cv2
import torch.backends.cudnn as cudnn

torch.cuda.set_device(0)

warnings.filterwarnings('ignore')
EXPECTED_CHANNEL = 256

ORIGINAL_HEIGHT = 966  # 966
ORIGINAL_WIDTH = 1296  # 1296
MODEL_HEIGHT = 512
MODEL_WIDTH = 512


class ModelWrapper:

    def __init__(self, args, num_class=3):

        # Define Saver
        self.args = args

        self.composed_transform = transforms.Compose([
            transforms.Resize((MODEL_HEIGHT, MODEL_WIDTH), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.nclass = num_class
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader_nostream(args, **kwargs)

        self.criterion = SegmentationLosses(n_classes=self.nclass, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model = self.load_model(self.args, self.criterion, self.nclass)
        self.evaluator = Evaluator(self.nclass)

    @staticmethod
    def load_model(args, criterion, nclass):

        # Define network
        distributed = args.local_rank >= 0
        if distributed:
            device = torch.device('cuda:0'.format(args.local_rank))
            torch.cuda.set_device(device)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://",
            )

        model = OHybridCR(args, nclass, backbone=args.backbone).cuda()  # model = HighResolutionNet(config)
        # model = get_seg_model(config)
        model = FullModel_nostream(model, criterion)

        if args.cuda:
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
            patch_replication_callback(model)
            model = model.cuda()
        if not osp.isfile(args.checkname):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.checkname))
        checkpoint = torch.load(args.checkname, map_location='cuda:0')

        if args.cuda:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch: {}, best_pred: {})"
              .format(args.checkname, checkpoint['epoch'], checkpoint['best_pred']))
        model.eval()
        return model

    def test_evaluation(self, epoch=''):
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        # ================================= Efficient inference time evaluation =================================
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = np.zeros((len(tbar), 1))
        # GPU-WARM-UP
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.long().cuda()
            # _loss, _output = self.model(image, target) # warm up strategy in pytorch profiler https://pytorch.org/docs/stable/autograd.html?highlight=autograd%20profiler#torch.autograd.profiler.profile
            with torch.no_grad():
                torch.cuda.synchronize()
                starter.record()  # start record
                #with profiler.profile(with_stack=False, use_cpu=True, use_cuda=True, profile_memory=True) as prof:
                loss, output = self.model(image, target)
                #print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
                ender.record()  # ending the record
                torch.cuda.synchronize() # WAIT FOR GPU SYNC


                curr_time = starter.elapsed_time(ender)
                timings[i] = curr_time
                test_loss += loss.item()
            # print(torch.cuda.get_device_properties(0).total_memory)
            # print(torch.cuda.get_device_properties('cuda:0'))

            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            # print(output[1].shape)
            pred = output[1].cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred_1 = pred.squeeze(0)

            # Save the prediction
            segmap = decode_segmap(pred_1, dataset=self.args.dataset, plot=False)
            segmap = np.array(segmap * 255).astype(np.uint8)

            # pred = Image.fromarray(segmap.astype(np.uint8))
            # pred = pred.resize((1024, 1024))
            # pred.save('%s/%s' % ("Datasets/Carrotweed/results/", str(i) + "_result.png"))

            rgb_img = cv2.resize(segmap, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT),
                                 interpolation=cv2.INTER_NEAREST)
            bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("Datasets/Carrotweed/results/" + str(i) + "_result.png", bgr)

            # Add batch sample into evaluator
            self.evaluator.add_batch(target, output)
        print("GPU Memory Usage: ",torch.cuda.get_device_properties('cuda:0')) # https://developer.download.nvidia.cn/compute/DevZone/docs/html/C/doc/html/group__CUDART__DEVICE_g5aa4f47938af8276f08074d09b7d520c.html
        iou = self.evaluator.Intersection_over_Union()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FwIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print("*" * 15, " Our method on ", self.args.backbone, " backbone using ", ""
        if self.args.dataset == "cweeds" " Carrot weeds " else " sugar beets ", "datsets", " *" * 15)

        print('crops IOU: ', iou[1])
        print('weeds IOU: ', iou[2])
        print('MIOU: ', mIoU)

        print('FwIoU: ', FwIoU)
        print('Precison : ', self.evaluator.precision_macro_average())
        print('Recall : ', self.evaluator.recall_macro_average())
        print('F1-score: ', self.evaluator.getF1Score())
        mean_syn = np.sum(timings) / len(tbar)
        print('Time --> Mean : ', mean_syn, " milliseconds, seconds : ", (mean_syn / 1000))
        print("*" * 100)


def main():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--backbone', type=str, default='baseline',
                        choices=['ours_l34rw_partial_weight', 'baseline',
                                 'ours_l34rw_partial_decoder', 'ours_l34rw_fully',
                                 ], help='Backone name (default: hrnet)')
    parser.add_argument("--local_rank", type=int, default=-1),
    parser.add_argument('opts', help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    # ----------------------------------Dataset and the loss function--------------------------------------------------
    parser.add_argument('--dataset', type=str, default='cweeds',
                        choices=['cweeds', 'rweeds', 'bweeds'],
                        help='dataset name (default: cweeds)')

    parser.add_argument('--workers', type=int, default=1,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--loss-type', type=str, default='log',
                        choices=['dice', 'ce', 'focal', 'log'],
                        help='loss func type (default: ce)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                    training (default: auto)')
    # cuda, seed and logging
    parser.add_argument('--sync-bn', type=bool, default=True, help='whether to use sync bn (default: auto)')
    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')

    # For testing purpose
    args = parser.parse_args()
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

    args.base_size = 512
    args.crop_size = 512

    if args.checkname is None:
        checkname = ""
        if args.dataset == "bweeds":
            if args.backbone == "ours_l34rw_partial_weight":
                checkname = "experiments/results/bweeds/pweight/1model_best.pth.tar"
            elif args.backbone == "ours_l34rw_partial_decoder":
                checkname = "experiments/results/bweeds/pdecoder/model_best.pth.tar"
            elif args.backbone == "ours_l34rw_fully":
                checkname = "experiments/results/bweeds/fully/model_best.pth.tar"

        elif args.dataset == "cweeds":
            if args.backbone == "ours_l34rw_partial_weight":
                checkname = "experiments/results/cwd/pweight/model_best.pth.tar"
            elif args.backbone == "ours_l34rw_partial_cwffd":
                checkname = "experiments/results/cwd/deeplab-ours_l34rw_partial_cwffd/model_best.pth.tar"
            elif args.backbone == "ours_l34rw_partial_decoder":
                checkname = "experiments/results/cwd/pdecoder/model_best.pth.tar"
            elif args.backbone == "ours_l34rw_fully":
                checkname = "experiments/results/cwd/fully/model_best.pth.tar"
        else:
            if args.backbone == "ours_l34rw_partial_weight":
                checkname = "experiments/results/rice/pweight/model_best.pth.tar"
            elif args.backbone == "ours_l34rw_partial_cwffd":
                checkname = "experiments/result/rice/deeplab-ours_l34rw_partial_cwffd/model_best.pth.tar"
            elif args.backbone == "ours_l34rw_partial_decoder":
                checkname = "experiments/results/rice/pdecoder/model_best.pth.tar"
            elif args.backbone == "ours_l34rw_fully":
                checkname = "experiments/results/rice/fully/model_best.pth.tar"

        args.checkname = checkname

    test_eval = ModelWrapper(args)
    test_eval.test_evaluation(1)


if __name__ == "__main__":
    main()
