import argparse
import os.path as osp
import numpy as np
from tqdm import tqdm
import warnings
# Architecture of different dataset
import torch
from PIL import Image
from models.nets.Unet import UNET
#from models.nets.ResUnet import ResUnet
#from models.nets.ULeafNet import ULeafNet
from models.backbone.extensions.sync_batchnorm import patch_replication_callback
#from models.nets.FCN import FCN8s
#from models.nets.SegNet import SegNet
from models.nets.Unet import UNET
#from models.nets.deeplab_resnet import DeepLabv3_plus
from dataloaders.data_util import make_data_loader_nostream
from dataloaders.data_util.utils import decode_segmap
from loss_functions.loss_enc_dec import SegmentationLosses
from loss_functions.metrics_test import Evaluator
from torchvision import transforms
import cv2
torch.cuda.set_device(0)
warnings.filterwarnings('ignore')

ORIGINAL_HEIGHT = 300 #966
ORIGINAL_WIDTH = 300 #1296
MODEL_HEIGHT = 256
MODEL_WIDTH = 256

class ModelWrapper:

    def __init__(self, args, num_class=3):

        # Define Saver
        self.args = args

        self.composed_transform = transforms.Compose([
            transforms.Resize((MODEL_HEIGHT, MODEL_WIDTH), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.nclass = num_class
        self.model = self.load_model(self.args, self.nclass)

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader_nostream(args, **kwargs)

        self.criterion = SegmentationLosses(n_classes=self.nclass, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.evaluator = Evaluator(self.nclass)

    @staticmethod
    def load_model(args, nclass):

        # Define network
        if args.model == "fcn":
            model = FCN8s(nInputChannels=3, n_class=nclass).cuda()
        elif args.model == "segnet":
            model = SegNet(num_channel=3, n_class=nclass).cuda()
        elif args.model == "unet":
            model = UNET(num_channel=3, num_class=nclass).cuda()
        elif args.model == "munet":
            model = MUNET(num_channel=3, num_class=nclass).cuda()
        elif args.model == "uleafnet":
            model = ULeafNet(num_channel=3, num_classes=nclass).cuda()
        elif args.model == "runet":
            model = ResUnet(3, nclass).cuda()
        elif args.model == "deeplab":
            model = DeepLabv3_plus(nInputChannels=3, n_classes=nclass)

        if args.cuda:
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
            patch_replication_callback(model)
            model = model.cuda()

        if not osp.isfile(args.checkname):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.checkname))
        checkpoint = torch.load(args.checkname, map_location='cuda:1')

        if args.cuda:
            try:
                model.module.load_state_dict(checkpoint['state_dict'])
            except RuntimeError as e:
                print("View error: ", str(e))
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
        repetitions = 300
        timings = np.zeros((len(tbar), 1))
        # GPU-WARM-UP
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                starter.record()  # start record
                output = self.model(image)
                ender.record()  # ending the record

                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[i] = curr_time

            loss = self.criterion(output, target)
            test_loss += loss.item()

            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.detach().cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pred_1 = pred.squeeze(0)

            # Save the prediction
            segmap = decode_segmap(pred_1, dataset=self.args.dataset, plot=False)
            segmap = np.array(segmap * 255).astype(np.uint8)

            rgb_img = cv2.resize(segmap, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT),
                                 interpolation=cv2.INTER_NEAREST)
            bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("Datasets/Carrotweed/results/" + str(i) + "_result.png", bgr)

            self.evaluator.add_batch_enc(target, pred)

        iou = self.evaluator.Intersection_over_Union()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FwIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print("*" * 20, self.args.model, " on ", ""
        if self.args.dataset == "cweeds" " Carrot weeds " else " sugar beets ", "datsets", " *" * 15)

        print('crops IOU: ', iou[1])
        print('weeds IOU: ', iou[2])
        print('MIOU: ', mIoU)
        print('FwIoU: ', FwIoU)
        
        # print('mAP@0.25: ', self.evaluator.mAP(0.25))
        # print('mAP@0.50: ', self.evaluator.mAP(0.50))
        # print('mAP@0.75: ', self.evaluator.mAP(0.75))
        
        print('Precison : ', self.evaluator.precision_macro_average())
        print('Recall : ', self.evaluator.recall_macro_average())
        print('F1-score: ', self.evaluator.getF1Score())

        mean_syn = np.sum(timings) / len(tbar)
        print('Time --> Mean : ', mean_syn, " milliseconds, seconds : ", (mean_syn / 1000))
        print("*" * 100)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

    parser.add_argument('--dataset', type=str, default='cweeds',
                        choices=['cweeds', 'bweeds', 'rweeds', 'sweeds', 'svweeds'],
                        help='dataset name (default: cweeds)')
    parser.add_argument('--model', type=str, default='fcn',
                        choices=['fcn', 'segnet', 'unet', 'deeplab','munet','uleafnet', 'runet'],
                        help='selected models ')
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
            if args.model == 'fcn':
                checkname = "experiments_enc_dec/fcn_bweeds.pth.tar"
            elif args.model == 'segnet':
                checkname = "experiments_enc_2/1_bsgnweeds_ori_segnetsugarbeet/original_unet-resnet/model_best.pth.tar"
            elif args.model == 'unet':
                checkname = "experiment/unet/model_best.pth.tar" #"experiments_enc_2/0_ubweeds_ori_unetsugarbeet/original_unet-resnet/model_best.pth.tar"
            elif args.model == 'uleafnet':
                checkname = "experiments_enc_2/0_bweeds_uleaf_selected/original_unet-resnet/model_best.pth.tar"
            elif args.model == 'munet':
                checkname = "experiments/1_bmuweeds/munet-resnet/model_best.pth.tar"
            elif args.model == 'runet':
                checkname = "experiments/2_uresidualnetbweed_selected/original_unet-resnet/model_best.pth.tar"
            elif args.model == 'deeplab':
                checkname = "experiments_12/1_bweed_deeplab_resnet_selected/deeplab-resnet/model_best.pth.tar"
        else:
            if args.model == 'fcn':
                checkname = "experiments_enc_dec/fcn_cweeds.pth.tar"
            elif args.model == 'segnet':
                checkname = "experiments_12/0_sgn_carrotweeds_ori_selected/original_unet-resnet/model_best.pth.tar"
            elif args.model == 'unet':
                checkname = "experiments/1_unet_carrotweeds_ori/original_unet-resnet/model_best.pth.tar"
            elif args.model == 'uleafnet':
                checkname = "experiments_12/2_carrotweeds_uleaf_selected/original_unet-resnet/model_best.pth.tar"
            elif args.model == 'runet':
                checkname = "experiments_12/1_uresidualnetcarrotweed_selected/original_unet-resnet/model_best.pth.tar"
            elif args.model == 'munet':
                checkname = "experiments/1_cmuweeds/munet-resnet/model_best.pth.tar"
            elif args.model == 'deeplab':
                checkname = "experiments_enc_2/1_carrot_weed_deeplab_carrot_selected/deeplab-carrot/model_best.pth.tar"

        args.checkname = checkname

    test_eval = ModelWrapper(args)
    test_eval.test_evaluation(1)


if __name__ == "__main__":
    main()


