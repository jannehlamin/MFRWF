import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import argparse
import os.path as osp
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models
from loss_functions.loss import SegmentationLosses
from models.backbone.utils_1.utils import FullModel_nostream
from models.nets.hybrid_ocr_nostream import OHybridCR
from dataloaders.data_util.utils import decode_segmap, get_weeds
from models.backbone.extensions.sync_batchnorm import patch_replication_callback
from grad_cam.pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise

from grad_cam.pytorch_grad_cam import GuidedBackpropReLUModel
from grad_cam.pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from grad_cam.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, SemanticSegmentationTarget


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')
    parser.add_argument('--dataset', type=str, default='cweeds', choices=['cweeds', 'rweeds', 'bweeds'],
                        help='dataset name (default: cweeds)')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true', default=True,
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--loss-type', type=str, default='log',
                        choices=['dice', 'ce', 'focal', 'log'],
                        help='loss func type (default: ce)')
    parser.add_argument('--backbone', type=str, default='ours_l34rw_fully',
                        choices=['baseline', 'ours_l34rw_partial_weight', 'ours_l34rw_fully',
                                 'ours_l34rw_partial_cwffd', 'ours_r34rw_fully', 'ours_r50rw_fully',
                                 'ours_l34rw_partial_decoder'], help='Backone name (default: hrnet)')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'hirescam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad', 'gradcamelementwise'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def mask_to_class(img, color_codes=get_weeds(), one_hot_encode=False):
    if color_codes is None:
        color_codes = {val: i for i, val in enumerate(set(tuple(v) for m2d in img for v in m2d))}

    n_labels = len(color_codes)
    result = np.ndarray(shape=img.shape[:2], dtype=int)
    result[:, :] = 0
    for rgb, idx in color_codes.items():
        result[(img == rgb).all(2)] = idx

    # if one_hot_encode:
    #     one_hot_labels = np.zeros((img.shape[0], img.shape[1], n_labels))
    #     # one-hot encoding
    #     for c in range(n_labels):
    #         one_hot_labels[:, :, c] = (result == c).astype(int)
    #     result = one_hot_labels
    return result


def load_model(args, nclass=3):
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
    criterion = SegmentationLosses(n_classes=3, cuda=args.cuda).build_loss(mode=args.loss_type)
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


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "hirescam": HiResCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad,
         "gradcamelementwise": GradCAMElementWise}

    if args.checkname is None:
        checkname = ""
        if args.dataset == "bweeds":
            checkname = "experiments/results/bweeds/fully/bweeds_fully_model_best.pth.tar"
        elif args.dataset == "cweeds":
            checkname = "experiments/cwd/fully/cweeds_fully_model_best.pth.tar"
        else:
            checkname = "experiments/results/rice/fully/rweeds_fully_model_best.pth.tar"

        args.checkname = checkname

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')


    # model = models.resnet50(pretrained=True)
    model = load_model(args, nclass=3)
    from torchinfo import summary
    print(model)
    summary(model)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    # image_url = "https://farm1.staticflickr.com/6/9606553_ccc7518589_z.jpg"
    # image = np.array(Image.open(requests.get(image_url, stream=True).raw))
    t_img = "053_annotation.png"
    t_image = np.array(cv2.imread(t_img))
    t_target = mask_to_class(t_image)
    gt = decode_segmap(t_target, dataset='cweeds', plot=False)

    print("Pass--------------------------")
    image_path = "053_image.png"

    # ============
    target_layers = [model.module.model.ctrancnn.conwf4.convwf]

    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    classId = 1
    mask = Image.fromarray(np.uint8(t_target))
    mask = np.array(mask.resize((512, 512), Image.BILINEAR))
    targets = [SemanticSegmentationTarget(classId, mask)]

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 1
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]
        a = Image.fromarray(np.uint8(rgb_img))
        # b = Image.fromarray(np.uint8(image))
        a = np.array(a.resize((514, 514), Image.BILINEAR))
        # b = np.array(b.resize((320, 320), Image.BILINEAR))
        print("Janneh------------>", a.shape, grayscale_cam.shape)
        cam_image = show_cam_on_image(a, grayscale_cam, use_rgb=True)
        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # target_category = classId
    # gb = gb_model(input_tensor, target_category=target_category)
    #
    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)

    cv2.imwrite(str("results/") + f'{args.method}_cam.jpg', cam_image)
    # cv2.imwrite(str("results/")+f'{args.method}_gb.jpg', gb)
    # cv2.imwrite(str("results/")+f'{args.method}_cam_gb.jpg', cam_gb)
