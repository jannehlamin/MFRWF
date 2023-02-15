# import warnings
# warnings.filterwarnings('ignore')
# warnings.simplefilter('ignore')

import warnings
warnings.filterwarnings('ignore')
from fvcore.nn import FlopCountAnalysis, flop_count_table
from models.nets.hybrid_ocr_nostream import OHybridCR
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--backbone', type=str, default='ours_l34rw_fully',
                        choices=['baseline', 'ours_l34rw_partial_weight', 'ours_l34rw_fully',
                                 'ours_l34rw_partial_cwffd', 'ours_r34rw_fully', 'ours_r50rw_fully',
                                 'ours_l34rw_partial_decoder'], help='Backone name (default: hrnet)')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0', help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args



if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    # from torchinfo import summary
    # model = models.resnet50(pretrained=True)
    model = OHybridCR(args, 3, backbone=args.backbone).cuda()
    input_tensor = torch.randn(5, 3, 128, 128).cuda()
    from collections import defaultdict
    flops = FlopCountAnalysis(model, input_tensor)

    # giga_flops = defaultdict(float)
    # for op, flop in flops.by_operator().items():
    #     giga_flops[op] = flop /  1e9

    print(flop_count_table(flops))
    print("Flobs", flops.total(),"   GFlobs",flops.total() / 1e9)

