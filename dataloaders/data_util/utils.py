import matplotlib.pyplot as plt
import numpy as np
import torch


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'cweeds':
        n_classes = 3
        label_colours = get_weed_labels()
    elif dataset == 'bweeds':
        n_classes = 3
        label_colours = get_weed_labels()
    elif dataset == 'rweeds':
        n_classes = 4
        label_colours = get_weed_rice1()
    elif dataset == 'sweeds':
        n_classes = 3
        label_colours = get_weed_labels()
    elif dataset == 'svweeds':
        n_classes = 3
        label_colours = get_weed_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()

    count = 0
    for ll in range(0, n_classes):
        if not (n_classes == 4 and ll == 3):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
            count = count + 1

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))  # change 1, 3

    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def cdecode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'cweeds':
        n_classes = 3
        label_colours = get_weed_rice()
    elif dataset == 'bweeds':
        n_classes = 3
        label_colours = get_weed_labels()
    elif dataset == 'rweeds':
        n_classes = 4
        label_colours = get_weed_rice1()
    elif dataset == 'sweeds':
        n_classes = 3
        label_colours = get_weed_labels()
    elif dataset == 'svweeds':
        n_classes = 3
        label_colours = get_weed_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()

    count = 1
    for ll in range(0, n_classes):
        if not (n_classes == 4):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
            count = count + 1

    print("Janneh==>", count)

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))  # change 1, 3
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for ii, label in enumerate(get_weed_rice1()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


# get Onion labels
# def get_onion():
#     color = {}
#     i = 0
#     for val in get_weed_onion():
#         color[tuple(val)] = i
#         i = i + 1
#     return color

def get_rice_encode():
    return {2: 0, 1: 1, 3: 2, 0: 0} # change


# def get_rice_encode1():
#     return {2: 0, 1: 1, 0: 2, 3:3}


def get_weeds():
    color = {}
    i = 0
    for val in get_weed_labels():
        color[tuple(val)] = i
        i = i + 1
    return color


def get_weed_labels():
    return np.array([  # RGB
        [0, 0, 0],  # bg
        [0, 255, 0],  # crop
        [255, 0, 0],  # weed
    ])

def get_weed_rice():  # since crops are label as blue colors
    return np.array([  # RGB
        [0, 0, 0],  # 0 shw
        [0, 255, 0],  # 1
        [255, 0, 0],  # 3

    ])


def get_weed_rice1():  # since crops are label as blue colors
    return np.array([  # RGB
        [0, 0, 0],  # 0 shw
        [0, 255, 0],  # 1
        [255, 0, 0],  # weed
        [255, 255, 0],  # 3
    ])


# def get_weed_rice():  # since crops are label as blue colors
#     return np.array([  # RGB
#         [0, 0, 0],  # 0 shw
#         [0, 255, 0],  # 1
#         [0, 0, 0],  # 2  bg
#         [255, 0, 0],  # 3
#
#     ])

def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])

# print(get_weeds())
