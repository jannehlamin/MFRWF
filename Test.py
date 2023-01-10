import os
import shutil
import numpy as np
from PIL import ImageFile, Image, ImageOps
import matplotlib.pyplot as plt
from dataloaders.data_util.utils import get_rice_encode, decode_segmap

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# A dataset
masks = str(ROOT_DIR) + '/masks/'
m_img_files = sorted(os.listdir(masks))


def mask_to_class(img, color_codes=get_rice_encode(), one_hot_encode=False):
    if color_codes is None:
        color_codes = {val: i for i, val in enumerate(set(tuple(v) for m2d in img for v in m2d))}

    n_labels = len(color_codes)
    result = np.ndarray(shape=img.shape[:2], dtype=int)
    result[:, :] = -1
    for rgb, idx in color_codes.items():
        # print(rgb, idx)  # (img == rgb).all(2)
        result[np.where(img == rgb)] = idx

    if one_hot_encode:
        one_hot_labels = np.zeros((img.shape[0], img.shape[1], n_labels))
        # one-hot encoding
        for c in range(n_labels):
            one_hot_labels[:, :, c] = (result == c).astype(int)
        result = one_hot_labels

    return result, color_codes

#
# count = 0
# import cv2
# file = m_img_files
# for file in m_img_files:
#     _target = Image.open(masks + str(file))
#     _tmp = np.array(_target, dtype=np.uint8)
#     print(_tmp.shape)
#
#     img, _ = mask_to_class(_tmp)
#     img = decode_segmap(img, dataset='rweeds', plot=False)
#
#     segmap = np.array(img * 255).astype(np.uint8)
#
#     rgb_img = cv2.resize(segmap, (_tmp.shape[1], _tmp.shape[0]),
#                          interpolation=cv2.INTER_NEAREST)
#     bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite("masks/results/" + str(count) + "_result.png", bgr)
#     count = count + 1
#


