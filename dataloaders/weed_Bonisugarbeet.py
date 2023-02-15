import os
import torch.utils.data as data
from PIL import ImageFile, Image, ImageOps
from dataloaders.data_util.utils import decode_segmap, get_weeds
import shutil
import cv2 as cv
from dataloaders.data_utils import preprocess_edge
from mypath import Path
from dataloaders.data_util import custom_transforms as tr
from torchvision import transforms
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BornsugarbeetWeed(data.Dataset):
    NUM_CLASSES = 3

    def __init__(self, args, base_dir=Path.db_root_dir('bweeds'), base_to='train'):

        super().__init__()
        self.base_dir = base_dir
        self.base_to = base_to
        self.base_root = os.path.join(base_dir, self.base_to, 'image')
        self.path_to_edges = os.path.join(base_dir, self.base_to)

        self.args = args
        self.edge_type = args.edge
        self.is_edge = args.use_edge

        self.files = [os.path.join(self.base_root, x) for x in sorted(os.listdir(str(self.base_root)))]
        self.images = [t for t in self.files]
        self.masks = [m.replace("image", "mask") for m in self.images]
        if self.is_edge:
            self.edgeMasks = self.edgeMask(self.images, self.edge_type)

    def preproces(self, image):
        img = cv.imread(image)
        # Convert to graycsale
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)
        # Canny Edge Detection
        return img_blur

    def hisequ(self, result):
        result = result.astype(np.uint8)
        result = cv.equalizeHist(result)
        return result

    def feature_map_color_Index(self, lst_image_edge, path_dir, c_index='ndi'):
        result = []
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)
            for mask in lst_image_edge:
                ndi_colorIndex = preprocess_edge(mask, c_index=c_index)
                cv.imwrite(path_dir + str("/") + mask.split("/")[-1], self.hisequ(ndi_colorIndex))
                result.append(path_dir + str("/") + mask.split("/")[-1])
        else:
            for mask in lst_image_edge:
                # img_blur = self.preproces(mask)
                # edges = cv.Laplacian(img_blur, cv.CV_64F)  # Canny Edge Detection
                # cv.imwrite(path_dir + str("/") + mask.split("/")[-1], edges)
                result.append(path_dir + str("/") + mask.split("/")[-1])
        return result

    def edgeMask(self, lst_image_edge, status="gray"):
        # # =================================== edge algorithms =======================##
        path_dir = os.path.join(self.path_to_edges, "pseudo_" + str(status))
        # Canny Edge Detection
        if status == "canny":
            result = []
            if not os.path.isdir(path_dir):
                os.makedirs(path_dir)
                for mask in lst_image_edge:
                    img_blur = self.preproces(mask)
                    edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
                    cv.imwrite(path_dir + str("/") + mask.split("/")[-1], edges)
                    result.append(path_dir + str("/") + mask.split("/")[-1])
            else:
                for mask in lst_image_edge:
                    # img_blur = self.preproces(mask)
                    # edges = cv.Laplacian(img_blur, cv.CV_64F)  # Canny Edge Detection
                    # cv.imwrite(path_dir + str("/") + mask.split("/")[-1], edges)
                    result.append(path_dir + str("/") + mask.split("/")[-1])
            return result
        # laplician Edge Detection
        elif status == "laplician":
            result = []
            if not os.path.isdir(path_dir):
                os.makedirs(path_dir)
                for mask in lst_image_edge:
                    img_blur = self.preproces(mask)
                    edges = cv.Laplacian(img_blur, cv.CV_64F)  # laplacian Edge Detection
                    cv.imwrite(path_dir + str("/") + mask.split("/")[-1], edges)
                    result.append(path_dir + str("/") + mask.split("/")[-1])
            else:
                for mask in lst_image_edge:
                    # img_blur = self.preproces(mask)
                    # edges = cv.Laplacian(img_blur, cv.CV_64F)  # Canny Edge Detection
                    # cv.imwrite(path_dir + str("/") + mask.split("/")[-1], edges)
                    result.append(path_dir + str("/") + mask.split("/")[-1])
            return result

        # # =================================== this is the color index transformation =======================##
        # elif status == "ndi":
        #     result = self.feature_map_color_Index(lst_image_edge, path_dir, c_index=status)
        if status == "exg":
            result = self.feature_map_color_Index(lst_image_edge, path_dir, c_index=status)
        elif status == "exgr":
            result = self.feature_map_color_Index(lst_image_edge, path_dir, c_index=status)
        elif status == "gray":
            result = self.feature_map_color_Index(lst_image_edge, path_dir, c_index=status)
        # # =================================== end ======================================##
        else:
            # sobel Edge Detection
            result = []
            if not os.path.isdir(path_dir):
                os.makedirs(path_dir)
                for mask in lst_image_edge:
                    img_blur = self.preproces(mask)
                    edges = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)  # laplacian Edge Detection
                    cv.imwrite(path_dir + str("/") + mask.split("/")[-1], edges)
                    result.append(path_dir + str("/") + mask.split("/")[-1])
            else:
                for mask in lst_image_edge:
                    # img_blur = self.preproces(mask)
                    # edges = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)  # Canny Edge Detection
                    # cv.imwrite(path_dir + str("/") + mask.split("/")[-1], edges)
                    result.append(path_dir + str("/") + mask.split("/")[-1])

        return result

    def mask_to_class(self, img, color_codes=get_weeds(), one_hot_encode=False):

        if color_codes is None:
            color_codes = {val: i for i, val in enumerate(set(tuple(v) for m2d in img for v in m2d))}

        n_labels = len(color_codes)
        result = np.ndarray(shape=img.shape[:2], dtype=int)
        result[:, :] = 255
        for rgb, idx in color_codes.items():
            result[(img == rgb).all(2)] = idx

        if one_hot_encode:
            one_hot_labels = np.zeros((img.shape[0], img.shape[1], n_labels))
            # one-hot encoding
            for c in range(n_labels):
                one_hot_labels[:, :, c] = (result == c).astype(int)
            result = one_hot_labels

        return result, color_codes

    def __getitem__(self, index):

        _img, _target, edge = self._make_img_gt_point_pair(index)

        sample = {'image': _img, 'label': _target, 'edge': edge}
        if self.base_to == 'train':
            sample = self.transform_tr(sample)

        elif self.base_to == 'val':
            sample = self.transform_val(sample)
        else:
            sample = self.transform_ts(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _make_img_gt_point_pair(self, index):

        _img = Image.open(self.images[index])
        _target = Image.open(self.masks[index])
        _tmp = np.array(_target, dtype=np.uint8)
        # enhance the target to pascal voc mask (1-channel color image)
        _target, _ = self.mask_to_class(_tmp)
        _target = Image.fromarray(np.uint8(_target))

        edge_img = Image.open(self.edgeMasks[index])

        # print("The shape of the inputs: ", _img, _target, edge_img)
        return _img, _target, edge_img

    def trans(self, mask):
        # use the same transformations for train/val in this example
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
        ])

        return trans(mask)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    # def __str__(self):
    #     return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = BornsugarbeetWeed(args, base_to='train')
    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):

        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            edge = sample['edge'].numpy()

            # tmp = np.uint8(gt[jj])
            # segmap = decode_segmap(tmp, dataset='bweeds')  # dataset='weeds'
            # img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            # img_tmp *= (0.229, 0.224, 0.225)
            # img_tmp += (0.485, 0.456, 0.406)
            # img_tmp *= 255.0
            # img_tmp = img_tmp.astype(np.uint8)
            # plt.figure()
            # plt.title('display')
            # plt.subplot(211)
            # plt.imshow(img_tmp)
            # plt.subplot(212)
            # plt.imshow(segmap)

        if ii == 1:
            break
        plt.show()