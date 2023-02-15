import numpy as np
# ## This is the implementation of the evaluation metric of a class # ##
# ## base off the class specific and the overall metrics # ##
from torch.nn import functional as F

class Evaluator(object):
    def __init__(self, num_class, ignore_index=0):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.ignore_index = ignore_index

    # The following are the general metric for the entire class
    # overall accuracy
    def Pixel_Accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)+0.000001
        Acc = np.nanmean(Acc)
        return Acc

    # average accuracy or mean accuracy
    def Mean_Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    # average prediction or mean prediction
    def Mean_Intersection_over_Union(self):
        MIoU = np.nanmean(self.Intersection_over_Union())
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def precision_macro_average(self):
        rows, columns = self.confusion_matrix.shape
        sum_of_precisions = 0
        for label in range(rows):
            a = self.precision(label)
            # print("observe", a, label)
            sum_of_precisions += a
        # print(rows, columns, sum_of_precisions)
        return sum_of_precisions / (rows-1)

    def mAP(self, threshold):
        rows, columns = self.confusion_matrix.shape
        iou = self.Intersection_over_Union()
        sum_of_precisions = 0
        for label in range(rows):
            if iou[label] >= threshold:
                a = self.precision(label)
                sum_of_precisions += a
        return sum_of_precisions / (rows - 1)

    def recall_macro_average(self):
        rows, columns = self.confusion_matrix.shape
        sum_of_recalls = 0
        for label in range(columns):
            sum_of_recalls += self.recall(label)
        return sum_of_recalls / (columns-1)

    def getF1Score(self):
        up = 2 * (self.precision_macro_average() * self.recall_macro_average())
        down = self.precision_macro_average() + self.recall_macro_average()
        return up / down

    # Confusion metrics generation
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        if self.ignore_index is not None:
            confusion_matrix[:, self.ignore_index] = 0
            confusion_matrix[self.ignore_index, :] = 0
        return confusion_matrix

    def view_confusion_matrics(self):
        return self.confusion_matrix
    # class specific evaluation metrics

    # prediction per class
    def Intersection_over_Union(self):
        intersection = np.diag(self.confusion_matrix)
        ground_truth = np.sum(self.confusion_matrix, axis=1)
        prediction = np.sum(self.confusion_matrix, axis=0)
        union = ground_truth + prediction - intersection
        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            return intersection / union

    def precision(self, label):
        if self.ignore_index == label:
            return 0
        else:
            col = self.confusion_matrix[:, label]
            return self.confusion_matrix[label, label] / col.sum()

    def recall(self, label):
        if self.ignore_index == label:
            return 0
        else:
            row = self.confusion_matrix[label, :]
            return self.confusion_matrix[label, label] / row.sum()

    def add_batch(self, label, pred):

        if not isinstance(pred, (list, tuple)):
            pred = [pred]
        for i, x in enumerate(pred):
            x = F.interpolate(
                input=x, size=label.shape[1:],
                mode='bilinear', align_corners=True
            )

            x = x.cpu().numpy().transpose(0, 2, 3, 1)
            x = np.asarray(np.argmax(x, axis=3), dtype=np.uint8)

            self.confusion_matrix += self._generate_matrix(label, x)

    # Confusion metrics generation
    def _generate_matrix_enc(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        if self.ignore_index is not None:
            confusion_matrix[:, self.ignore_index] = 0
            confusion_matrix[self.ignore_index, :] = 0
        return confusion_matrix

    def add_batch_enc(self, gt_image, pre_image):
        # print("check->", pre_image.shape, gt_image.shape)
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix_enc(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == '__main__':
    imgPredict = np.array([0, 0, 1, 1, 2, 2])
    imgLabel = np.array([0, 0, 1, 1, 2, 2])
    metric = Evaluator(3)
    metric.add_batch_enc(imgPredict, imgLabel)
    iou = metric.Intersection_over_Union()
    mIoU = metric.Mean_Intersection_over_Union()
    acc = metric.Pixel_Accuracy()
    map1 = metric.mAP(0.25)
    map2 = metric.mAP(0.5)
    map3 = metric.mAP(0.75)
    print(map1, map2, map3)
    print(iou, mIoU, acc)
    print('Precision : ', metric.precision_macro_average())
    print('Recall : ', metric.recall_macro_average())
    print('F-score : ', metric.getF1Score())
