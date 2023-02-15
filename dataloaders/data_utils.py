import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def readImage(file):
    np.seterr(invalid='ignore', divide='ignore')
    img = cv.imread(file)
    b, g, r = cv.split(img)

    b = (b - np.min(b)) / np.ptp(b)
    g = (g - np.min(g)) / np.ptp(g)
    r = (r - np.min(r)) / np.ptp(r)
    return r, g, b


def getNDI(filename):
    r, g, b = readImage(filename)
    # ignore the invalid warning message
    result = 128 * (((g - r) / (g + r)) + 1)
    result = result.astype(np.uint8)
    result = cv.equalizeHist(result)
    return result


def getExG(filename):
    r, g, b = readImage(filename)
    denom = (r + g + b)
    r = r / denom
    g = g / denom
    b = b / denom
    result = (2 * g) - r - b
    return result


def getExR(filename):
    r, g, b = readImage(filename)
    result = (1.3 * r) - g
    result = result.astype(np.uint8)
    result = cv.equalizeHist(result)
    return result


def getExGR(filename):
    result = getExG(filename) - getExR(filename)
    return result


def getCive(filename):
    r, g, b = readImage(filename)
    result = (0.441 * r) - (0.811 * g) + (0.385 * b) + 18.78745
    result = result.astype(np.uint8)
    result = cv.equalizeHist(result)
    return result


def getVEG(filename):
    # np.seterr(invalid='ignore', divide='ignore')
    r, g, b = readImage(filename)
    a = 0.667
    result = g / (np.power(r, a) * np.power(b, (1 - a)))
    return result


def getGray(filename):
    img1 = cv.imread(filename)
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    return img1


def getCom1(filename):
    result = getExG(filename) + getCive(filename) + getExR(filename) + getVEG(filename)
    result = result.astype(np.uint8)
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    result = clahe.apply(result)
    return result


def adativehisequ(result):
    result = result.astype(np.uint8)
    clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(10, 10))
    result = clahe.apply(result)
    result = cv.Laplacian(result, cv.CV_64F)  # laplacian Edge Detection
    return result


def hisequ(result):
    result = result.astype(np.uint8)
    result = cv.equalizeHist(result)
    # result = cv.Laplacian(result, cv.CV_64F)  # laplacian Edge Detection
    return result


def preprocess_edge(filename, c_index="ndi"):
    if c_index == "ndi":
        return hisequ(getNDI(filename))
    elif c_index == "exg":
        return getExG(filename)
    elif c_index == "exr":
        return getExR(filename)
    elif c_index == "exgr":
        return hisequ(getExGR(filename))
    elif c_index == "cive":
        return getCive(filename)
    elif c_index == "veg":
        return getVEG(filename)
    elif c_index == "gray":
        return getGray(filename)
    elif c_index == "com1":
        return getCom1(filename)

# from PIL import Image
# img1 = getVEG("035_image.png")
# img = cv.imread("035_image.png")
# # Convert to graycsale
# print(img1.shape)
# norm_img = np.zeros(img1.shape)
# img_ob = cv.normalize(img1,  norm_img, 0, 255, cv.NORM_MINMAX)
# img_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# print(img.shape)
# # Blur the image for better edge detection
# img_blur = cv.GaussianBlur(img1, (3, 3), 0)
# edges = cv.Canny(image=img_ob, threshold1=100, threshold2=200)  # Canny Edge Detection
# plt.subplot(221), plt.imshow(img1)
# plt.show()
