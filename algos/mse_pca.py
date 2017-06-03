import cv2
import numpy as np
from utils import *
from skimage.measure import compare_mse
from sklearn.decomposition import PCA


class MSEPCA():
    def __init__(self, params=None):
        print("Init PCA")
        images = load("dat/aligned_target_images.dat")
        n_images, h, w = images.shape
        x_train = images.reshape((n_images, -1))
        n_components = 150
        self.pca = PCA(n_components=n_components, svd_solver='randomized',
                       whiten=True).fit(x_train)

    def calc_sim(self, img1, img2):
        img1_pca = self.pca.transform(img1.reshape(1, -1))
        img2_pca = self.pca.transform(img2.reshape(1, -1))
        ret = compare_mse(img1_pca, img2_pca)
        return ret


def create(params=None):
    algo = MSEPCA(params)
    return algo
