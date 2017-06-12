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

    def calc_reps(self, images):
        reps = []
        for img in images:
            rep = self.pca.transform(img.reshape(1, -1))
            reps.append(rep)

        return reps

    def calc_sim(self, rep1, rep2):
        ret = compare_mse(rep1, rep2)
        return ret


def create(params=None):
    algo = MSEPCA(params)
    return algo
