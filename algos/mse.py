import numpy as np
from skimage.measure import compare_mse


class MSE():
    def __init__(self, params=None):
        print("Init MSE")

    def calc_sim(self, img1, img2):
        ret = compare_mse(img1, img2)
        return ret


def create(self, params=None):
    algo = MSE(params)
    return algo
