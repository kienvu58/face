import numpy as np
from skimage.measure import compare_mse


class MSE():
    def __init__(self, params=None):
        print("Init MSE")

    def calc_reps(self, images):
        return images

    def calc_sim(self, rep1, rep2):
        ret = compare_mse(rep1, rep2)
        return ret


def create(self, params=None):
    algo = MSE(params)
    return algo
