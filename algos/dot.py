import numpy as np


class DotAlgo():
    def __init__(self, params=None):
        print("Init DotAlgo")

    def calc_sim(self, img1, img2):
        img1 = img1.reshape(-1)
        img2 = img2.reshape(-1)
        return np.dot(img1, img2)


def create(params=None):
    algo = DotAlgo(params)
    return algo
