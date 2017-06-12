import numpy as np


class DotAlgo():
    def __init__(self, params=None):
        print("Init DotAlgo")

    def calc_reps(self, images):
        reps = []
        for img in images:
            rep = img.reshape(-1)
            reps.append(reps)

        return reps

    def calc_sim(self, rep1, rep2):
        return np.dot(rep1, rep2)


def create(params=None):
    algo = DotAlgo(params)
    return algo
