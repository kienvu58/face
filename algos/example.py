import random


class ExampleAlgo:
    def __init__(self, params=None):
        print("Init ExampleAlgo")
        print(params)

    def calc_reps(self, images):
        return images

    def calc_sim(self, rep1, rep2):
        ret = random.random()
        return ret


def create(params=None):
    algo = ExampleAlgo(params)
    return algo


if __name__ == "__main__":
    algo = create(params=None)
    print(algo.calc_sim(None, None))
