import pickle
import os


def dump(filename, data, protocol=3):
    dir = os.path.split(filename)[0]
    if not os.path.isdir(dir):
        os.makedirs(dir)

    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=protocol)


def load(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data
