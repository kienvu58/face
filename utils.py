import sys
import os

if (sys.version_info > (3, 0)):
    import pickle
else:
    import cPickle as pickle


def dump(filename, data, protocol=2):
    dir = os.path.split(filename)[0]
    if not os.path.isdir(dir):
        os.makedirs(dir)

    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=protocol)


def load(filename):
    with open(filename, "rb") as f:
        if (sys.version_info > (3, 0)):
            data = pickle.load(f, encoding="latin1")
        else:
            data = pickle.load(f)
    return data
