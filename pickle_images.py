import cv2
import numpy as np
import pickle
import os
from utils import *
from config import *
from score import *


def pickle_images(output_fn, align=False):
    target = load(TARGET_SET)
    images = []
    for t in target:
        t_img_path = os.path.join(FERET_DIR, *t)
        img = cv2.imread(t_img_path, cv2.IMREAD_GRAYSCALE)

        if align:
            img = align_feret(img, *t)

        images.append(img)
        dump(output_fn, np.array(images), pickle.HIGHEST_PROTOCOL)


def main():
    pickle_images("dat/aligned_target_images.dat", True)


if __name__ == "__main__":
    main()
