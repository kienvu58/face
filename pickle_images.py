import cv2
import numpy as np
import pickle
import os
from utils import *
from config import *


def main():
    target = load(TARGET_SET)
    images = []
    # for t in target:
    #     t_img_path = os.path.join(FERET_DIR, *t)
    #     img = cv2.imread(t_img_path, cv2.IMREAD_GRAYSCALE)
    #     images.append(img)

    dump("dat/target_images_1.dat", np.array(images), pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
