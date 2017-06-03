import numpy as np
from random import randint
import os
import cv2
from utils import *
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse

FERET_DIR = "D:/colorferet/data/thumbnails/"


def calculate_similarity_mse(target_path, query_path):
    target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    ret = mse(target_img, query_img)
    return ret


def calculate_similarity_ssim(target_path, query_path):
    target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    ret = 1 - ssim(target_img, query_img)
    return ret


def main():
    target = load("target.dat")
    query = load("query.dat")
    scores = np.empty((len(query), len(target)))
    count_down = scores.size
    for q_ids, q in enumerate(query):
        q_img_path = os.path.join(FERET_DIR, *q)
        for t_ids, t in enumerate(target):
            t_img_path = os.path.join(FERET_DIR, *t)
            scores[q_ids][t_ids] = calculate_similarity_mse(
                t_img_path, q_img_path)
            count_down -= 1
            print(count_down)

    dump("scores.dat", scores)


if __name__ == "__main__":
    main()
