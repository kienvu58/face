import cv2
from utils import *
import numpy as np
# import matplotlib.pyplot as plt
from evaluate import *
# from scipy.stats import multivariate_normal
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
from config import *
import xml.etree.ElementTree as ET
import os
import cv2
from score import *
from utils import *


def modify(scores):
    scores[0][0] = 1


# def main():
#     scores = load("scores/algos_example.dat")
#     print(scores[0])
#     print(scores)
#     # scores = np.empty((3, 3))
#     # print(scores)
#     # modify(scores)
#     # print(scores)


def main():
    subject = "00001"
    img_fn = "00001_930831_fa_a.ppm"
    face_info = get_face_info(subject, img_fn)
    print(face_info)
    img_path = os.path.join(FERET_DIR, "00001/00001_930831_fa_a.ppm")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    print(img.dtype)
    # align_feret(img, subject, img_fn)

    # path = os.path.join(GROUND_TRUTH, "00001/00001_930831_fa_a.xml")
    # tree = ET.parse(path)
    # root = tree.getroot()
    # face_info = root[0][4][0][0]
    # left_eye = face_info[4]
    # right_eye = face_info[5]
    # nose = face_info[6]
    # mouth = face_info[7]
    # print(mouth.attrib)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

# def main():
#     x, y = np.mgrid[-1.0:1.0:30j, -1.0:1.0:30j]
#     # Need an (N, 2) array of (x, y) pairs.
#     xy = np.column_stack([x.flat, y.flat])
#     mu = np.array([0.0, 0.0])
#     sigma = np.array([.5, .5])
#     covariance = np.diag(sigma**2)
#     covariance = np.array([
#         [1, 0.8],
#         [-0.1, 1]
#     ])
#     z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
#     # Reshape back to a (30, 30) grid.
#     z = z.reshape(x.shape)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(x, y, z, cmap=cm.jet)
#     # ax.plot_wireframe(x,y,z)
#     plt.show()

# def main():
#     scores = load("scores/scores_ssim.dat")
#     scores = normalize_scores(scores)
#     probe = load("dat/probe_dup1.dat")
#     ids = load("dat/ids_dup1.dat")
#     D, F = partition(scores, 0, probe, ids)
#     plt.hist(scores.transpose()[0], 20)
#     plt.hist(D, 20)
#     plt.hist(F, 20)
#     plt.show()


if __name__ == "__main__":
    main()
