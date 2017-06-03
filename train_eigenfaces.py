import numpy as np
from sklearn.decomposition import PCA
from utils import *
from config import *
import cv2


def main():
    images = load("dat/target_images.dat")
    n_images, h, w = images.shape
    x_train = images.reshape((n_images, -1))
    n_components = 150
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(x_train)
    eigenfaces = pca.components_.reshape((n_components, h, w))
    # for i in range(n_components):
    #     print(eigenfaces[i])
    #     cv2.imshow(str(i), eigenfaces[i])
    #     cv2.waitKey(0)

    images_transform = pca.transform(x_train)
    images_transform = pca.inverse_transform(images_transform)
    images = images_transform.reshape((n_images, h, w))
    for i in range(10):
        print(eigenfaces[i])
        cv2.imshow(str(i), images[i])
        cv2.waitKey(0)
    

if __name__ == "__main__":
    main()
