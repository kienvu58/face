from __future__ import print_function
import os
import openface
import cv2
import numpy as np


class OpenFace():
    def __init__(self, params=None):
        print("Init OpenFace")
        model_dir = "/data/kienvt/face/models/"
        dlib_model_dir = os.path.join(model_dir, "dlib")
        openface_model_dir = os.path.join(model_dir, "openface")

        dlib_face_predictor = os.path.join(
            dlib_model_dir, "shape_predictor_68_face_landmarks.dat")
        network_model = os.path.join(openface_model_dir, "nn4.small2.v1.t7")
        self.img_dim = 96

        self.align = openface.AlignDlib(dlib_face_predictor)
        self.net = openface.TorchNeuralNet(network_model, self.img_dim)

    def get_rep(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bb = self.align.getLargestFaceBoundingBox(img)
        if bb is None:
            raise Exception("Unable to find a face!")

        aligned_face = self.align.align(self.img_dim, img, bb,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if aligned_face is None:
            raise Exception("Unable to align image!")

        rep = self.net.forward(aligned_face)
        return rep

    def calc_reps(self, images):
        reps = []
        for img in images:
            rep = self.get_rep(img)
            reps.append(rep)
        return reps

    def calc_sim(self, rep1, rep2):
        d = rep1 - rep2
        ret = np.dot(d, d)
        return ret


def create(params=None):
    algo = OpenFace(params)
    return algo

if __name__ == "__main__":
    algo = create()
    img1 = cv2.imread("/data/kienvt/colorferet/data/thumbnails/00001/00001_930831_fa_a.ppm", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("/data/kienvt/colorferet/data/thumbnails/00001/00001_930831_fb_a.ppm", cv2.IMREAD_GRAYSCALE)
    ret = algo.calc_sim(img1, img2)
    print(ret)

