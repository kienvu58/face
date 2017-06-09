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

        dlib_face_predictor = os.path.join(dlib_model_dir, "shape_predictor_68_face_landmarks.dat")
        network_model = os.path.join(openface_model_dir, "nn4.small2.v1.t7")
        self.img_dim = 96

        self.align = openface.AlignDlib(dlib_face_predictor)
        self.net = openface.TorchNeuralNet(network_model, self.img_dim)

    def get_rep(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        bb = self.align.getLargestFaceBoundingBox(img)
        if bb is None:
            raise Exception("Unable to find a face!")

        aligned_face = self.align.align(self.img_dim, img, bb,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if aligned_face is None:
            raise Exception("Unable to align image!")

        rep = self.net.forward(aligned_face)
        return rep
    
    def calc_sim(self, img1, img2):
        try:
            d = self.get_rep(img1) - self.get_rep(img2)
            ret = np.dot(d, d)
        except Exception as e:
            print("calc_sim", e)
            return 4.0
        return ret


def create(params):
    algo = OpenFace(params)
    return algo
