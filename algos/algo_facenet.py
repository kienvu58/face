from __future__ import print_function
import os
from algos import align_dlib
import cv2
import numpy as np
from algos import facenet


class FacenetAlgo():
    def __init__(self, params=None):
        print("Init FacenetAlgo")
        face_predictor = "algos/shape_predictor_68_face_landmarks.dat"
        self.align = align_dlib.AlignDlib(face_predictor)
        self.img_dim = 160

    def calc_reps(self, imgs):
        align_faces = []
        for img in imgs:
            bb = self.align.getLargestFaceBoundingBox(img)
            if bb is None:
                raise Exception("Unable to find a face!")

            aligned_face = self.align.align(self.img_dim, img, bb,
                                            landmarkIndices=align_dlib.AlignDlib.OUTER_EYES_AND_NOSE)
            if aligned_face is None:
                raise Exception("Unable to align image!")
            align_faces.append(align_face)

        model = "algos/20170512-110547"
        images = facenet.prewhiten(np.array(align_faces))
        with tf.Graph().as_default():

            with tf.Session() as sess:

                # Load the model
                facenet.load_model(model)
                print(time.time() - start)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: images,
                             phase_train_placeholder: False}
                emb = sess.run(embeddings, feed_dict=feed_dict)
        return emb

    def calc_sim(self, rep1, rep2):
        dist = np.sqrt(np.sum(np.square(np.subtract(rep1, rep2))))
        return dist


def create(params=None):
    algo = FacenetAlgo(params)
    return algo
