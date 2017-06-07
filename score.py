from __future__ import print_function
import argparse
import os
import numpy as np
from config import *
from utils import *
import cv2
import xml.etree.ElementTree as ET
import sys


if (sys.version_info > (3, 0)):
    import importlib.util
else:
    import importlib


def get_affine_transform(scale_factor, left_eye, right_eye, mouth):
    original = np.float32([left_eye, right_eye, mouth])
    landmarks = LANDMARKS
    original = np.multiply(original, scale_factor)
    landmarks = np.multiply(landmarks, scale_factor)
    M = cv2.getAffineTransform(original, landmarks)
    return M


def get_coordinates(attrib):
    x = int(attrib["x"])
    y = int(attrib["y"])
    return [x, y]


def calc_scale_factor(height, width):
    scale_h = height / HEIGHT
    scale_w = width / WIDTH
    return np.float32([scale_h, scale_w])


def get_face_info(subject, img_file_name):
    name, ext = os.path.splitext(img_file_name)
    ground_truth_fn = name + ".xml"
    ground_truth_path = os.path.join(GROUND_TRUTH, subject, ground_truth_fn)
    tree = ET.parse(ground_truth_path)
    root = tree.getroot()
    try:
        face_info = root[0][4][0][0]
        left_eye = face_info[4]
        right_eye = face_info[5]
        # nose = face_info[6]
        mouth = face_info[7]
    except Exception:
        print(subject, img_file_name)
        return None

    left_eye_coords = get_coordinates(left_eye.attrib)
    right_eye_coords = get_coordinates(right_eye.attrib)
    mouth_coords = get_coordinates(mouth.attrib)
    return left_eye_coords, right_eye_coords, mouth_coords


def align_feret(img, subject, img_file_name):
    h, w = img.shape[:2]
    face_info = get_face_info(subject, img_file_name)
    scale_factor = calc_scale_factor(h, w)
    if face_info is not None:
        affine_transform = get_affine_transform(scale_factor, *face_info)
        aligned_img = cv2.warpAffine(img, affine_transform, (w, h))
    else:
        aligned_img = img
    # cv2.imshow("original", img)
    # cv2.imshow("affine", aligned_img)
    # cv2.waitKey(0)
    return aligned_img


def calc_scores(algo, target, query, align=False):
    scores = np.empty((len(query), len(target)))
    # count_down = scores.size
    for q_ids, q in enumerate(query):
        q_img_path = os.path.join(FERET_DIR, *q)
        for t_ids, t in enumerate(target):
            # print(count_down)
            t_img_path = os.path.join(FERET_DIR, *t)
            target_img = cv2.imread(t_img_path, cv2.IMREAD_GRAYSCALE)
            query_img = cv2.imread(q_img_path, cv2.IMREAD_GRAYSCALE)

            if align:
                target_img = align_feret(target_img, *t)
                query_img = align_feret(query_img, *q)

            scores[q_ids][t_ids] = algo.calc_sim(target_img, query_img)
            # count_down -= 1
    return scores


def main(args):
    module_name = args["module"]
    module = importlib.import_module(module_name)

    align = args["align"]

    if module is None:
        print("Cannot load module: {}".format(module_name))
    algo = module.create(args["params"])

    if args["output"] is not None:
        output = args["output"]
    else:
        output_file_name = "aligned" + module_name.replace(".", "_") + ".dat"
        output = os.path.join("scores", output_file_name)

    target = load(TARGET_SET)
    query = load(QUERY_SET)

    scores = calc_scores(algo, target, query, align=align)
    dump(output, scores)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score similarity measure algorithm.")
    parser.add_argument("module", type=str,
                        help="Algorithm module e.g. algos.example")
    parser.add_argument("-o", "--output", type=str,
                        help="Output file")
    parser.add_argument("-p", "--params", type=str,
                        help="Optional params string to run with algorithm")
    parser.add_argument("-al", "--align", dest="align", action="store_true",
                        help="Score aligned images")
    main(vars(parser.parse_args()))
