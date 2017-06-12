from __future__ import print_function
import argparse
import os
import numpy as np
from config import *
from utils import *
import cv2
import xml.etree.ElementTree as ET
import sys
from multiprocessing import Pool, Array, Process


if (sys.version_info > (3, 0)):
    import importlib.util
else:
    import importlib

g_target_reps = None
g_query_reps = None
g_scores = None
g_algo = None


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
        # print(subject, img_file_name)
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


def calc_scores_worker(q_id):
    global g_algo
    global g_query_reps
    global g_target_reps
    global g_scores
    q_rep = g_query_reps[q_id]
    for t_id, t_rep in enumerate(g_target_reps):
        score = g_algo.calc_sim(q_rep, t_rep)
        g_scores[q_id][t_id] = score


def calc_scores():
    global g_algo
    global g_query_reps
    global g_target_reps
    global g_scores
    g_scores = np.empty((len(g_query_reps), len(g_target_reps)))
    q_ids = list(range(len(g_query_reps)))
    # pool = Pool()
    # pool.map(calc_scores_worker, q_ids)
    for q_id in q_ids:
        calc_scores_worker(q_id)


def load_images(list_path, align):
    images = []
    for subj, fn in list_path:
        img_path = os.path.join(FERET_DIR, subj, fn)
        img = cv2.imread(img_path)

        if align:
            img = align_feret(img, *t)

        images.append(img)

    return images


def main(args):
    global g_algo
    global g_query_reps
    global g_target_reps
    global g_scores

    module_name = args["module"]
    module = importlib.import_module(module_name)
    params = args["params"]

    align = args["align"]

    if module is None:
        print("Cannot load module: {}".format(module_name))
        exit()

    g_algo = module.create(params)

    if args["output"] is not None:
        output = args["output"]
    else:
        output_file_name = "aligned" + module_name.replace(".", "_") + ".dat"
        output = os.path.join("scores", output_file_name)

    target = load(TARGET_SET)
    query = load(QUERY_SET)

    target_imgs = load_images(target, align)
    query_imgs = load_images(query, align)

    g_target_reps = g_algo.calc_reps(target_imgs)
    g_query_reps = g_algo.calc_reps(query_imgs)

    calc_scores()
    dump(output, g_scores)
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
