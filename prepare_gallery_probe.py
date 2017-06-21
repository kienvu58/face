import numpy as np
import os
from random import randint
from utils import *
from config import *
from score import get_face_info


PARTITIONS = "partitions"
FA = "fa.txt"
FB = "fb.txt"
DUP1 = "dup1.txt"
DUP2 = "dup2.txt"

# target set
# index, path, subject

# query set
# index, path, subject

# ids set
# index is index of query, index of corresponding target

# gallery set
# list of indices in target set

# probe set
# list of indices in query set

# ids_probe
# list of corresponding target set

# gallery: 994 images random in fa, fb

# probe fafb: 994 images random in fa, fb (not gallery)

# probe dup1: 736 images in dup1
# probe dup2: 228 images in dup2


def have_ground_truth(subject, path):
    info = get_face_info(subject, path)
    if info is not None:
        return True
    return False


def process_fa_fb():
    fa_file = os.path.join(PARTITIONS, FA)
    fb_file = os.path.join(PARTITIONS, FB)
    fa_dict = {}
    fb_dict = {}
    with open(fa_file, "r") as fa:
        for line in fa:
            subject, path = line.split()
            if have_ground_truth(subject, path):
                fa_dict[subject] = path

    with open(fb_file, "r") as fb:
        for line in fb:
            subject, path = line.split()
            if have_ground_truth(subject, path):
                fb_dict[subject] = path

    target = []
    query = []
    ids = {}
    for key in fa_dict.keys():
        if key in fb_dict.keys():
            if randint(1, 2) == 1:
                target.append((key, fa_dict[key]))
                query.append((key, fb_dict[key]))
            else:
                target.append((key, fb_dict[key]))
                query.append((key, fa_dict[key]))
        else:
            target.append((key, fa_dict[key]))

        ids[key] = len(target) - 1

    return target, query, ids


def have_target(subject, ids):
    return subject in ids.keys()


def process_dup(dup, ids):
    dup_file = os.path.join(PARTITIONS, dup)
    dup_query = []
    with open(dup_file, "r") as fb:
        for line in fb:
            subject, path = line.split()[:2]
            if have_ground_truth(subject, path) and have_target(subject, ids):
                dup_query.append((subject, path))

    return dup_query


def get_true_ids(queries, ids):
    true_ids = []
    for q in queries:
        true_ids.append(ids[q[0]])
    return true_ids


def main():
    target, query, ids = process_fa_fb()

    n_gallery = len(target)
    gallery = list(range(n_gallery))

    n_fafb = len(query)
    probe_fafb = list(range(n_fafb))
    ids_fafb = get_true_ids(query, ids)

    dup1_query = process_dup(DUP1, ids)
    n_dup1 = len(dup1_query)
    probe_dup1 = list(range(n_fafb, n_fafb + n_dup1))
    ids_dup1 = get_true_ids(dup1_query, ids)
    print(probe_dup1)

    dup2_query = process_dup(DUP2, ids)
    n_dup2 = len(dup2_query)
    probe_dup2 = list(range(n_fafb + n_dup1, n_fafb + n_dup1 + n_dup2))
    ids_dup2 = get_true_ids(dup2_query, ids)
    print(probe_dup2)

    query.extend(dup1_query)
    query.extend(dup2_query)

    dump(TARGET_SET, target)
    dump(QUERY_SET, query)
    dump(GALLERY, gallery)
    dump(PROBE_FAFB, probe_fafb)
    dump(PROBE_DUP1, probe_dup1)
    dump(PROBE_DUP2, probe_dup2)
    dump(IDS_FAFB, ids_fafb)
    dump(IDS_DUP1, ids_dup1)
    dump(IDS_DUP2, ids_dup2)


if __name__ == "__main__":
    main()
