"""
scores: ndarray mxn
    row_idx: query image index
    col_idx: target image index
    value: score
"""
from __future__ import print_function
import numpy as np
import argparse
from config import *
from utils import *
import score


def sort(scores, G, P):
    """
    scores: all similarity scores between images in gallery and probe set
    G: indices of images in virtual gallery
    P: indices of images in virtual probe set

    return a dictionary
        key: virtual probe index
        value: list of virtual gallery index sorted by score (smaller first)
    """
    sorted_scores = {}
    for idx in P:
        scores_idx = scores[idx].transpose()[G]
        sorted_ids = np.argsort(scores_idx)
        sorted_scores[idx] = np.array(G)[sorted_ids].tolist()
    return sorted_scores


def identity(scores, G, P, ids, ranks):
    """
    scores: all similarity scores between images in gallery and probe set
    G: indices of images in virtual gallery
    P: indices of images in virtual probe set
    ids: give the indices of the gallery images of the person in probes
    ranks: top k-th smallest score

    return performance score: R_k / len(P)
    """
    ret = []

    for k in ranks:
        sorted_scores = sort(scores, G, P)
        R_k = 0
        for i, probe_idx in enumerate(P):
            idx = ids[i]
            if idx in sorted_scores[probe_idx][:k]:
                R_k += 1

        if len(P) == 0:
            ret.append(0.0)
        else:
            ret.append(float(R_k) / len(P))
    return ret


def partition(scores, G_idx, P, ids):
    """
    Divide P into two disjoint D and F.
    D consists of all probes p such that p ~ g_i
    F consists of all probes p such that p not ~ g_i
    """
    similar_idx = np.array(ids) == G_idx
    D = (np.array(P)[np.where(similar_idx)]).tolist()
    F = (np.array(P)[np.where(~similar_idx)]).tolist()
    scores_current = scores.transpose()[G_idx]
    D = scores_current[D].tolist()
    F = scores_current[F].tolist()
    ret = D, F
    return ret


def count(c, partition):
    """
    return number of scores smaller than c in partition
    """
    n = 0
    for e in partition:
        if e < c:
            n += 1

    return n


def verify(scores, G, P, ids, c_range):
    """
    scores: all similarity scores between images in gallery and probe set
    G: indices of images in virtual gallery
    P: indices of images in virtual probe set
    ids: give the indices of the gallery images of the person in probes
    c_range: threshold range

    return P_V^c and P_F^c
    """
    P_V = []
    P_F = []
    for c in c_range:
        P_Vi = []
        P_Fi = []
        len_Di = []
        len_Fi = []
        for gallery_idx in G:
            D, F = partition(scores, gallery_idx, P, ids)

            if len(D) == 0:
                P_Vi.append(0)
                len_Di.append(0)
            else:
                P_Vi.append(count(c, D) / len(D))
                len_Di.append(len(D))

            if len(F) == 0:
                P_Fi.append(0)
                len_Fi.append(0)
            else:
                P_Fi.append(count(c, F) / len(F))
                len_Fi.append(len(F))

        sum_Di = sum(len_Di) + 1
        sum_Fi = sum(len_Fi) + 1
        P_V.append(1.0 / sum_Di * sum([a * b for a, b in zip(len_Di, P_Vi)]))
        P_F.append(1.0 / sum_Fi * sum([a * b for a, b in zip(len_Fi, P_Fi)]))

    return P_V, P_F


def normalize_scores(scores):
    """
    scale scores in range [0, 1]
    """
    min_score = scores.min()
    max_score = scores.max()
    scale_scores = (scores - min_score) / (max_score - min_score)
    return scale_scores


def evaluate(scores, gallery, probe, ids):
    """
    return a dict:
        {
            "cmc": ([cumulative match score], [rank]),
            "roc": ([P_V], [P_F])
        }
    """
    cms = identity(scores, gallery, probe, ids, RANKS)
    P_V, P_F = verify(scores, gallery, probe, ids, C_RANGE)
    ret = {
        "cmc": (RANKS, cms),
        "roc": (P_F, P_V)
    }
    return ret


def evaluate_from_file(scores_file_name):
    scores = load(scores_file_name)
    scores = normalize_scores(scores)
    gallery = load(GALLERY)
    probe_fafb = load(PROBE_FAFB)
    probe_dup1 = load(PROBE_DUP1)
    probe_dup2 = load(PROBE_DUP2)
    ids_fafb = load(IDS_FAFB)
    ids_dup1 = load(IDS_DUP1)
    ids_dup2 = load(IDS_DUP2)
    fafb = evaluate(scores, gallery, probe_fafb, ids_fafb)
    dup1 = evaluate(scores, gallery, probe_dup1, ids_dup1)
    dup2 = evaluate(scores, gallery, probe_dup2, ids_dup2)
    ret = {
        "fafb": fafb,
        "dup1": dup1,
        "dup2": dup2
    }
    return ret


def main(args):
    output_eval = args["output_eval"]
    output_score = args["output_score"]
    if args["algo"]:
        module_name = args["input"]
        score_args = {"module": module_name,
                      "params": args["params"],
                      "output": output_score,
                      "align": args["align"]}
        scores_file_name = score.main(score_args)
    else:
        scores_file_name = args["input"]

    result = evaluate_from_file(scores_file_name)
    dump(output_eval, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate algorithm.")
    parser.add_argument("input", type=str,
                        help="Input scores file or algorithm")
    parser.add_argument("-a", "--algo", dest="algo", action="store_true",
                        help="Evaluate algorithm")
    parser.add_argument("-na", "--no-algo", dest="algo", action="store_false",
                        help="Evaluate scores from file")
    parser.add_argument("-oe", "--output-eval", type=str, required=True,
                        help="Evaluation output file")
    parser.add_argument("-os", "--output-score", type=str,
                        help="Scores output file")
    parser.add_argument("-p", "--params", type=str,
                        help="Optional params string to run with algorithm")
    parser.add_argument("-al", "--align", dest="align", action="store_true",
                        help="Score aligned images")
    main(vars(parser.parse_args()))
