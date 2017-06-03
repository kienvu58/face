import numpy as np


BASE_DIR = "D:/"
FERET_DIR = BASE_DIR + "colorferet/data/thumbnails/"
GROUND_TRUTH = BASE_DIR + "colorferet/data/ground_truths/xml/"

# target_set is list of tuples: (subject_id, image_filename)
TARGET_SET = "dat/target.dat"

# query_set is list of tuples: (subject_id, image_filename)
QUERY_SET = "dat/query.dat"

# gallery_set is a subset of target_set
# probe_set is a subset of query_set
# they are list of indices of line in parent set
GALLERY = "dat/gallery.dat"

PROBE_FAFB = "dat/probe_fafb.dat"
PROBE_DUP1 = "dat/probe_dup1.dat"
PROBE_DUP2 = "dat/probe_dup2.dat"

# ids_set corresonding with probe_set is a list of indices in gallery that
# match probe
IDS_FAFB = "dat/ids_fafb.dat"
IDS_DUP1 = "dat/ids_dup1.dat"
IDS_DUP2 = "dat/ids_dup2.dat"


RANKS = list(range(1, 50))
C_RANGE = np.linspace(0, 1, 100)

WIDTH = 512
HEIGHT = 768

LEFT_EYE = [170, 336]
RIGHT_EYE = [340, 336]
MOUTH = [255, 592]
LANDMARKS = np.float32([LEFT_EYE, RIGHT_EYE, MOUTH])
