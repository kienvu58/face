import numpy as np
from evaluate import *


def main():
    scores = np.array([
        [17, 4, 6, 2, 33, 42],
        [38, 12, 65, 13, 5, 4],
        [34, 41, 16, 22, 23, 2],
        [64, 1, 12, 2, 43, 21]
    ])
    P = [0, 1, 2, 3]
    ids = [3, 5, 5, 3]
    G = [0, 1, 2, 3, 4, 5]
    D, F = partition(scores, 5, P, ids)
    print(D, F)
    # c_range = range(70)
    # ranks = range(1, 4)
    # cms = identity(scores, G, P, ids, ranks)
    # plot_CMC(ranks, cms)
    # P_V, P_F = verify(scores, G, P, ids, c_range)
    # plot_ROC(P_V, P_F)
    # plt.show()
    # print(scores.min(), scores.max())
    


if __name__ == "__main__":
    main()
