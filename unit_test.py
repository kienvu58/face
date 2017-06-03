import unittest
from evaluate import *

scores_1 = np.array([
    [17, 4, 6, 2, 33, 42],
    [38, 12, 65, 13, 5, 4],
    [34, 41, 16, 22, 23, 2],
    [64, 1, 12, 2, 43, 21]
])
P_1 = [1, 2, 3]
ids_1 = [2, 4, 5]
G_1 = [2, 3, 4, 5]


class TestEvaluate(unittest.TestCase):

    def test_sort(self):
        output = sort(scores_1, G_1, P_1)
        expected_output = {
            1: [5, 4, 3, 2],
            2: [5, 2, 3, 4],
            3: [3, 2, 5, 4]
        }
        self.assertEqual(output, expected_output)

    def test_partition(self):
        output = partition(scores_1, 4, P_1, ids_1)
        expected_output = ([23], [5, 43])
        self.assertEqual(output, expected_output)

    def test_identity(self):
        output = identity(scores_1, G_1, P_1, ids_1, range(1, 5))
        expected_output = [0, 0, 1 / 3, 1]
        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    unittest.main()
