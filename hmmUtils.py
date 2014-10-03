"""Utilities used by the HMM library"""

from __future__ import division
def normalize(xs):
    """Normalizes the given list to sum to 1"""
    s = sum(xs)
    return [x/s for x in xs]

def transpose(matrix):
    """Transposes a matrix

    Params:
    matrix - The matrix to transpose

    Result: The transposed matrix
    """

    return [[matrix[i][j]
            for i in xrange(len(matrix))]
           for j in xrange(len(matrix[0]))]
