"""Provides random functions for HMM library"""

from __future__ import division
from random import random

def randomDist(length):
    """Creates a random list of floats that add up to 1."""
    randomNs = [random() for _ in xrange(length)]
    s = sum(randomNs)

    return [n / s for n in randomNs]
