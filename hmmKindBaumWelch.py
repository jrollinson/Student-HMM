"""Library for running baum welch kinds algorithm multiple times and with a
given stopping parameter"""

from hmmKind import HMMKind

from baumWelch import baumWelchRestarts

def baumWelchKindsRestarts(nRuns, stopping, nKinds, obsDT, parallel=True):
    """
    Runs the baum welch algorithm nRuns times, returning the one with the
    largest likelihood.

    Params:
    nRuns: Number of runs
    stopping: change in likelihood to stop at
    nKinds: number of kinds of students
    obsDT: sequence of observation sequences
    """

    models = [HMMKind.randomStudent(nKinds) for _ in xrange(nRuns)]

    return baumWelchRestarts(models, obsDT, stopping, parallel)

