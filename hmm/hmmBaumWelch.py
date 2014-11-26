"""Library for running the baum welch algorithm on sequences with restarts and a
stopping criteria"""

from hmm import HMM
from baumWelch import baumWelchRestarts

def hmmBaumWelchIndividual(nRuns, stopping, obsSeqs):
    '''Performs the baum welch algorithm nRuns times on each observation'''
    hmmLL = [hmmBaumWelchRestarts(nRuns, stopping, [obs], parallel=False)
             for obs in obsSeqs]

    return sum([ll for (_, ll) in hmmLL])


def hmmBaumWelchRestarts(nRuns, stopping, obs, parallel=True):
    '''Performs the baum welch nRuns times stopping when the likelihood
    changes by less than stopping'''

    models = [HMM.randomStudent() for _ in xrange(nRuns)]

    return baumWelchRestarts(models, obs, stopping, parallel)
