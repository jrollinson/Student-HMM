"""Functions for running the baum welch algorithm on a model """

from __future__ import with_statement
import functools
from multiprocessing import Pool

def baumWelchRestarts(models, observationSeqs, stopping,
                      parallel=True):
    """
    Performs the baum welch algorithm on nRuns random models and returns the
    result with highest likelihood

    Arguments:
        models: The starting models for baumWelch
        observationSeqs: a list of observation sequences.
        nRuns: The number of runs
        stopping: The difference in log likelihood between two steps required to
            keep iterating.
        parallel: Whether to run the runs in parallel

    Returns: (model, logLikelihood)
        model: The model with the highest log likelihood.
        logLikelihood: The log likelihood of that model
    """

    if parallel:

        arguments = [(model, observationSeqs, stopping) for model in models]

        with Pool() as pool:
            resultModels = list(pool.map(baumWelchStoppingStar, arguments))

    else:
        resultModels = [baumWelchStopping(model, observationSeqs, stopping)
                        for model in models]

    # Pick the hmm with the maximum log likelihood
    (maxLogLikelihood, maxModel) = resultModels[0]

    for (logLikelihood, model) in resultModels[1:]:
        if logLikelihood > maxLogLikelihood:
            maxLogLikelihood = logLikelihood
            maxModel = model

    return (maxModel, maxLogLikelihood)

def baumWelchStoppingStar(args):
    """
    Calls baumWelchStopping with arguments in tuple args
    """
    return baumWelchStopping(*args)

def baumWelchStopping(model, observationSeqs, stopping):
    """
    Performs the baum welch algorithm on a random model until the change in log
    likelihood is less than stopping.

    Parameters:
        model: The starting model
        observationSeqs: The list of observation sequences
        stopping: The change in log likelihood required to keep going.

    Returns: (model, logLikelihood)
        model: The final model
        logLikelihood: The log likelihood of the final model
    """

    logLikelihood = model.logLikelihood(observationSeqs)

    while True:
        model = model.baumWelch(observationSeqs)

        prevLogLikelihood = logLikelihood
        logLikelihood = model.logLikelihood(observationSeqs)

        if prevLogLikelihood > logLikelihood:
            print "ERROR on loglikelihood. logLikelihood:", logLikelihood,
            print "prevLogLikelihood:", prevLogLikelihood

        if abs(logLikelihood - prevLogLikelihood) < stopping:
            break

    return (logLikelihood, model)
