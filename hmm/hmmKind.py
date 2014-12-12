"""Class for HMM kinds with baum welch algorithm"""

from __future__ import division
from hmmUtils import normalize, transpose
from hmm import HMM
from math import log
from hmmRandom import randomDist

import math

class HMMKind(object):
    """ A model of multiple kinds of HMMs """

    def __init__(self, hmms, pKinds):
        assert len(hmms) == len(pKinds)

        self.hmms = hmms
        self.pKinds = pKinds

        self.nStates = self.hmms[0].nStates
        self.nEvents = self.hmms[0].nEvents

    @classmethod
    def randomStudent(cls, nKinds):
        """ Returns a random student HMMKind """
        hmms = [HMM.randomStudent() for _ in xrange(nKinds)]
        pKinds = randomDist(nKinds)

        return HMMKind(hmms, pKinds)

    def baumWelch(self, obsDT):
        """Performs the baum-welch algorithm to improve the likelihood of the
        given observation sequences.

        Params:
        hmms - The hidden markov models to update
        kindPs - A list of the probabilities for each hmm given
        obs - A list of observation sequences

        Result: (hmms', kindPs') - Updated hmms and kind probabilities
        """

        kindPsDH = self.calcKindPs(obsDT)
        kindPsHD = transpose(kindPsDH)

        # Update kindPs
        newKindPs = [self.updateKindP(kindPsD) for kindPsD in kindPsHD]

        # Update hmmsK
        newHmmsK = [self.baumWelchKind(self.hmms[k], kindPsHD[k], obsDT)
                   for k in xrange(len(self.hmms))]

        return HMMKind(newHmmsK, newKindPs)


    def calcKindPs(self, obsDT):
        """Calculates the kind probabilities conditioned on observation sequence
        and HMM

        Params:
        hmms - list of K hidden markov models
        kindPs - List of K kind probabilities
        obs - A list of observation sequences
        """

        unNormedkindPsDH = [[self.hmms[k].sequenceLikelihood(obsT) *
            self.pKinds[k]
                            for k in xrange(len(self.hmms))]
                            for obsT in obsDT]

        return [normalize(kindPsH) for kindPsH in unNormedkindPsDH]


    def updateKindP(self, kindPsD):
        """Calculates the updated pKind

        Params:
        kindPsD - A list of kind probabilities, one for each observation

        Result: updated kindP
        """
        return sum(kindPsD) / len(kindPsD)



    def baumWelchKind(self, hmm, kindPsD, obsDT):
        """Performs the augmented baum-welch algorithm to update the hmm

        Params:
        hmm - The hidden markov model to update
        kindPsD - The kind Probabilities conditioned on each observation
                  sequence
        obsDT - The observation sequences
        """

        alphasDTI = [hmm.computeAlphas(obsT) for obsT in obsDT]
        betasDTI = [hmm.computeBetas(obsT) for obsT in obsDT]

        gammasDTI = [hmm.computeGammas(alphasDTI[d], betasDTI[d])
                  for d in xrange(len(obsDT))]

        sigmasDTIJ = [hmm.computeSigmas(betasDTI[d], gammasDTI[d], obsDT[d])
                  for d in xrange(len(obsDT))]

        newInit = self.updateInitKind(hmm, kindPsD, gammasDTI)
        newTrans = self.updateTransKind(hmm, kindPsD, gammasDTI, sigmasDTIJ)
        newEmit = self.updateEmitKind(hmm, kindPsD, gammasDTI, obsDT)

        return HMM(newInit, newTrans, newEmit)

    def updateInitKind(self, hmm, kindPsD, gammasDTI):
        """Calculates the updated initial distribution

        Params:
        hmm - The original hidden markov model
        kindPsD - The conditional probability of this kind for each observation
                  sequence.
        gammasDTI - The gamma values for each observation sequence.
        """

        newInit = []
        for i in xrange(hmm.nStates):
            s = 0
            for d in xrange(len(gammasDTI)):
                s += kindPsD[d] * gammasDTI[d][0][i]
            newInit.append(s)

        totalSum = sum(newInit)

        return [p / totalSum for p in newInit]


    def updateTransKind(self, hmm, kindPsD, gammasDTI, sigmasDTIJ):
        """Calculates the updated transformation matrix

        Params:
        hmm - original HMM
        kindPsD - Probability of this kind for each observation sequence
        gammasDIT - gamma values
        sigmasDTIJ - sigma values
        """

        newTrans = []
        for i in xrange(hmm.nStates):

            gammaSum = 0
            for d in xrange(len(gammasDTI)):
                dGammaSum = 0
                for t in xrange(len(gammasDTI[d]) - 1):
                    dGammaSum += gammasDTI[d][t][i]
                gammaSum += dGammaSum * kindPsD[d]

            transI = []
            for j in xrange(hmm.nStates):

                sigmaSum = 0
                for d in xrange(len(sigmasDTIJ)):
                    dSigmaSum = 0
                    for t in xrange(len(sigmasDTIJ[d])):
                        dSigmaSum += sigmasDTIJ[d][t][i][j]
                    sigmaSum += dSigmaSum * kindPsD[d]

                transI.append(sigmaSum / gammaSum)
            newTrans.append(transI)

        return newTrans


    def updateEmitKind(self, hmm, kindPsD, gammasDTI, obsDT):
        """Calculates the updated emission matrix

        Params:
        hmm - original HMM
        kindPsD - Probability of this kind for each observation sequence
        gammasDIT - gamma values
        obsDT - List of observation sequences
        """

        newEmit = []
        for i in xrange(hmm.nStates):

            gammaSum = 0
            for d in xrange(len(gammasDTI)):
                dGammaSum = 0
                for gammasI in gammasDTI[d]:
                    dGammaSum += gammasI[i]
                gammaSum += dGammaSum * kindPsD[d]

            emitI = []
            for o in xrange(hmm.nEvents):

                eventGammaSum = 0
                for d in xrange(len(gammasDTI)):
                    dGammaSum = 0
                    for t in xrange(len(gammasDTI[d])):
                        if obsDT[d][t] == o:
                            dGammaSum += gammasDTI[d][t][i]
                    eventGammaSum += dGammaSum * kindPsD[d]

                emitI.append(eventGammaSum / gammaSum)
            newEmit.append(emitI)
        return newEmit

    def logLikelihood(self, observations):
        """Computes the log-likelihood of a sequence of observations given a set
        of HMMS and their probabilities.

        Params:
        hmms: List of hmms
        kindPs: List of probabilities corresponding to each hmm
        obs: A sequence of observation sequences
        """

        logLikelihood = 0
        for obsT in observations:

            observationLikelihood = 0
            for k in xrange(len(self.hmms)):
                observationLikelihood += (self.pKinds[k] *
                                          self.hmms[k].sequenceLikelihood(obsT))

            logLikelihood += log(observationLikelihood)

        return logLikelihood

    def filter(self, observation):
        """
            Returns a list (one per observation) of lists (one per kind) of
            of probabilities of each state given the observations so far.

            P(Qt = qt, kind=k | model)
        """

        # This equals P(O1 = o1, ..., Ot = ot, Qt = q, kind | model)
        obsStatePTK = []
        for kind in xrange(len(self.hmms)):
            # This provides P(O1 = o1, ..., Ot = ot, Qt = q | model, kind)
            alphasT = self.hmms[kind].computeAlphas(observation)

            for t in xrange(len(alphasT)):
                for i in xrange(len(alphasT[t])):
                    # This moves 'kind' from the conditional
                    alphasT[t][i] *= self.pKinds[kind]

            obsStatePTK.append(alphasT)

        result = []
        for t in xrange(len(observation)):

            obsProbs = 0.0 # P(O1=o1, ..., Ot=ot | model)
            for kind in xrange(len(self.hmms)):
                for state in xrange(self.nStates):
                    obsProbs += obsStatePTK[kind][t][state]

            obsResult = [] # Result for this observation
            for kind in xrange(len(self.hmms)):
                kindResult = [prob / obsProbs for prob in obsStatePTK[kind][t]]
                obsResult.append(kindResult)

            result.append(obsResult)

        return result

    def rmsq(self, observations):
        """
            Returns the root mean squared error for the given set of
            observations.
        """
        squaredError = 0.0
        n = 0

        # Iterate through the observation sequences
        for observationSeq in observations:

            filterRes = self.filter(observationSeq)

            for t in xrange(len(observationSeq)):

                observation = observationSeq[t]

                if t == 0:
                    p_prev_states = [[self.pKinds[i] * pinit
                                      for pinit in self.hmms[i].init]
                                     for i in xrange(len(self.pKinds))]
                else:
                    p_prev_states = filterRes[t-1]

                observationProb = 0.0
                for k in xrange(len(self.hmms)):
                    hmm = self.hmms[k]
                    for i in xrange(hmm.nStates):

                        state_prob = 0.0
                        for prev_state in xrange(hmm.nStates):
                            state_prob += (hmm.trans[prev_state][i] *
                                           p_prev_states[k][prev_state])

                        observationProb += (state_prob *
                                            hmm.emit[i][observation])

                squaredError += (1.0 - observationProb) ** 2
                n += 1

        return  math.sqrt(squaredError / n)

    def __str__(self):
        """ String representation """

        result = []
        for i in xrange(len(self.hmms)):
            result.append("P" + str(i) + ": " + str(self.pKinds[i]) + "\n")
            result.append(str(self.hmms[i]))
            result.append("\n")

        return "".join(result)
