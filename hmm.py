"""Simple class for storing HMM data structures"""

from __future__ import division
from hmmUtils import normalize
from math import log
from hmmRandom import randomDist

class HMM(object):
    """Defines a hidden markov model"""

    def __init__(self, init, trans, emit):
        assert len(init) == len(trans)
        assert len(init) == len(emit)
        assert len(init) == len(trans[0])

        self.init = init
        self.trans = trans
        self.emit = emit
        self.nStates = len(init)
        self.nEvents = len(emit[0])

    @classmethod
    def random(cls, nStates, nObs):
        """Creates a random Hidden Markov Model"""
        init = randomDist(nStates)

        trans = [randomDist(nStates) for _ in xrange(nStates)]

        emit = [randomDist(nObs) for _ in xrange(nStates)]

        return HMM(init, trans, emit)

    @classmethod
    def randomStudent(cls):
        """Creates an hmm that correctly models a student.

        It is impossible to forget something.
        If you learn a skill, you are more likely to get the question right than
        wrong.
        If you haven't learnt a skill, it is more likely that you get the
        question wrong.
        """

        hmm = HMM.random(2, 2)

        # This makes it impossible to forget.
        hmm.trans[1] = [0.0, 1.0]

        if hmm.emit[0][0] < hmm.emit[0][1]:
            hmm.emit[0].reverse()
            assert hmm.emit[0][0] >= hmm.emit[0][1]

        if hmm.emit[1][0] > hmm.emit[1][1]:
            hmm.emit[1].reverse()
            assert hmm.emit[1][0] <= hmm.emit[1][1]

        return hmm


    def computeAlphas(self, obs):
        """Performs forward algorithm to build alphas matrix.
        Calculates P(O1 = o1, O2 = o2, ..., Ot = ot,  Qt=i | HMM)
        Builds a list of lists by obs time and then state
        """

        # Create the alpha values at time t = 0
        startAlphas = [self.init[i] * self.emit[i][obs[0]]
                       for i in xrange(self.nStates)]

        alphas = [startAlphas]

        # Go through each time step, adding the values for each state
        for t in xrange(1, len(obs)):

            currentAlphas = []
            for j in xrange(self.nStates):

                # Compute the product of probabilities of transition.
                alphaSum = 0
                for i in xrange(self.nStates):
                    alphaSum += alphas[t-1][i] * self.trans[i][j]

                # Multiply by probability of getting the observation from this
                # state
                currentAlphas.append(alphaSum * self.emit[j][obs[t]])

            alphas.append(currentAlphas)

        assert len(alphas) == len(obs)
        for alphasI in alphas:
            assert len(alphasI) == self.nStates

        # Cover up float errors by normalizing each row
        return alphas


    def computeBetas(self, obs):
        """Performs backward algorithm to build beta matrix

        Calculates P(Ot+1 = ot+1, ..., OT = oT | Qt = i, HMM)
        Builds a list of lists by obs time and then state
        """

        finalBeta = [1 for i in xrange(self.nStates)]

        betas = [finalBeta]

        # goes from (n-1) to 0
        for t in xrange(len(obs) - 2, -1, -1):
            currentBetas = []

            for i in xrange(self.nStates):

                s = 0
                for j in xrange(self.nStates):
                    val = (self.trans[i][j]
                         * self.emit[j][obs[t+1]]
                         * betas[0][j])
                    s += val

                currentBetas.append(s)

            assert len(currentBetas) == self.nStates
            # insert into the front
            betas.insert(0, currentBetas)

        assert len(betas) == len(obs)

        return betas


    def computeGammas(self, alphas, betas):
        """Combines alphas and betas to compute P(Qt = i | O, HMM)"""

        nObs = len(alphas)
        assert len(alphas) == len(betas)

        for states in alphas:
            assert len(states) == self.nStates

        for states in betas:
            assert len(states) == self.nStates

        gammas = []
        for t in xrange(nObs):

            abProd = [alphas[t][i] * betas[t][i] for i in xrange(self.nStates)]

            s = sum(abProd)

            # Normalize values in abProd
            currentGammas = [ab / s for ab in abProd]

            gammas.append(currentGammas)

        assert len(gammas) == nObs
        return gammas


    def computeSigmas(self, betas, gammas, obs):
        """Computes P(Qt = i, Qt+1 = j | O, self)
        Produces a 3d list by t, i, j
        """

        calcSig = lambda i, j, t: (gammas[t][i] * self.trans[i][j] *
                                   self.emit[j][obs[t+1]] * betas[t+1][j] /
                                   betas[t][i])

        sigmas = [[[calcSig(i, j, t)
                   for j in xrange(self.nStates)]
                  for i in xrange(self.nStates)]
                 for t in xrange(len(obs)-1)]

        return sigmas

    def baumWelch(self, obs):
        """Performs the Baum-Welch alogorithm on an HMM with multiple
        observation sequences.
        """

        alphas = [self.computeAlphas(oSeq) for oSeq in obs]
        betas = [self.computeBetas(oSeq) for oSeq in obs]

        gammas = [self.computeGammas(alphas[i], betas[i])
                    for i in xrange(len(obs))]
        sigmas = [self.computeSigmas(betas[i], gammas[i], obs[i])
                    for i in xrange(len(obs))]

        newInit = self.updateInit(gammas)
        newTrans = self.updateTrans(gammas, sigmas)
        newEmit = self.updateEmit(gammas, obs)

        return HMM(newInit, newTrans, newEmit)


    def updateInit(self, gammas):
        """Computes new init probabilities."""

        def sumValues(i):
            """Sums gamma values for init"""
            initSum = 0
            for d in xrange(len(gammas)):
                initSum += gammas[d][0][i]
            return initSum

        init = [sumValues(i) for i in xrange(self.nStates)]

        return normalize(init)


    def updateTrans(self, gammas, sigmas):
        """Calculates updated transformation matrix"""

        trans = []
        for i in xrange(self.nStates):
            transI = []

            gammaSum = 0
            for d in xrange(len(gammas)):
                for t in xrange(len(gammas[d]) - 1):
                    gammaSum += gammas[d][t][i]

            def sigmaSum(i, j):
                """Sums sigma values from i to j"""
                s = 0
                for d in xrange(len(gammas)):
                    for t in xrange(len(gammas[d]) - 1):
                        s += sigmas[d][t][i][j]
                return s

            transI = [sigmaSum(i, j) / gammaSum
                      for j in xrange(self.nStates)]

            s = sum(transI)
            trans.append([t / s for t in transI])

        return trans


    def updateEmit(self, gammas, obs):
        """Calculates the updated emission matrix"""

        emitM = []
        for i in xrange(self.nStates):

            totalGammas = 0
            for d in xrange(len(gammas)):
                for t in xrange(len(gammas[d])):
                    totalGammas += gammas[d][t][i]

            emitIs = []
            for k in xrange(self.nEvents):

                matchingGammas = 0
                for d in xrange(len(gammas)):
                    for t in xrange(len(gammas[d])):
                        if obs[d][t] == k:
                            matchingGammas += gammas[d][t][i]

                emitIs.append(matchingGammas / totalGammas)
            emitM.append(emitIs)
        return emitM

    def sequenceLikelihood(self, observation):
        """Computes the likelihood of a sequence of observations given an HMM"""

        alpha = self.computeAlphas(observation)
        return sum(alpha[-1])

    def logLikelihood(self, observations):
        """Computes the log-likelihood of a sequence of observations for the
        given hmm"""
        logLikelihood = 0
        for o in observations:
            logLikelihood += log(self.sequenceLikelihood(o))

        return logLikelihood

    def filter(self, observation):
        """
            Returns a list of lists of probabilities of each state given the
            observations so far.

            P(Qt = i | o1, o2, ... ot, HMM)

            There is a list of probabilities per observation step.
        """

        # alphasT is P(O1 = o1, ..., Ot = ot, Qt = i | HMM)
        alphasT = self.computeAlphas(observation)

        assert len(alphasT) == len(observation)

        result = []
        for alphas in alphasT:
            assert len(alphas) == self.nStates

            # sumAlphas gives us P(O1 = o1, ..., Ot = ot | HMM)
            sumAlphas = sum(alphas)

            result.append([alpha / sumAlphas for alpha in alphas])

        return result

    def __str__(self):
        """
            Returns a string form of the HMM
        """

        initS = str(self.init) + "\n"

        tranS = ""
        for dist in self.trans:
            tranS += str(dist) + "\n"

        emitS = ""
        for dist in self.emit:
            emitS += str(dist) + "\n"

        return ( "Init:" + "\n" + initS
               + "Trans:" + "\n" + tranS
               + "Emit:" + "\n" + emitS)
