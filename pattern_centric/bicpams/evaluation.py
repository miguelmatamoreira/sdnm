import unittest, math, numpy as np
from scipy.stats import binom

class DiscriminativePower:

    def __init__(self, data):
        self.targets = data.targets
        self.ntargets = len(data.targets)
        self.nclasses = len(data.targetnames)
        self.classSup = np.zeros(self.nclasses)
        for t in self.targets: self.classSup[t]+=1

    def runLift(self, bics, threshold=1.05):
        result = []
        for bic in bics:
            sups, lifts = np.zeros(self.nclasses), np.zeros(self.nclasses)
            for row in bic.rows:
                sups[self.targets[row]] += 1
            for k in range(self.nclasses):
                lifts[k] = sups[k] / (bic.nrows * (self.classSup[k]/self.ntargets))
            if max(lifts) < threshold: continue
            bic.metrics['lift'] = lifts
            result.append(bic)
        return result

class Significance:

    def __init__(self, data, bins):
        self.dist = []
        self.bins = bins
        self.N = len(data)
        for k in range(len(data[0])): self.dist.append({})
        for row in data:
            for j, v in enumerate(row):
                if v in self.dist[j]:
                    self.dist[j][v] += 1
                else:
                    self.dist[j][v] = 1
        for col in self.dist:
            for v in col: col[v] = col[v] / self.N

    def run(self, bics, threshold=0.1):
        result = []
        for bic in bics:
            prob = 0
            if bic.btype == 'orderpreserving':
                prob = 1/math.factorial(len(bic.cols))
            else:
                p = np.array(bic.pattern)
                patterns = [p]
                if bic.btype == 'additive':
                    diff = self.bins-max(bic.pattern)
                    for i in range(1, diff):
                        patterns.append(p+i)
                elif bic.btype == 'multiplicative':
                    diff = math.floor(self.bins/max(bic.pattern))
                    for i in range(2, diff):
                        patterns.append(p*i)
                for pattern in patterns:
                    prob += self._get_prob(bic.cols, pattern)
            pvalue = 1 - binom.cdf(len(bic.rows), self.N, prob)
            if pvalue > threshold: continue
            bic.metrics['significance'] = pvalue
            result.append(bic)
        return result
    
    def _get_prob(self, cols, pattern):
        prob = 1
        for i, col in enumerate(cols):
            col_dist = self.dist[col]
            prob = prob * (col_dist[pattern[i]] if pattern[i] in col_dist else 0.01)
        return prob


class EvaluationTests(unittest.TestCase):

    def test_significance(self):
        for btype in ['constant', 'additive', 'orderpreserving', 'multiplicative']:
            self._significance_with_pattern(btype)

    def _significance_with_pattern(self, btype):
        from generation import Generator
        gen = Generator(nrows=400, ncols=20, back='symbolic', bins=5)
        print('Symbolic data\n', gen.data)

        from domain import Biclusters
        bics = gen.plant_biclusters(btype=btype, numbics=2, distrows='uniform', prows=(10, 20), distcols='uniform',
                                    pcols=(3, 5), contiguous=False, rowdir=True, overlapping=False)
        print('Biclusters', Biclusters.to_string(Biclusters.order(bics)))
        significance = Significance(gen.data, gen.bins)
        bics = significance.run(bics, threshold=1)
        print('Biclusters with significance', Biclusters.to_string(Biclusters.order(bics), gen.data))


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(unittest.TestLoader().loadTestsFromTestCase(EvaluationTests))
