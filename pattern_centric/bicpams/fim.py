import numpy as np
import unittest


class FIM:

    def __init__(self, min_support, min_cols=2, apriori=True):
        self.min_rows = min_support
        self.min_cols = min_cols
        self.apriori = apriori

    def run(self, data):
        if not self.apriori:
            print('Vertical FIM not supported in current version. Running Apriori.')
        print('Apriori with sup=%.3f' % self.min_rows)
        L, C1, k = [[]], {}, 0
        ntrans, nitems = len(data), 0

        # A: compute C1
        for row in data:
            if len(row)>0: nitems = max(nitems, row[-1])
        for item in range(nitems + 1): C1[item] = np.full(ntrans, False)
        for i, transaction in enumerate(data):
            for item in transaction: C1[item][i] = True

        # B: compute L1
        for candidate, entry in C1.items():
            sup = entry.sum()
            if (sup / ntrans) >= self.min_rows:
                L[0].append([[candidate], entry, sup, True])

        # C: expand L
        while (True):
            k += 1
            #print('Apriori: %i-th iteration with %i candidates' % (k, len(L[-1])))
            LK = self._computeLK(L[-1], k, ntrans)
            if len(LK) == 0:
                break
            else:
                L.append(LK)

        # D: return patterns
        patterns = []
        for LK in L[self.min_cols - 1:]:
            for P in LK:
                if P[3]: patterns.append((P[0], np.where(P[1])[0]))
        return patterns

    def _computeLK(self, LK_, k, num_trans):
        LK = []
        for i in range(len(LK_)):
            P1 = LK_[i]
            for j in range(i + 1, len(LK_)):
                P2 = LK_[j]
                if P1[0][:k - 1] == P2[0][:k - 1]:  # if first k-1 terms are the same
                    trans = np.logical_and(P1[1], P2[1])
                    sup = trans.sum()
                    if sup / num_trans >= self.min_rows:
                        v1, v2 = P1[0][k - 1], P2[0][k - 1]
                        if P1[2] == sup: P1[3] = False
                        if P2[2] == sup: P2[3] = False
                        LK.append([P1[0] + [v2] if v1 < v2 else P2[0] + [v1], trans, sup, True])  # union
                else: break
            if P1[3]:
                for P3 in LK:
                    if P1[2] == P3[2]:
                        if self._subset(P1[0], P3[0]):
                            P1[3] = False
                            break
        return LK

    def _subset(self, l1, l2):
        ln, j = len(l1), 0
        for ele in l2:
            if ele == l1[j]: j += 1
            if j == ln: return True
        return False

    def eclat(self, data, min_support, min_cols):
        return None


class FIMTests(unittest.TestCase):

    def test_fim(self):
        patterns = FIM(0.5, 2).run([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4]])
        print("First dataset", patterns)
        patterns = FIM(0.5, 2).run([[5, 2, 4, 1], [2, 4, 5, 6], [0, 9], [1, 2, 4]])
        print("Second dataset", patterns)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(unittest.TestLoader().loadTestsFromTestCase(FIMTests))