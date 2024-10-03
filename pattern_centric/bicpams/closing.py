import unittest

from .domain import Biclusters, Bicluster


class Closer:

    def __init__(self, mergeOverlap=None, filterOverlap=None, order='area'):
        self.mergeOverlap = mergeOverlap
        self.filterOverlap = filterOverlap
        self.criteria = order

    def order(self, bic1, bic2):
        return bic1.relevance(self.criteria) >= bic2.relevance(self.criteria)

    def run(self, bics):
        if self.mergeOverlap is not None:
            print("#bics before merging = %d" % len(bics))
            bics = self.merge_bics(bics)
            print("#bics after merging = %d"%len(bics))
        if self.filterOverlap is not None:
            print("#bics before filtering = %d" % len(bics))
            bics = self.filter_bics(bics)
            print("#bics after filtering = %d" % len(bics))
        bics = sorted(bics, key=lambda x: -x.relevance(self.criteria))
        print("sorted biclusters according to %s"%self.criteria)
        return bics

    def merge_bics(self, bics):
        for k in range(2):
            newbics, removals = self.merge_iteration(bics)
            if len(newbics) == 0: break
            for index in sorted(removals, reverse=True):
                del bics[index]
            bics += newbics
        return bics

    def merge_iteration(self, bics):
        newbics, removals = [], set()
        nbics = len(bics)
        for i in range(nbics):
            bic1 = bics[i]
            for j in range(i + 1, nbics):
                bic2 = bics[j]
                if self.order(bic1, bic2):
                    bic = bic1.merge(bic2, self.mergeOverlap)
                else:
                    bic = bic2.merge(bic1, self.mergeOverlap)
                if bic is not None:
                    newbics.append(bic)
                    removals.add(i)
                    removals.add(j)
        return newbics, removals

    def filter_bics(self, bics):
        removals = set()
        nbics = len(bics)
        for i in range(nbics):
            bic1 = bics[i]
            for j in range(i + 1, nbics):
                bic2 = bics[j]
                if self.order(bic1, bic2):
                    if bic1.merge(bic2, self.filterOverlap) is not None:
                        removals.add(j)
                else:
                    if bic2.merge(bic1, self.filterOverlap) is not None:
                        removals.add(i)
        #print(len(bics))
        #print(removals)
        for index in sorted(removals, reverse=True):
            del bics[index]
        return bics


class ClosingTests(unittest.TestCase):

    def test_merging(self):
        print('MergingTest')
        closer = Closer(mergeOverlap=0.7)
        bic1 = Bicluster(rows=[1,2,3,4], cols=[1,2,3,4])
        bic2 = Bicluster(rows=[2,3,4], cols=[2,3,4])
        bics = closer.run([bic1,bic2])
        print("Single merged bicluster:\n",Biclusters.to_string(bics))

        closer = Closer(mergeOverlap=0.8)
        bics = closer.run([bic1,bic2])
        print("Original biclusters:\n",Biclusters.to_string(bics))

    def test_filtering(self):
        print('FilteringTest')
        closer = Closer(filterOverlap=0.3)
        bic1 = Bicluster(rows=[1,2,3], cols=[1,2,3])
        bic2 = Bicluster(rows=[2,3,4], cols=[2,3,4])
        bics = closer.run([bic1,bic2])
        print("Larger bicluster:\n",Biclusters.to_string(bics))

        closer = Closer(filterOverlap=0.5)
        bics = closer.run([bic1,bic2])
        print("Original biclusters:\n",Biclusters.to_string(bics))


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(unittest.TestLoader().loadTestsFromTestCase(ClosingTests))