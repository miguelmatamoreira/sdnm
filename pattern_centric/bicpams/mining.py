import math, unittest
import numpy as np, pandas as pd
from .fim import FIM
from .spm import SPM
from .domain import Bicluster, Biclusters
from .evaluation import Significance, DiscriminativePower


class BicPAMS:

    @staticmethod
    def run(data, mapper, miner, closer):
        return miner.run(data, mapper, closer)


class Miner:

    def __init__(self, pattern, stopConditions=None, niterations=1, symmetry=False, partition=False, rowdir=True):
        self.stop = {'mincols': 2, 'minrows': 2, 'minbics': 1, 'minarea': None, 'minsup': None, 'minsig': 0.1, 'minlift': None}
        if stopConditions is not None:
            for istop in stopConditions:
                self.stop[istop] = stopConditions[istop]
        self.pattern = pattern
        self.niterations = niterations
        self.symmetry = symmetry
        self.partition = partition
        self.rowdir = rowdir

    def run(self, data, mapper, closer):
        bics = []
        removalDegree = 0.6
        self.ncols = data.ncols
        if not self.rowdir: data = data.transpose()
        mapper.preprocess(data.data)
        if self.stop['minsig'] is not None:
            self.significance = Significance(mapper.itemdata, mapper.bins)
        if self.stop['minlift'] is not None:
            self.discPower = DiscriminativePower(data)

        print("finding biclusters")
        #print(mapper.itemdata[0:10,0:4])
        for i in range(self.niterations):
            print("=== Iteration %d ==="%i)
            if self.pattern == 'additive' or self.pattern == 'multiplicative' or self.symmetry:
                for j in range(data.ncols):
                    adjust = self._alignBy(mapper.itemdata[:, j], mapper.bins)
                    print('Adjust:',adjust)
                    itemsets, labels = mapper.itemize(pattern=self.pattern, negative=self.symmetry, factors=adjust)
                    print("adjusted:\n",itemsets[:2])
                    bics += self.mine(itemsets, labels)
            elif self.pattern.startswith('constant'):
                bics += self.mine(mapper.itemize(), mapper.bins)
            elif self.pattern == 'orderpreserving':
                bics += self.mine(mapper.order(), bins=data.ncols, spm=True)
            else:
                print('Error: pattern type should be in {constant,additive,multiplicative,orderpreserving}')
            print('masking...')
            mapper.itemdata = self._mask(mapper.itemdata, bics, removalDegree)
        if not self.rowdir:
            data = data.transpose()
            bics = Biclusters.transpose(bics)

        #print(mapper.itemdata[0:10,0:4])
        bics = closer.run(bics)
        Biclusters.compute_patterns(bics, mapper.discata)
        bics = self._validBics(bics)
        Biclusters.reset_keys(bics)
        #Biclusters.compute_string_patterns(bics, data.data)
        return bics

    def mine(self, itemsets, bins, constantOverall=False, spm=False, iclass=-1):
        bics, minsup = [], 0.95
        lowersup = self.stop['minrows'] / len(itemsets)
        if self.stop['minsup'] is not None: lowersup = max(lowersup, self.stop['minsup'])
        '''if iclass < 0 and self.partition:
            for j in range(nclasses): bics += self.mine(itemsets, iclass=j)
            return bics'''

        while minsup >= lowersup:
            if spm:  patterns = SPM(minsup*len(itemsets), self.stop['mincols']).run(itemsets,self.ncols)
            else: patterns = FIM(minsup, self.stop['mincols']).run(itemsets)
            minsup *= 0.9
            print('bics from patterns...')
            bics = self._toBics(patterns, bins, spm)
            #print('satisfying...')
            if not (self._satisfy(bics)): continue
            #print('validating...')
            bics = self._validBics(bics)
            if self._satisfy(bics): break
        #print('satisfied!')
        return bics

    def _mask(self, data, bics, degree):
        posCount = {}
        for bic in bics:
            for row in bic.rows:
                for col in bic.cols:
                    key = (row, col)  # '%d:%d'%(row,col)
                    if key in posCount: posCount[key] += 1
                    else: posCount[key] = 1
        k, limit = 0, int(degree * len(posCount))
        for key in sorted(posCount, key=posCount.__getitem__, reverse=True):
            data[key[0], key[1]] = np.nan
            if k == limit: break
            k += 1
        '''v = posCount.values().sort()
            cutoff = max(2,v[int(len(v)*degree)])
            for key, v in posCount.items():
                if v>=cutoff: 
                    #keys = key.split(':')
                    data[key[0],key[1]]=None'''
        return data

    def _satisfy(self, bics):
        if self.stop['minbics'] is not None:
            if len(bics) > self.stop['minbics']: return True
        if self.stop['minarea'] is not None:
            area = 0  # overlaps estimated to account for 50% of bic elements
            for p in bics: area += len(bics.rows) * len(bics.cols)
            if area > self.stop['minarea'] * 2: return True
        return False

    def _alignBy(self, col, bins):
        res = col.copy()
        if self.symmetry:
            res[res >= 0] = 1
            res[res < 0] = -1
        elif self.pattern == 'additive':
            _max = max(col)
            res = [_max - v for v in col]
        elif self.pattern == 'multiplicative':
            vec = col[col != 0]
            print(vec)
            _lcm = math.lcm(*vec)
            res = [(np.nan if v==0 else _lcm / v) for v in col]
        return res

    def _toBics(self, patterns, bins, spm=False):
        bics, key = [], 0
        if spm:
            for entry in patterns:
                for p in patterns[entry]:
                    bics.append(Bicluster(p[2], list(p[1]), self.pattern, key, rowdir=True, pattern=p[0], metrics={}))
                key += 1
            return bics
        for p in patterns:
            cols, pattern = [], []
            for item in p[0]:
                col = math.floor(item / bins)
                if col not in cols:
                    cols.append(col)
                    pattern.append(item % bins)
            if self.pattern == 'additive':
                pattern = np.array(pattern)-min(pattern)
            elif self.pattern == 'multiplicative':
                vec = np.array(pattern)
                vec = vec[vec>0].astype(int).tolist()
                pattern = np.array(pattern)/math.gcd(*vec)
            bics.append(Bicluster(p[1], cols, self.pattern, key, rowdir=True, pattern=pattern, metrics={}))
            key += 1
        return bics

    def _validBics(self, bics):
        if self.stop['minsig'] is not None:
            bics = self.significance.run(bics, threshold=self.stop['minsig'])
        if self.stop['minlift'] is not None:
            bics = self.discPower.runLift(bics, threshold=self.stop['minlift'])
        return bics


from .domain import Dataset, Biclusters
from .closing import Closer
from .mapping import Mapper
from .generation import Generator


class MiningTests(unittest.TestCase):

    def test_bicpams_synthetic(self):
        for btype in ['multiplicative']: #'orderpreserving','constant','additive'
            self._bicpams_with_pattern(btype)

    def _bicpams_with_pattern(self, btype):
        gen = Generator(nrows=100, ncols=10, back='symbolic', bins=5, seed=0)
        print('Symbolic data\n', gen.data[:3],'...')
        bics = gen.plant_biclusters(btype=btype, numbics=1, distrows='uniform', prows=(20, 30), distcols='uniform',
                                    pcols=(4, 5), contiguous=False, rowdir=True, overlapping=False, seed=0)
        print('Biclusters', Biclusters.to_string(Biclusters.order(bics),gen.data))
        data = Dataset(gen.data)
        mapper = Mapper(strategy='flexible', bins=5, noise=True)
        miner = Miner(pattern=btype, stopConditions={'minrows': 20, 'mincols': 3, 'minbics': 1}, niterations=1)
        closer = Closer()
        bics = BicPAMS.run(data, mapper, miner, closer)
        print(Biclusters.to_string(bics))

    '''def test_bicpams_real(self):
        data = Dataset(data='data/example_constant.arff')
        print(str(data))
        mapper = Mapper(strategy='flexible', noise=True)
        miner = Miner(pattern='constant', stopConditions={'mincols': 3, 'minbics': 3}, niterations=2)
        closer = Closer()
        bics = BicPAMS.run(data, mapper, miner, closer)
        print(Biclusters.to_string(bics))'''


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(unittest.TestLoader().loadTestsFromTestCase(MiningTests))
