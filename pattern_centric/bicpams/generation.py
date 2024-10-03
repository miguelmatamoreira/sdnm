import numpy as np
import math, random, unittest
from .domain import Bicluster, Biclusters
from scipy import sparse


class Generator:

    # ========== PART I: Background Data Generator =========== #
    '''Generates background data from #rows, #cols, #bics, type (null, zeros, #bins, U(min,max), N(u,std))'''
    def __init__(self, nrows=1000, ncols=100, nodes=None, density=1.0, ntargets=0, imbalance=0.0,
                 back='symbolic', mask=None, bins=5, symmetries=False, minV=0, maxV=1, mean=0, std=1, seed=None):

        # A: initialization
        if seed is not None: np.random.seed(seed)
        self.nrows = nrows if nodes is None else nodes
        self.ncols = ncols if nodes is None else nodes
        self.dense = density >= 0.5
        self.bins = bins
        self.back = back
        self.bics = []
        self.symmetries = symmetries
        self.ntargets = ntargets

        # B: fix distributions
        if back == 'random':
            self.bounds = (minV, maxV)
        elif back == 'normal':
            self.bounds = (mean, std)
        elif back == 'symbolic':
            self.bounds = (math.ceil(-bins / 2), math.floor(bins / 2) + 1) if symmetries else (0, bins)
            if symmetries and bins % 2 == 0: self.bins = bins + 1

        # C: generate values
        nelements = self.nrows * self.ncols if self.dense else int(density * self.nrows * self.ncols)
        if mask == 'zeros':
            values = np.zeros(shape=nelements, dtype=float)
        elif mask == 'null':
            values = np.full(nelements, np.nan)
        elif back == 'random':
            values = np.random.rand(nelements) * (maxV - minV) + minV
        elif back == 'normal':
            values = np.random.randn(nelements) * std + mean
        elif back == 'symbolic':
            values = np.random.randint(self.bounds[0], self.bounds[1], size=nelements).astype("float")
        else:
            values = None
            print('background values of discrete data must be in {null,zeros,symbolic,random,normal}')
        if self.dense:
            self.data = values.reshape(self.nrows, self.ncols)
            if density < 1: self.plant_missings(1 - density)
        else:
            rows, cols = np.random.randint(0, self.nrows, size=nelements), np.random.randint(0, self.ncols,
                                                                                             size=nelements)
            self.data = sparse.csr_matrix((values, (rows, cols)), shape=(self.nrows, self.ncols))

        # D: generate targets
        if ntargets > 0:
            self.targets = np.zeros(shape=nrows)
            weights = np.interp(np.arange(ntargets), [0, ntargets - 1], [1, 1 - imbalance])
            self.ptargets = weights / sum(weights)
            start, iclass = 0, 0
            for weight in self.ptargets:
                nelements = int(weight * nrows)
                self.targets[start:(start + nelements)] = iclass
                start, iclass = start + nelements, iclass + 1


    # ========== PART II: Plant Biclusters =========== #
    def plant_biclusters(self, btype='constant', numbics=10,
                         distrows='uniform', prows=(3, 4), distcols='uniform', pcols=(3, 4),
                         contiguous=False, rowdir=True, overlapping=True, minconf=0.7, seed=None):

        # A: #rows and #cols using U(min,max) or N(u,std)
        if seed is not None: np.random.seed(seed)
        bicsrows = np.random.randn(prows[0], prows[1], numbics) if distrows == 'normal' else np.random.randint(prows[0],
                                                                                                               prows[1],
                                                                                                               numbics)
        bicscols = np.random.randn(pcols[0], pcols[1], numbics) if distcols == 'normal' else np.random.randint(pcols[0],
                                                                                                               pcols[1],
                                                                                                               numbics)
        if not overlapping:
            self.overrows, self.overcols = None, None
            if sum(bicsrows) < self.nrows:
                self.overrows = np.arange(self.nrows)
                np.random.shuffle(self.overrows)
            elif sum(bicscols) < self.ncols:
                self.overcols = np.arange(self.ncols)
                if not contiguous: np.random.shuffle(self.overcols)
            else:
                print('Biclusters footprint is too large to place overlapping constraint')

        # B: handle multiple pattern types
        if btype == 'multiple':
            for btype in ['constant', 'additive', 'orderpreserving']:
                inumbics = int(numbics / 3) + (int(numbics % 3) if btype == 'constant' else 0)
                print(btype,inumbics)
                self.plant_biclusters(btype, inumbics, distrows, prows, distcols, pcols, contiguous, rowdir,
                                                   True)
            return self.bics

        else:
            for k in range(numbics):
                nrows, ncols, target = bicsrows[k], bicscols[k], None
                refrows, refcols = nrows if rowdir else ncols, ncols if rowdir else nrows

                # C: generate pattern
                brange = self.bins - 1 if btype == 'additive' else (
                    (math.floor(self.bins / 2)+1) if btype == 'multiplicative' else self.bins)
                if self.symmetries: brange = max(1, math.floor(brange / 2))

                if btype == 'constantoverall':
                    pattern = np.full(shape=refcols, fill_value=np.random.randint(0, self.bins))
                else:
                    pattern = np.random.randint(0, brange, size=refcols)

                idata, adjs = np.empty(shape=(refrows, refcols)), np.full(refrows, 1)
                if btype == 'orderpreserving':
                    inds = pattern.argsort()
                    for i in range(refrows):
                        idata[i] = np.sort(np.random.randint(0, brange + 1, size=refcols))[inds]
                else:
                    if btype == 'additive':
                        adjs = np.random.randint(0, self.bins-np.max(pattern), size=refrows)
                    else:
                        if btype == 'multiplicative':
                            adjs = np.random.randint(1, math.floor((self.bins-1) / max(1, np.max(pattern)))+1, size=refrows)
                        if self.symmetries: adjs = adjs * np.random.choice([-1, 1])
                    for i in range(refrows): idata[i] = pattern + adjs[i] if btype == 'additive' else pattern * adjs[i]
                if not rowdir: idata.transpose()

                # D: generate rows and cols according to overlapping and contiguity constraints
                if self.overrows is None:
                    rows = random.sample(range(self.nrows), nrows)
                else:
                    rows, self.overrows = self.overrows[:nrows], self.overrows[nrows:]
                if self.overcols is not None:
                    cols, self.overcols = self.overcols[:ncols], self.overcols[ncols:]
                elif contiguous:
                    startcol = np.random.randint(0, self.ncols - ncols)
                    cols = range(startcol, startcol + ncols)
                else:
                    cols = random.sample(range(self.ncols), ncols)

                # E: generate targets
                if self.ntargets > 0:
                    if int(self.ptargets[0] * self.nrows) < nrows:
                        target = 0
                    else:
                        target = np.random.randint(0, self.ntargets)
                    nsel = int(nrows * minconf)
                    consequent = np.full(nrows, target)
                    consequent[nsel:] = np.random.randint(0, self.ntargets, nrows - nsel)
                    self.targets[rows] = consequent

                # F: plant bicluster
                if rowdir: cols.sort()
                else: rows.sort()
                for i, ix in enumerate(rows):
                    for j, jx in enumerate(cols):
                        self.data[ix, jx] = idata[i, j]
                self.bics.append(Bicluster(rows=set(rows), cols=set(cols), btype=btype, pattern=pattern, target=target))

        return self.bics

    # ========== PART III: Plant Noise and Missings =========== #
    ''' Plants missing values using the provided ratio (percentage)'''
    def plant_missings(self, rate, bicsonly=False):
        if bicsonly:
            nels, indices = 0, []
            for bic in self.bics:
                indices += [(i, j) for i in bic.rows for j in bic.cols]
                nels += int(rate * bic.nrows * bic.ncols)
        else:
            if self.dense:
                indices = [(i, j) for i in range(self.nrows) for j in range(self.ncols)]
            else:
                indices = list(zip(self.data.row, self.data.col))
            nels = int(rate * self.nrows * self.ncols)
        self.data[tuple(np.transpose(random.sample(indices, nels)))] = np.nan

    ''' Plants noisy elements using the provided ratio (percentage)'''
    def plant_noisy_elements(self, rate):
        amount = int(rate * self.nrows * self.ncols)
        if self.dense:
            indices = [(i, j) for i in range(self.nrows) for j in range(self.ncols)]
        else:
            rows, cols = self.data.nonzero()
            indices = list(zip(rows, cols))
        if len(indices) < amount: print('warning: more noisy elements than non-missing elements')
        r1, r2 = random.sample(indices, min(len(indices), amount)), random.sample(indices, min(len(indices), amount))
        for i in range(len(r1)): self.data[r1[i]] = self.data[r2[i]]

    def _add_value_noise(self, x, minv, maxv, noise, ntype):
        if np.isnan(x): return x
        inoise = noise * (np.random.normal(0, 1) if ntype == 'normal' else random.uniform(-.5, .5))
        return max(minv, min(maxv, x + inoise))

    ''' Adds normal or random noise using the provided degree (percent of amplitude)'''
    def add_noise(self, degree, ntype='random'):
        noise = (self.bounds[1] - self.bounds[0]) * degree
        if self.dense:
            func = np.vectorize(lambda x: self._add_value_noise(x, self.bounds[0], self.bounds[1], noise, ntype))
            self.data = func(self.data)
        else:
            rows, cols = self.data.nonzero()
            for i, j in zip(rows, cols):
                self.data[i, j] = self._add_value_noise(self.data[i, j], self.bounds[0], self.bounds[1], noise, ntype)


class GenerationTests(unittest.TestCase):

    '''def test_generate_dense_data(self):
        gen = Generator(nrows=1000, ncols=100, back='random', ntargets=5, imbalance=0.9)
        print('\nU(0,1) data\n', gen.data, "\nClass frequencies:", gen.ptargets)
        gen.plant_missings(0.2)
        print('\n20% missings planted\n', gen.data)
        gen.add_noise(0.1)
        print('\nup to 10% value deviations\n', gen.data)
        gen.plant_noisy_elements(0.5)
        print('\n50% of noisy elements\n', gen.data)

    def test_generate_sparse_data(self):
        gen = Generator(nodes=8, density=0.2, back='random', ntargets=2, imbalance=0.5)
        print('\nU(0,1) data\n', gen.data, "\nClass frequencies:", gen.ptargets)
        gen.add_noise(0.1)
        print('\nup to 10% value deviations\n', gen.data)
        gen.plant_noisy_elements(0.1)
        print('\n10% of noisy elements\n', gen.data)'''

    def test_plant_biclusters(self):
        gen = Generator(nrows=20, ncols=1000, back='symbolic', symmetries=False, ntargets=2, imbalance=0)
        print('Symbolic data\n', gen.data, '\nClasses before planting', gen.targets)
        bics = gen.plant_biclusters(btype='multiple', numbics=3, distrows='uniform', prows=(4, 6), distcols='uniform',
                                    pcols=(4, 6),
                                    contiguous=False, rowdir=True, overlapping=False)
        print('Classes after planting', gen.targets)
        print('Ordered biclusters', Biclusters.to_string(Biclusters.order(bics), gen.data))


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(unittest.TestLoader().loadTestsFromTestCase(GenerationTests))
