import numpy as np, pandas as pd
import math, unittest
from scipy import sparse, stats
from .domain import Dataset


class Mapper:

    def __init__(self, strategy='flexible', bins=5, noise=True, removals={}, outliers=False, impute=False):
        self.strategy = strategy
        self.bins = bins
        self.noise = noise
        self.removals = removals
        self.outliers = outliers
        self.impute = impute

    def preprocess(self, data):
        tempdata = self.imputeMissings(data.copy()) if self.impute else data.copy()
        self.itemdata, self.noise = self.discretize(tempdata)
        #print(self.itemdata)
        print("removing non-relevant items")

        if type(self.removals) is list:
            for removal in self.removals:
                self.itemdata = np.where(np.isin(self.itemdata, removal), np.nan, self.itemdata)
                self.noise = np.where(np.isin(self.noise, removal), np.nan, self.noise)
        else:
            for cols,removal in self.removals.items():
                if isinstance(data, sparse.csr_matrix):
                    self.itemdata.data[self.itemdata.data in removal] = np.nan
                else:
                    self.itemdata[:,cols]=np.where(np.isin(self.itemdata[:,cols],removal), np.nan, self.itemdata[:,cols])
                self.noise[:,cols]=np.where(np.isin(self.noise[:,cols], removal), np.nan, self.noise[:,cols])
        self.discata = self.itemdata.copy()

    def itemize(self, pattern='constant', negative=False, factors=None):
        ditem, dnoise = self.itemdata.copy(), self.noise.copy()
        labels = self.bins
        if pattern == 'multiplicative':
            labels = self.bins * np.nanmax(factors)
        elif pattern == 'additive':
            labels = self.bins + np.nanmax(factors)
        if negative:
            labels, up = self.bins * 2, math.floor(self.bins / 2)
            ditem, dnoise = ditem + up, ditem + up
        if factors is not None:
            ncols = self.itemdata.shape[1]
            if pattern == 'additive':
                ditem, dnoise = ditem + factors, dnoise + factors
            else:
                for j in range(ncols):
                    ditem[:,j], dnoise[:,j] = ditem[:,j] * factors, dnoise[:,j] * factors
        nrows, ncols = ditem.shape
        transactions = []
        for i in range(nrows):
            transaction = []
            for j in range(ncols):
                for v in [ditem[i, j], dnoise[i, j]]:
                    if not (np.isnan(v)):
                        transaction.append(int(v + j * labels))
            transactions.append(transaction)
        return transactions

    def order(self):
        order = []
        for row in self.itemdata:
            irow = pd.Series(range(len(row))).groupby(row, sort=True).apply(list).tolist()
            order.append(irow)
        #print(order)
        return order

    def _cutoffs(self, gaussian=True, delta=0.2):
        d = (1.0 / self.bins) * delta
        percentils, noiseintervals = [], []
        for i in range(self.bins + 1):
            v = i / self.bins
            percentils.append(stats.norm.ppf(v) if gaussian else v)
            if i == 0 or i == self.bins: continue
            low, high = stats.norm.ppf(v - d) if gaussian else v - d, stats.norm.ppf(v + d) if gaussian else v + d
            noiseintervals.append((low, high))
        print("Percentils:",percentils)
        return percentils, noiseintervals

    def discretize(self, data):  # frequency,width,normal
        print("discretizing input data if real-valued")
        pd.options.mode.chained_assignment = None

        # A: handle sparse data (homogeneous)
        if isinstance(data, sparse.csr_matrix):
            unique = set(data.data)
            if len(unique) <= 10: return data, None
            npdata = np.array(data.data)
            if self.strategy == 'flexible' or 'norm':
                npdata = (npdata - npdata.mean()) / npdata.std()
            else:
                print("error: non-support strategy for sparse data, please use 'flexible'")

            discdata, noise = np.empty(shape=len(npdata)), np.full(shape=len(npdata), fill_value=np.nan)
            dbins, noisebins = self._cutoffs(gaussian=True)
            for i in range(self.bins):
                with np.errstate(invalid='ignore'):
                    mask1 = (npdata > dbins[i]) & (npdata <= dbins[i + 1])
                discdata[mask1] = i
                if i > 0:
                    with np.errstate(invalid='ignore'):
                        mask2 = npdata < noisebins[i - 1][1]
                    noise[mask1 & mask2] = i - 1
                if i < self.bins - 1:
                    with np.errstate(invalid='ignore'):
                        mask2 = npdata > noisebins[i][0]
                    noise[mask1 & mask2] = i + 1
            data.data = discdata.tolist()
            #print(data.data[:3], noise[:3])
            return data, noise

        # B: handle integer variables (choose encoding)
        minIntCardinality = 10
        mask = data.dtypes.isin([int, np.int32, pd.Int64Dtype()])
        for col in data.loc[:, mask].columns:
            if data[col].nunique() > minIntCardinality: data[col] = data[col].astype('Float64')

        # C: handle categorical variables (remove oversized categories)
        maxCategoryProb = 0.8
        mask = data.dtypes == 'category'
        catdata = data.loc[:, mask]
        for col in catdata.columns:
            catdata[col] = catdata[col].cat.codes
            catdata.replace(-1, np.nan, inplace=True)
            counts = catdata[col].value_counts(normalize=True)
            if counts[0] >= maxCategoryProb: catdata[col].replace(counts.index[0], np.nan, inplace=True)
        data[data.columns[mask]] = catdata.astype('Int64')

        # D: normalize numeric variables
        mask = data.dtypes.isin([np.float64, np.float32, pd.Float64Dtype()])
        numdata = data.loc[:, mask]
        if len(numdata.columns)>0:
            dbins, noisebins = None, None
            print(self.strategy)
            if self.strategy == 'flexible': self.strategy = 'normcols'
            if self.strategy.startswith('norm'):
                dbins, noisebins = self._cutoffs(gaussian=True)
                if self.strategy == 'normcols':
                    numdata = (numdata - numdata.mean()) / numdata.std()
                elif self.strategy == 'normrows':
                    dataT = numdata.transpose()
                    dataT = (dataT - dataT.mean()) / dataT.std()
                    numdata = pd.DataFrame(dataT.transpose(), columns=data.columns)
                elif self.strategy == 'normall':
                    print("1. NUM DATA\n", numdata)
                    matrix = numdata.to_numpy()  # overall normalization
                    numdata = pd.DataFrame((matrix - matrix.mean()) / matrix.std(), columns=data.columns)
                    print("2. NUM DATA\n", numdata)
                else:
                    print('error: unknown itemization or normalization strategy')
            elif self.strategy == 'width':
                dbins, noisebins = self._cutoffs(gaussian=False)
                numdata = (data - data.min()) / (data.max() - data.min())
            elif self.strategy == 'rowsfrequency':
                res = data.apply(lambda x: pd.qcut(x, self.bins, labels=False), axis=1) #duplicates='drop'
                return res.to_numpy(dtype='float32'), np.full(shape=data.shape, fill_value=np.nan)
            elif self.strategy == 'frequency':
                if self.noise: print("warning: frequency strategy not compatible with multi-item assignments")
                for col in np.array(data.columns)[mask]:
                    data[col] = pd.qcut(data[col], q=self.bins, labels=False)
                return data.to_numpy(dtype='float32'), np.full(shape=data.shape, fill_value=np.nan)
            else:
                print('error: unknown itemization strategy')

            # E: noise-tolerant discretization
            print("2. TRANSF DATA\n", numdata)
            npdata, npdiscdata = numdata.to_numpy(na_value=np.nan), np.empty(shape=numdata.shape)
            numnoise = np.full(shape=numdata.shape, fill_value=np.nan)
            for i in range(self.bins):
                with np.errstate(invalid='ignore'):
                    mask1 = (npdata > dbins[i]) & (npdata <= dbins[i + 1])
                npdiscdata[mask1] = i
                if i > 0:
                    with np.errstate(invalid='ignore'):
                        mask2 = npdata < noisebins[i - 1][1]
                    numnoise[mask1 & mask2] = i - 1
                if i < self.bins - 1:
                    with np.errstate(invalid='ignore'):
                        mask2 = npdata > noisebins[i][0]
                    numnoise[mask1 & mask2] = i + 1
            print("3. DISC DATA\n", npdiscdata)
            # print('=======\n',npdiscdata,"\n",numnoise)
            # numdata = numdata.apply(pd.cut, bins=dbins, labels=[*range(self.bins)])

        noise, npdata = np.full(shape=data.shape, fill_value=np.nan), data.to_numpy(na_value=np.nan)
        if len(numdata.columns)>0:
            colindexes = np.flatnonzero(mask)
            noise[:, colindexes], npdata[:, colindexes] = numnoise, npdiscdata
        print("Noise", noise[:2, :])
        return npdata, noise

    def imputeMissings(self, data):
        #knn_imputer = impute.KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
        #return pd.DataFrame(knn_imputer.fit_transform(data), columns=data.columns)
        return None

class MappingTests(unittest.TestCase):

    def test_mapping_dense_data(self):
        for filename in ['gyeast.arff','original.arff','thoraric_surgery.arff']:
            dataset = Dataset(data='data/'+filename) #classindex=0
            #dataset.data.iloc[0, 3] = np.nan
            print(str(dataset))
            mapper = Mapper(strategy='flexible')
            mapper.preprocess(dataset.data)
            print("Data after mapping (first 3 observations):\n", mapper.itemdata[:3, :], "\n", mapper.noise[:3, :])

    def test_mapping_sparse_data(self):
        data_net = pd.read_csv('data/DRYGIN_sgadata.txt', delimiter='\t', header=None)
        values, rows, cols = [], [], []
        nodes, inodes, i = set(data_net[0]).union(set(data_net[2])), {}, 0
        for node in nodes: inodes[node], i = i, i + 1
        for __, row in data_net.iterrows():
            rows.append(inodes[row[0]])
            cols.append(inodes[row[2]])
            values.append(row[4])
        sparse_matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(nodes), len(nodes)))
        dataset_net = Dataset(sparse_matrix)
        mapper = Mapper(strategy='flexible')
        mapper.preprocess(dataset_net.data)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(unittest.TestLoader().loadTestsFromTestCase(MappingTests))
