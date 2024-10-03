import numpy as np, pandas as pd
import unittest, csv
from scipy import sparse, io, stats
import sys


class Bicluster:

    # ========== PART I: Store bicluster =========== #

    def __init__(self, rows=None, cols=None, btype='constant', key=None, rowdir=True, pattern=None, metrics=None,
                 target=None, quality=1):
        self.key = key
        self.rows = set() if rows is None else rows
        self.cols = set() if cols is None else cols
        self.ncols = len(cols)
        self.nrows = len(rows)
        self.btype = btype
        self.rowdir = rowdir
        self.pattern = pattern
        self.pattern_str = None
        self.metrics = {} if metrics is None else metrics
        self.target = target
        self.quality = quality

    # ========== PART II: Major methods =========== #

    def relevance(self, criteria):
        if criteria == 'area': return self.ncols*self.nrows
        elif criteria == 'lift': return self.metrics['lift']
        elif criteria == 'pvalue': return -self.metrics['pvalue']
        elif criteria == 'rows': return self.nrows
        elif criteria == 'columns': return self.ncols
        else: print('order criteria for closing step is unknown')

    def merge(self, bic, maxoverlap):
        mincols = maxoverlap * max(self.ncols, bic.ncols)
        setcols = set(self.cols).copy()
        overlapcols = setcols.intersection(bic.cols)
        if len(overlapcols) < mincols: return None
        setrows = set(self.rows).copy()
        overlaprows = setrows.intersection(bic.rows)
        nelements = maxoverlap * min(self.nrows * self.ncols, bic.nrows * bic.ncols)
        if len(overlapcols) * len(overlaprows) < nelements: return None
        return Bicluster(list(setrows.union(bic.rows)), list(setcols.union(bic.cols)),
                         btype=self.btype, rowdir=self.rowdir)

    def compute_pattern(self, itemdata):
        if self.pattern is not None: return
        bicdata = itemdata[self.rows][:,self.cols]
        self.pattern = stats.mode(bicdata, keepdims=True)[0][0]
        

    def compute_string_pattern(self, data, real=False):
        self.pattern_str = []
        bicdata = data.iloc[self.rows,self.cols]
        for col in bicdata.columns:
            if bicdata[col].dtype.name == 'category':
                self.pattern_str.append(bicdata[col].mode().iat[0])
            else:
                self.pattern_str.append(bicdata[col].median().iat[0])

    def compute_metrics(self, classvalues, classsup, nrows):
        self.metrics, counts = {'lifts': [], 'lift': min}, {}
        for i in range(len(classsup)): counts[i] = 0
        for row in self.rows:  counts[classvalues[row]] += 1
        for i in range(len(classsup)):
            supAB, supA, supB = counts[i] / nrows, len(self.rows) / nrows, classsup[i] / nrows
            lift = supAB / (supA * supB)  # lift(a=>b)=sup(ab)/(sup(a)sup(b))
            self.metrics['lifts'].append(lift)
            if lift > self.metrics['lift']: self.metrics['lift'] = lift

    # ========== PART III: Auxiliary methods =========== #

    def __str__(self, data=None, detail=False, disc=None):
        colnames = data.cols[self.cols] if data else str(self.cols)
        row_names = data.rows[self.rows] if data else str(self.rows)
        res = '(|X|,|Y|)=(%d, %d) Y=%s X=%s ' % (self.nrows, self.ncols, colnames, row_names)
        if self.pattern is not None: res = '%s type=%s pattern=%s' % (res, self.btype, str(self.pattern))
        if self.target is not None: res = '%s target=%d' % (res, self.target)
        if len(self.metrics) > 0: res = '%s metrics=%s' % (res, str(self.metrics))
        if detail and data is not None:
            if disc is None: res += '\n' + str(data.data.iloc[list(self.rows),list(self.cols)]) + '\n'
            else: res += '\n' + str(disc[self.rows,:][:,self.cols]) + '\n'
        return res if self.key is None else 'ID:' + str(self.key) + ' ' + res

    def overlap(self, bic):
        overlap = len(self.rows & bic.rows) * len(self.cols & bic.cols)
        return overlap / min(self.nrows * self.ncols, bic.nrows * bic.ncols)

    def transpose(self):
        cols, ncols = self.cols.copy(), self.ncols
        self.cols, self.ncols = self.rows, self.nrows
        self.rows, self.nrows = cols, ncols
        return self


class Biclusters:

    @staticmethod
    def order(bics, order='area'):
        if order == 'area':
            bics.sort(key=lambda x: x.ncols * x.nrows, reverse=True)
        elif order == 'lift':
            bics.sort(key=lambda x: x.metrics['lift'], reverse=True)
        elif order == 'pvalue':
            bics.sort(key=lambda x: x.metrics['pvalue'])
        else:
            bics.sort(key=lambda x: x.nrows, reverse=True)
        return bics

    @staticmethod
    def above(bics, metrics):
        result = []
        for bic in bics:
            for metric in metrics:
                if bic.metrics[metric] >= metrics[metric]: result.append(bic)
        return result

    @staticmethod
    def compute_patterns(bics, itemdata):
        for bic in bics: bic.compute_pattern(itemdata)

    @staticmethod
    def compute_string_patterns(bics, data):
        for bic in bics: bic.compute_string_pattern(data)

    @staticmethod
    def transpose(bics):
        for bic in bics: bic.transpose()

    @staticmethod
    def to_string(bics, data=None, detail=True, disc=None):
        res = '#bics = %d\n' % len(bics)
        for bic in bics: res += bic.__str__(data,False) + '\n'
        if data and detail:
            res += '\nDETAIL\n\n'
            for bic in bics: res += bic.__str__(data,True,disc) + '\n'
        return res

    @staticmethod
    def reset_keys(bics):
        key = 0
        for bic in bics:
            bic.key = key
            key += 1


class Dataset:

    def __init__(self, data=None, targets=None, classindex=None, transpose=False):

        # A: load data
        formats = 'pandas DataFrame, numpy ndarray, or scipy sparse.csr_matrix'
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, sparse.csr_matrix):
            self.data, self.sparse = data, True
        elif isinstance(data, list):
            self.data = pd.DataFrame(np.array(data))
        elif isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data)
        elif isinstance(data, str):
            print('warning: reading data from file\nsuggestion: input data as %s' % formats)
            if data.endswith('arff'):
                self.data = pd.DataFrame(io.arff.loadarff(data)[0])
                cols = self.data.select_dtypes(include='object').columns.tolist()
                for col in cols: self.data[col] = self.data[col].str.decode('utf-8')
            else:
                with open(data, "r") as f:
                    delimiter = csv.Sniffer().sniff(f.readline()).delimiter
                    self.data = pd.read_csv(data, sep=str(delimiter))
            if self.data.iloc[:, 0].nunique() == self.data.shape[0]:
                self.data.set_index(self.data.columns[0], inplace=True)
        else:
            print('error: data format not accepted. Input data as %s' % formats)
        if transpose: self.data = self.data.transpose()  # same for pandas and sparse data

        # B: extract properties
        self.targets = None if targets is None else pd.DataFrame({'class': targets})
        if isinstance(self.data, sparse.csr_matrix):
            self.nrows, self.ncols = self.data.get_shape()
            self.rows, self.cols = list(map(str, range(self.nrows))), list(map(str, range(self.ncols)))
        elif isinstance(self.data, pd.DataFrame):
            self.nrows, self.ncols = self.data.shape
            self.sparse, self.rows, self.cols = False, self.data.index, self.data.columns
            self.data = self.data.convert_dtypes()
            mask = self.data.dtypes == 'string'
            self.data[self.data.columns[mask]] = self.data.loc[:, mask].astype('category')

            if targets is None:
                if classindex is None:
                    for target in ['class', 'output', 'target']:
                        if target in self.cols:
                            classindex = self.data.columns.get_loc(target)
                            print(
                                'warning: output variables detected (replace "class", "output", "target" names for unsupervised searches)')
                            break
                if classindex is not None:
                    self.targets = self.data.iloc[:, classindex]
                    self.targetnames = self.targets.unique()
                    self.targets = self.targets.cat.codes.tolist()
                    self.data.drop(self.data.columns[classindex], axis=1, inplace=True)
                    self.ncols, self.cols = self.ncols - 1, self.data.columns

    def get_categorical_map(self, variable):
        if variable == 'class': return dict(zip(self.targets.cat.codes, self.targets))
        return dict(zip(self.data[variable].cat.codes, self.data[variable]))

    def statistics(self):
        if self.sparse:
            return "Network data (%d nodes and %d interactions)" % (self.nrows, self.data.getnnz())
        else:
            return "Multivariate data (|X|=%d |Y|=%d)" % (self.nrows, self.ncols)

    def __str__(self, real=True):
        classinfo = "" if self.targets is None else ("#classes=%d %s\n" % (self.ntargets, str(self.targets)))
        return "%s\n%s\n%s" % (self.statistics(), str(self.data), classinfo)


class DomainTests(unittest.TestCase):

    def test_biclusters(self):
        bic1 = Bicluster({1, 2, 4}, {2, 4, 5})
        print("Bic_1:", str(bic1), "\nBic_1_T:", str(bic1.transpose()))
        bic2 = Bicluster({1, 2, 3}, {3, 4, 5})
        bic3 = bic1.transpose().merge(bic2, 0.3)
        print("Bic_1U2|0.6:", str(bic1.merge(bic2, 0.6)), "\nBic_1U2|0.3:", str(bic3))
        bics = [bic1, bic3, bic2]
        print("\nUnordered:", str(Biclusters.to_string(bics)), "\nOrdered:",
              str(Biclusters.to_string(Biclusters.order(bics))))

    def test_data_matrix(self):
        data_simple = ['example_constant.arff', 'example_op.arff', 'gyeast.arff', 'gyeastcompact.txt']
        for dataname in data_simple: print(str(Dataset(data='data/'+dataname)))

        data_class = [('original.arff', None), ('eeg_eye_state.arff', -1), ('fertility_Diagnosis.txt', -1),
                      ('BreastTissue.csv', 1)]
        for dataname, classindex in data_class: print(str(Dataset(data='data/'+dataname,classindex=classindex)))

        data_mix = [('joana4.arff', -2), ('thoraric_surgery.arff', -1), ('post-operative.data', -1), ('andre.csv', -1),
                    ('CTG.csv', -2)]
        for dataname, classindex in data_mix: print(str(Dataset(data='data/'+dataname,classindex=classindex)))

        data_highdim = ['ColonDiff62x2000.arff', 'Embryo60x7129.arff']  # lymphomas
        for dataname in data_highdim: print(str(Dataset(data='data/'+dataname)))

        datasets_gen = ['dataConstantOverallS1000normalN0M0.txt', 'dataAdditiveS1000normalN0M0.txt']
        for dataname in datasets_gen: print(str(Dataset(data='data/'+dataname)))

    def test_data_network(self):
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
        print(data_net, "\n", str(dataset_net))


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(unittest.TestLoader().loadTestsFromTestCase(DomainTests))