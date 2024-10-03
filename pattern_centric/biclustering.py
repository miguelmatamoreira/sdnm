from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import SpectralBiclustering
import numpy as np
from pattern_centric.bicpams.domain import Dataset, Biclusters
from pattern_centric.bicpams.mining import BicPAMS, Miner
from pattern_centric.bicpams.mapping import Mapper
from pattern_centric.bicpams.closing import Closer
from scipy.stats import binom
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



class Bicluster:
    def __init__(self, data, I, J):
        self.I, self.J = I, J
        subspace = data[np.ix_(I,J)]
        self.subspace = subspace
        self.n, self.m = subspace.shape
        self.pattern = np.nanmean(subspace, axis=0)
class NullModel:
    def __init__(self, data, coherence, uniform):
        self.is2D = (len(data.shape)==2)
        self.coherence = coherence
        self.uniform = uniform
        if not(uniform): 
            self.empirical, self.mins, self.maxs = [], [], []
            for j in range(data.shape[1]):
                values = data[:,j] if self.is2D else np.reshape(data[:,j,:],-1)
                self.mins.append(np.nanmin(values))
                self.maxs.append(np.nanmax(values))
                self.empirical.append(ECDF(values))
    def pattern_prob(self, xic):
        if self.uniform: 
            return self.coherence**(xic.m) if self.is2D else self.coherence**(tric.m*tric.t)
        p = 1
        for j, var in enumerate(xic.J):
            delta = (self.maxs[var]-self.mins[var])*self.coherence/2
            values = [xic.pattern[j]] if self.is2D else [tric.pattern[j,k] for k in range(tric.t)]
            for value in values:
                lower, upper = max(value-delta,self.mins[var]), min(value+delta,self.maxs[var])
                p *= (self.empirical[var](upper)-self.empirical[var](lower))
        return p
def significance(data, xics, coherence, uniform=False):
    # print(f"xics:{xics}")
    model = NullModel(data, coherence, uniform)
    pvalues = []
    n, m = data.shape
    for xic in xics:
        pvalue = 0
        p = model.pattern_prob(xic)
        for i in range(xic.n, n+1):
            pvalue += binom.pmf(i,n,p)
        pvalues.append(pvalue)
        # print("p_value = %E (p_pattern = %f)"%(pvalue,p))
    return pvalues



def calculate_significances(data, row_indices, column_indices, biclusters):
    '''
    Calculates the signifiances for each bicluster.

    Parameters:
    - data (numpy.ndarray): Data to be considered.
    - row_indices (numpy.ndarray): Row indices for each bicluster.
    - column_indices (numpy.ndarray): Column indices for each bicluster.
    - biclusters (numpy.ndarray): Biclusters found.

    Returns:
    - new_p_values (list of int): Significances calculated.
    '''
    p_values = []
    for ri, ci, bi in zip(row_indices, column_indices, biclusters):
        bi_object = Bicluster(data, I=ri.tolist(), J=ci.tolist())
        p_value = significance(data, [bi_object], coherence=0.3)
        p_values.append(p_value[0])
    # replace for 1 if very near
    threshold = 1e-10
    new_p_values = [1 if abs(x - 1) < threshold else x for x in p_values]
    # print(new_p_values)
    # print(p_values)
    return new_p_values



def spectral_coclustering(X, n_clusters_):
    '''
    Applies the sklearn spectral coclustering algorithm.

    Parameters:
    - X (numpy.ndarray): Data to be considered.
    - n_clusters_ (int): Maximum number of clusters.

    Returns:
    - row_indices (numpy.ndarray): Row indices for each bicluster.
    - column_indices (numpy.ndarray): Column indices for each bicluster.
    - biclusters (numpy.ndarray): Biclusters found.
    '''
    # handle near-zero values to avoid numerical issues
    epsilon = 1e-10
    near_zero_mask = np.isclose(X, 0, atol=epsilon)
    X[near_zero_mask] = epsilon
    # print(f"Data:\n{X}")
    clustering = SpectralCoclustering(n_clusters=n_clusters_, random_state=0).fit(X)
    # print(f"Row labels: {clustering.row_labels_}")
    # print(f"Column labels:{clustering.column_labels_}")
    row_labels = clustering.row_labels_
    column_labels = clustering.column_labels_
    # know the indices
    row_indices = []
    column_indices = []
    biclusters = []
    p_values = []
    for cluster in range(n_clusters_):
        row_indices.append(np.where(row_labels == cluster)[0])
        column_indices.append(np.where(column_labels == cluster)[0])
        bicluster = X[row_indices[-1]][:, column_indices[-1]]
        biclusters.append(bicluster)
        # print(f"Bicluster {cluster}:\n{X[row_indices[-1]][:, column_indices[-1]]}")
    # print(row_indices)
    # print(column_indices)
    return row_indices, column_indices, biclusters



def spectral_biclustering(X, n_clusters_):
    '''
    Applies the sklearn spectral biclustering algorithm.

    Parameters:
    - X (numpy.ndarray): Data to be considered.
    - n_clusters_ (int): Maximum number of clusters.

    Returns:
    - row_indices (numpy.ndarray): Row indices for each bicluster.
    - column_indices (numpy.ndarray): Column indices for each bicluster.
    - biclusters (numpy.ndarray): Biclusters found.
    '''
    # handle near-zero values to avoid numerical issues
    epsilon = 1e-10
    near_zero_mask = np.isclose(X, 0, atol=epsilon)
    X[near_zero_mask] = epsilon
    # print(f"Data:\n{X}")
    clustering = SpectralBiclustering(n_clusters=n_clusters_, random_state=0).fit(X)
    # print(f"Row labels: {clustering.row_labels_}")
    # print(f"Column labels:{clustering.column_labels_}")
    row_labels = clustering.row_labels_
    column_labels = clustering.column_labels_
    # know the indices
    row_indices = []
    column_indices = []
    biclusters = []
    for cluster in range(n_clusters_):
        row_indices.append(np.where(row_labels == cluster)[0])
        column_indices.append(np.where(column_labels == cluster)[0])
        bicluster = X[row_indices[-1]][:, column_indices[-1]]
        biclusters.append(bicluster)
        # print(f"Bicluster {cluster}:\n{X[row_indices[-1]][:, column_indices[-1]]}")
    # print(row_indices)
    # print(column_indices)
    return row_indices, column_indices, biclusters



def bicpams(df_X, pattern_type, mapper_strategy, mincols_):
    '''
    Applies the bicpams algorithm.

    Parameters:
    - df_X (numpy.ndarray): Data to be considered in dataset.
    - pattern_type (str): Pattern chosen to find.
    - mapper_strategy: Mapper strategy selected to apply.
    - mincols_ (int): Minimum number of columns.

    Returns:
    - row_indices (numpy.ndarray): Row indices for each bicluster.
    - column_indices (numpy.ndarray): Column indices for each bicluster.
    - biclusters (numpy.ndarray): Biclusters found.
    - significances (list of float): Significances for each bicluster found.
    '''
    # choose the pattern type
    if pattern_type == "Constant":
        pattern_ = "constant"
    else:
        pattern_ = "orderpreserving"
    # choose the mapper strategy
    if mapper_strategy == "Column Normalization":
        mapper_strategy_ = "flexible"
    elif mapper_strategy == "Row Normalization":
        mapper_strategy_ = "normrows"
    elif mapper_strategy == "Overall Normalization":
        mapper_strategy_ = "normall"
    elif mapper_strategy == "Width":
        mapper_strategy_ = "width"
    elif mapper_strategy == "Rows Frequency":
        mapper_strategy_ = "rowsfrequency"
    else:
        mapper_strategy_ = "frequency"

    # algorithm
    data = Dataset(data=df_X, transpose=False)
    if pattern_type == "Constant":
        mapper = Mapper(strategy=mapper_strategy_, bins = 5, noise=True, removals=[]) 
    else:
        mapper = Mapper(strategy=mapper_strategy_, bins = 20, noise=True, removals=[]) 
    miner = Miner(pattern=pattern_, stopConditions={'mincols': mincols_, 'minbics': 5, 'minsig':1}, niterations=2)
    closer = Closer(mergeOverlap=0.7, filterOverlap=0.5, order='area')
    bics = BicPAMS.run(data, mapper, miner, closer)
    # save the data
    # open('data/output_bicpams.txt', 'w').write(Biclusters.to_string(bics, data, detail=False, disc=mapper.discata))
    # prepare data to send to app
    row_indices = []
    column_indices = []
    biclusters = []
    significances = []
    for bic in bics:
        row_names = bic.rows
        col_names = bic.cols
        bicluster = data.data.iloc[list(bic.rows),list(bic.cols)]
        # get cols repeated
        unique_col_names = []
        repeated_indices = []
        for i, col in enumerate(col_names):
            if col not in unique_col_names:
                unique_col_names.append(col)
            else:
                repeated_indices.append(i)
        # rearrange the bicluster
        bicluster_np = bicluster.values
        new_bicluster = []
        for row in bicluster_np:
            new_row = np.delete(row, repeated_indices)
            new_bicluster.append(new_row)
        # convert to np
        row_names_np = np.array(row_names)
        unique_cols_np = np.array(unique_col_names)
        new_bicluster_np = np.array(new_bicluster)  
        # add to list 
        row_indices.append(row_names_np)
        column_indices.append(unique_cols_np)
        biclusters.append(new_bicluster_np)
        significances.append(bic.metrics["significance"])
    return row_indices, column_indices, biclusters, significances



if __name__ == "__main__":
    print("Run the app.py")