import subprocess
import numpy as np
from scipy.stats import binom
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
tric = None



class Tricluster:
    def __init__(self, data, I, J, K):
        self.I, self.J, self.K = I, J, K
        subspace = data[np.ix_(I,J,K)]
        self.n, self.m, self.t = subspace.shape
        self.pattern = np.array([np.nanmean(subspace[:,j,:], axis=0) for j, var in enumerate(J)])
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
    n, m, k = data.shape
    for xic in xics:
        pvalue = 0
        p = model.pattern_prob(xic)
        for i in range(xic.n, n+1):
            pvalue += binom.pmf(i,n,p)
        pvalues.append(pvalue)
        # print("p_value = %E (p_pattern = %f)"%(pvalue,p))
    return pvalues



def calculate_significances(data, triclusters):
    '''
    Calculates the signifiances for each bicluster.

    Parameters:
    - data (numpy.ndarray): Data to be considered.
    - triclusters (list): List of triclusters.

    Returns:
    - p_values (list of int): Significances calculated.
    '''
    global tric
    p_values = []
    for tricluster in triclusters:
        tri_object = Tricluster(data, I=tricluster[0], J=tricluster[1], K=tricluster[2])
        tric = tri_object
        p_value = significance(data, [tri_object], coherence=0.3)
        p_values.append(p_value[0])
    return p_values



def get_triclusters_from_file(file_path):
    '''
    Retrieves the triclusters obtained by the triclustering algorithm.

    Parameters:
    - file_path (str): Path of the text file where the algorithm has written the output.

    Returns:
    - triclusters (list): List of triclusters.
    '''
    keyword = "|T|x|S|x|G|"
    triclusters = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if keyword in line:
            # get the dimensions
            dimensions = [int(num) for num in line[len(keyword)+2:].split('x')]
            
            # get the tissues
            tissues = [int(lines[i+1].strip().split(":")[1])]
            # get the other tissues
            detour = dimensions[2] + 3
            for d in range(1, dimensions[0]):
                tissue_n = int(lines[i+1+d*detour].strip().split(":")[1]) 
                tissues.append(tissue_n)
            # print(tissues)

            # get the samples
            samples = [val for val in lines[i+2].strip().split() if val]
            stripped_samples = [int(item.replace('S-', '')) for item in samples]
            # print(samples)

            # get the genes
            genes = []
            for j in range(1, dimensions[2]+1):
                gene = [val for val in lines[(i+2)+j].strip().split() if val][0]
                genes.append(gene)
            stripped_genes = [int(item.replace('G-', '')) for item in genes]
            # print(genes)

            triclusters.append([tissues, stripped_samples, stripped_genes])
    return triclusters



def create_tab_file_from_cube(cube):
    '''
    Creates tab file to be later processed by triclustering algorithm.

    Parameters:
    - cube (numpy.ndarray): Cube to be considered.
    '''
    with open("data/cube.tab", "w") as f:
        # write total times, samples, and genes
        f.write("Total Tissues:\t{}\n".format(cube.shape[0]))
        f.write("Total Samples:\t{}\n".format(cube.shape[1]))
        f.write("Total Genes:\t{}\n".format(cube.shape[2]))
        # por cada linha
        for tissue_idx in range(cube.shape[0]):
            f.write("Tissue\t{}\n".format(tissue_idx))
            f.write("{}\n".format("\t".join(["ID", "NAME"] + ["S-{}".format(i) for i in range(cube.shape[1])])))
            for gene_idx in range(cube.shape[2]):
                for sample_idx in range(cube.shape[1]):
                    values = cube[tissue_idx, :, gene_idx]
                    value_str = "\t".join(str(value) for value in values)
                    f.write("{}\tG-{}\t{}\n".format(gene_idx, gene_idx, value_str))
                    break



def tricluster(cube, minsize_tissues, minsize_indiv, minsize_genes):
    '''
    Performs the TriCluster algorithm on the data selected.

    Parameters:
    - cube (numpy.ndarray): Cube to be considered.
    - minsize_tissues (int): Minimum size for clustering tissues.
    - minsize_indiv (int): Minimum size for clustering individuals.
    - minsize_genes (int): Minimum size for clustering genes.

    Returns:
    - triclusters (list): List of triclusters obtained.
    '''
    # create tab file for the algorithm
    create_tab_file_from_cube(cube)
    # run the algorithm
    subprocess.run('make', cwd='pattern_centric/tricluster', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # for mac
    subprocess.run(f'./triCluster -f../../data/cube.tab -s"[{minsize_tissues},{minsize_indiv},{minsize_genes}]" -w0.05 -o1 -r"../../data/output_tricluster.txt"', cwd='pattern_centric/tricluster', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # for ubuntu
    # subprocess.run(f'./triCluster -f{os.path.abspath("../../data/cube.tab")} -s"[{minsize_tissues},{minsize_indiv},{minsize_genes}]" -w0.05 -o1 -r"{os.path.abspath("../../data/output_tricluster.txt")}"', cwd='pattern_centric/tricluster', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run('make clean', cwd='pattern_centric/tricluster', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # retrieve the triclusters
    triclusters = get_triclusters_from_file("data/output_tricluster.txt")
    return triclusters



if __name__ == "__main__":
    print("Run the app.py")