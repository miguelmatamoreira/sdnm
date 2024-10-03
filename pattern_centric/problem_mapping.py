import numpy as np
import pandas as pd
import pandas as pd
from sqlalchemy import create_engine
from functools import reduce



def read_values(table_name, genes):
    '''
    Reads the genes from the table.

    Parameters:
    - table_name (str): Table to look at in the database.
    - genes (list of str): List of genes to retrieve from the table.

    Returns:
    - df_from_db (dataset): Part of the table with genes of interest.
    '''
    engine = create_engine('postgresql://ist195643:2800@db.tecnico.ulisboa.pt:5432/ist195643')
    select_query = f'SELECT * FROM {table_name} WHERE "Description" IN {tuple(genes)}'
    df_from_db = pd.read_sql(select_query, con=engine)
    engine.dispose()
    return df_from_db



def get_values(genes, tissues):
    '''
    Get the respective values in all samples for each gene and tissue.

    Parameters:
    - genes (list of str): List of genes of interest.
    - tissues (list of str): List of considered tissues.

    Returns:
    - values (dict): Dictionary of values for each specific gene and tissue, values[gene][tissue] = ([values], [samples]).
    - tissue_samples (dict): Samples for each tissue, tissue_samples[tissue] = list().
    - tissue_indiv (dict): Individuals for each tissue, tissue_indiv[tissue] = list().
    '''
    tissue_samples = {}
    tissue_indiv = {}
    values = {}
    for tissue in tissues:
        if tissue == "Amygdala":
            df = read_values("amygdala", genes)
        elif tissue == "Anterior cingulate cortex (BA24)":
            df = read_values("anterior", genes)
        elif tissue == "Caudate (basal ganglia)":
            df = read_values("caudate", genes)
        elif tissue == "Cerebellar Hemisphere":
            df = read_values("cerebellar", genes)
        elif tissue == "Cerebellum":
            df = read_values("cerebellum", genes)
        elif tissue == "Cortex":
            df = read_values("cortex", genes)
        elif tissue == "Frontal Cortex (BA9)":
            df = read_values("frontal_cortex", genes)
        elif tissue == "Hippocampus":
            df = read_values("hippocampus", genes)
        elif tissue == "Hypothalamus":
            df = read_values("hypothalamus", genes)
        elif tissue == "Nucleus accumbens (basal ganglia)":
            df = read_values("nucleus", genes)
        elif tissue == "Putamen (basal ganglia)":
            df = read_values("putamen", genes)
        elif tissue == "Spinal cord (cervical c-1)":
            df = read_values("spinal", genes)
        elif tissue == "Substantia Nigra":
            df = read_values("nigra", genes)
        elif tissue == "Pituitary":
            df = read_values("pituitary", genes)
        elif tissue == "Whole Blood":
            df = read_values("blood", genes)
        elif tissue == "Skin (Sun Exposed)":
            df = read_values("skin_sun", genes)
        elif tissue == "Adipose (Visceral)":
            df = read_values("adipose", genes)
        elif tissue == "Skin (Not Sun Exposed)":
            df = read_values("skin_not_sun", genes)
        # samples of each tissue
        tissue_samples[tissue] = df.columns.tolist()[2:]
        # individuals of each tissue
        tissue_indiv[tissue] = [indiv.split('-')[0] + '-' + indiv.split('-')[1] for indiv in tissue_samples[tissue]]
        df.iloc[:, 2:] = np.log1p(df.iloc[:, 2:])
        for gene in genes:
            values[gene] = values.get(gene, {})
            values[gene][tissue] = (df.loc[df[df["Description"] == gene].index.tolist()[0]].tolist()[2:], 
                list(df.columns[2:]))
    return values, tissue_samples, tissue_indiv



def create_matrix(values, genes, tissues_selected):
    '''
    Creates the matrix for pattern discovery (biclustering).

    Parameters:
    - values (dict): Dictionary of values for each specific gene and tissue, values[gene][tissue] = ([values], [samples]).
    - genes (list of str): Genes considered.
    - tissues_selected (list of tissues): Tissues considered.

    Returns:
    - matrix (numpy.ndarray): Matrix to be considered.
    '''
    # get the expressions
    matrix = np.zeros((len(genes), len(tissues_selected)))
    for i, gene in enumerate(genes):
        gene_vector = []
        for tissue in tissues_selected:
            mean_value = np.mean(values[gene][tissue][0])
            gene_vector.append(mean_value)
        matrix[i] = gene_vector
    # display the board
    # df_matrix = pd.DataFrame(matrix, index=genes, columns=tissues_selected)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    display(df_matrix.iloc[:, :5].head())
    return matrix



def get_individuals(tissue_indiv, tissues_selected):
    '''
    Retrieve the individuals that are present in the tissues selected.

    Parameters:
    - tissue_indiv (dict): Individuals for each tissue, tissue_indiv[tissue] = [...].
    - tissues_selected (list): List of tissues.

    Returns:
    - intersection (list): Individuals present in the tissues selected.
    '''
    if len(tissues_selected) == 1:
        # print("Select at least 2 tissues for triclustering")
        return []
    else:
        tissues_to_consider = {}
        for t in tissues_selected:
            tissues_to_consider[t] = tissue_indiv[t]
        intersection = list(reduce(set.intersection, (set(indiv_list) for indiv_list in tissues_to_consider.values())))
        return intersection
    


def create_cube(values, genes, tissues_selected, tissue_indiv):
    '''
    Creates the cube for pattern discovery (triclustering).

    Parameters:
    - values (dict): Dictionary of values for each specific gene and tissue, values[gene][tissue] = ([values], [samples]).
    - genes (list of str): Genes considered.
    - tissues (list of tissues): Tissues considered.
    - tissue_indiv (dict): Individuals for each tissue, tissue_indiv[tissue] = list().

    Returns:
    - cube (numpy.ndarray): Cube to be considered.
    - individuals (list of str): Individuals with information in the tissues selected.
    '''
    individuals = get_individuals(tissue_indiv, tissues_selected)
    cube = np.zeros((len(tissues_selected), len(individuals), len(genes)))
    # print(f"Cube shape: {cube.shape}")
    # get the values
    for i, tissue in enumerate(tissues_selected):
        for j, individual in enumerate(individuals):
            for k, gene in enumerate(genes):
                for index, element in enumerate(values[gene][tissue][1]):
                    if element.startswith(individual):
                        # print(f'Index: {index}\nValue: {values[gene][tissue][0][index]}')
                        cube[i, j, k] = values[gene][tissue][0][index]
    return cube, individuals



if __name__ == "__main__":
    print("Run app.py")