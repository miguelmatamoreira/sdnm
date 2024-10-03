import numpy as np
import pandas as pd
import concurrent.futures
from .read_tables import read_values



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

    def read_tissue_values(tissue):
        # determine table name based on tissue
        if tissue == "Amygdala":
            table_name = "amygdala"
        elif tissue == "Anterior cingulate cortex (BA24)":
            table_name = "anterior"
        elif tissue == "Caudate (basal ganglia)":
            table_name = "caudate"
        elif tissue == "Cerebellar Hemisphere":
            table_name = "cerebellar"
        elif tissue == "Cerebellum":
            table_name = "cerebellum"
        elif tissue == "Cortex":
            table_name = "cortex"
        elif tissue == "Frontal Cortex (BA9)":
            table_name = "frontal_cortex"
        elif tissue == "Hippocampus":
            table_name = "hippocampus"
        elif tissue == "Hypothalamus":
            table_name = "hypothalamus"
        elif tissue == "Nucleus accumbens (basal ganglia)":
            table_name = "nucleus"
        elif tissue == "Putamen (basal ganglia)":
            table_name = "putamen"
        elif tissue == "Spinal cord (cervical c-1)":
            table_name = "spinal"
        elif tissue == "Substantia Nigra":
            table_name = "nigra"
        elif tissue == "Pituitary":
            table_name = "pituitary"
        elif tissue == "Whole Blood":
            table_name = "blood"
        elif tissue == "Skin (Sun Exposed)":
            table_name = "skin_sun"
        elif tissue == "Adipose (Visceral)":
            table_name = "adipose"
        elif tissue == "Skin (Not Sun Exposed)":
            table_name = "skin_not_sun"
        # read values from the database
        df = read_values(table_name, genes)
        # log-transform the values
        df.iloc[:, 2:] = np.log1p(df.iloc[:, 2:])
        # collect samples and individuals for the tissue
        samples = df.columns.tolist()[2:]
        individuals = [indiv.split('-')[0] + '-' + indiv.split('-')[1] for indiv in samples]
        # collect gene values
        gene_values = {}
        for gene in genes:
            gene_values[gene] = (df.loc[df[df["Description"] == gene].index.tolist()[0]].tolist()[2:], samples)
        return tissue, samples, individuals, gene_values
    # use ThreadPoolExecutor to read tissue values in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(read_tissue_values, tissue): tissue for tissue in tissues}
        for future in concurrent.futures.as_completed(futures):
            tissue = futures[future]
            try:
                tissue, samples, individuals, gene_values = future.result()
                if samples and individuals:
                    tissue_samples[tissue] = samples
                    tissue_indiv[tissue] = individuals
                    for gene in gene_values:
                        if gene not in values:
                            values[gene] = {}
                        values[gene][tissue] = gene_values[gene]
            except Exception as e:
                # print(f"Error processing tissue {tissue}: {e}")
                raise KeyboardInterrupt
    return values, tissue_samples, tissue_indiv
    


def get_most_expressive_genes_tissue(values, tissue):
    '''
    Get the most expressive genes the tissue provided.

    Parameters:
    - values (dict): Dictionary of values for each specific gene and tissue, values[gene][tissue].
    - tissue (str): Tissue to be considered.

    Returns:
    - most_expressive_genes (list of str): List of the most expressive genes.
    '''
    most_expressive_genes = []
    for gene in values:
        most_expressive_genes.append((gene, np.mean(values[gene][tissue][0])))
    most_expressive_genes = sorted(most_expressive_genes, key=lambda x: x[1], reverse=True)
    return most_expressive_genes



def construct_samples_dataset():
    '''
    Construct the samples dataset for each unique individual profile and saves as csv file.
    '''
    sp = pd.read_csv("data/samples/subject_phenotypes.txt", delimiter="\t")
    sa = pd.read_csv("data/samples/sample_attributes.txt", delimiter="\t")
    # add the subject id in the sa dataset
    sa["SUBJID"] = sa['SAMPID'].str.split('-', n=2).str[:2].str.join('-')
    column_order = ["SUBJID"] + [col for col in sa.columns if col != "SUBJID"]
    sa = sa[column_order]
    # only concerned with the samples that we want
    sa_brain = sa.loc[(sa["SMTS"] == "Brain") | (sa["SMTS"] == "Pituitary") | (sa["SMTS"] == "Blood") | (sa["SMTS"] == "Adipose Tissue") | (sa["SMTS"] == "Skin")]
    # remove disposable columns
    sa_brain = sa_brain[["SUBJID", "SAMPID", "SMTSD"]]
    # add the individual profiles sex, age, agonal state
    samples = pd.merge(sa_brain, sp, on='SUBJID', how='inner')
    samples.rename(columns={'SUBJID': 'subject', 'SAMPID': 'sample', 'SMTSD': 'tissue', 'SEX': 'sex', 'AGE': 'age',
        'DTHHRDY': 'agonal_state'}, inplace=True)
    samples.to_csv("data/samples/samples.csv", index=False)



def get_samples_dataset():
    '''
    Gets the samples dataset.
    
    Returns:
    - samples (dataset): Dataset with columns (subject, sample, tissue, sex, age, agonal_state)
    '''
    samples = pd.read_csv("data/samples/samples.csv")
    return samples



if __name__ == "__main__":
    print("Run the app.py")