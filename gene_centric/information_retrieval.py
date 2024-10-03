import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine



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
    # print(df_from_db.head())
    # print(df_from_db.shape)
    # print(f"Read {table_name} successfully!")
    engine.dispose()
    return df_from_db



def read_genes(path):
    '''
    Reads the uniprot genes and removes missing/duplicate values.

    Parameters:
    - path (str): Path of the file to be read.

    Returns:
    - up (DataFrame): Contains for each gene the columns ["Gene Names", "Function [CC]"].
    '''
    # up = pd.read_excel(path)
    # up.drop(['Entry', 'Reviewed', 'Entry Name', 'Protein names', 'Organism', 'Length', 'Gene Ontology (cellular component)', 'Protein families', 'Gene Ontology (molecular function)', 'Gene Ontology (biological process)'], axis='columns', inplace=True)
    up = pd.read_json(path)
    up.drop(["Entry"], axis='columns', inplace=True)
    # removes every row that as at least one NA
    up = up.dropna()
    # removes duplicated genes, not interesting for us
    up = up.drop_duplicates(subset=['Gene Names'])
    return up



def sentences_preprocessing(up):
    '''
    Preprocesses the sentences with lowercase, stop words and lemmatization.

    Parameters:
    - up (DataFrame): Contains for each gene the columns ["Gene Names", "Function [CC]"].

    Returns:
    - sentences (list of str): List of sentences preprocessed.
    '''
    phrases = up["Function [CC]"]
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    sentences = []
    for phrase in phrases:
        words = word_tokenize(phrase)
        # lower and remove stop words
        normalized_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
        # lemmatization
        singular_words = [lemmatizer.lemmatize(word, pos='n') for word in normalized_words]
        sentences.append(" ".join(singular_words))
    return sentences



def tfidf(sentences, up):
    '''
    Calculates the TF-IDF in the sentences.
    The score for each gene is given by "receptor" + "brain" + "neuro".

    Parameters:
    - sentences (list of str): List of sentences preprocessed.
    - up (DataFrame): Contains for each gene the columns ["Gene Names", "Function [CC]"].

    Returns:
    - scores (DataFrame): Contains for each gene the columns ["gene", "score"].
    '''
    # tokenize the sentences
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    # back to text for tf-idf analysis
    text_sentences = [" ".join(sentence) for sentence in tokenized_sentences]
    # create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_sentences)
    # get the feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    # create a df to display the matrix
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    # get the only words important
    scores = tfidf_df[["receptor", "brain", "neuro", "gaba", "dopamine"]].copy()
    # add the gene name at the new dataset
    up_reset = up.reset_index(drop=True)
    scores['gene'] = up_reset['Gene Names']
    # reorder
    scores = scores[['gene'] + [col for col in scores.columns if col != 'gene']]
    # create the score variable
    scores["score"] = scores["receptor"] + scores["brain"] + scores["neuro"] + scores["gaba"] + scores["dopamine"]
    # drop the three words columns
    scores.drop(['receptor', 'brain', 'neuro'], axis='columns', inplace=True)
    return scores



def filter_scores(scores, threshold):
    '''
    Filter the scores provided by the model given a certain threshold.

    Parameters:
    - scores (DataFrame): Contains for each gene the columns ["gene", "score"].
    - threshold (float): Threshold to choose from.

    Returns:
    - genes_filtered (list of str): List of genes selected with score above the threshold.
    '''
    filtered = scores.loc[scores["score"] >= threshold]
    genes_filtered = []
    for gene in filtered["gene"]:
        genes_filtered.extend([word.upper() for word in gene.split()])
    return genes_filtered




def all_genes_file():
    '''
    Creates a file with all the genes available in the GTEx.
    '''
    df = pd.read_csv("../data/ir/gene_tpm_brain_Amygdala.gct", sep='\t', skiprows=2, index_col=0)
    # print(df.shape)
    gene_names = df['Description'].tolist()
    with open("../data/ir/all_genes.txt", 'w') as f:
        for gene in gene_names:
            f.write(gene + '\n')




def create_genes_file(genes_filtered):
    '''
    Creates a text file with a single gene at each line (existing).

    Parameters:
    - genes_filtered (list of str): List of genes selected.
    '''
    with open("../data/ir/all_genes.txt", "r") as f:
        genes_read = [line.strip() for line in f.readlines()]
        all_genes = sorted(genes_read)
    # df = pd.read_csv("../data/ir/gene_tpm_brain_Amygdala.gct", sep='\t', skiprows=2, index_col=0)
    # see which genes filtered are existant in our dataframes
    genes_existing = []
    for gene in genes_filtered:
        if gene in all_genes and gene not in genes_existing:
            genes_existing.append(gene)
    # save those genes in a text file
    # print(len(genes_existing))
    with open("../data/ir/filtered_genes_hs.txt", "w") as file:
        for gene in genes_existing:
            file.write(gene + "\n")



def run():
    '''
    Run the information retrieval step and store the genes retrieved.
    '''
    # up = read_genes("../data/ir/up_mm.xlsx")
    up = pd.read_csv("../data/ir/up_hs.csv")
    up = up[~up["Gene Names"].str.startswith("OR")]
    sentences = sentences_preprocessing(up)
    scores = tfidf(sentences, up)
    genes_filtered = filter_scores(scores, 0.3)
    # print(len(genes_filtered))
    create_genes_file(genes_filtered)



if __name__ == "__main__":
    print("Run the app.py")