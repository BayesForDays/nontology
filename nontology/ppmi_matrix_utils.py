import scipy as sp
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from scipy.sparse import hstack


def vectorize(df, colname, min_df=1):
    """
    Take an arbitrary text column in a dataframe and turn it into sparse vectors
    :param df: A dataframe
    :param colname: The text column you want to transform
    :param min_df: How many rows that feature must be in to be added as a dimension
    :return: Sparse vectors for each row and the object that created those features
    """
    # transform into useful features
    df[colname] = df[colname].astype(unicode)
    vectorizer = CountVectorizer(min_df=min_df)
    vectorized_ = vectorizer.fit_transform(df[colname])
    return vectorizer, vectorized_


def generate_co_occurrence_matrix(sparse_tokens):
    """
    Take sparse features and calculate the number of times you see each feature
    occur with every other feature.
    :param sparse_tokens: Typically the output of vectorize, a sparse matrix of words or "words"
    :return: The co-occurrence matrix in log space
    """
    # transpose sparse matrices
    sparse_tokens_t = sparse_tokens.T
    # compute normalized log co-occurance counts
    co_occur = sparse_tokens_t.dot(sparse_tokens)
    dense_co_occur = co_occur.todense()
    co_occur_log_matrix = sp.special.xlogy(sp.sign(dense_co_occur),
                                           dense_co_occur) - np.log(co_occur.sum())
    np.fill_diagonal(co_occur_log_matrix, 0)
    return co_occur_log_matrix


def generate_marginal_matrix(sparse_tokens):
    """
    Compute outer product of normalized marginal token counts
    :param sparse_tokens: Typically the output of vectorize, a sparse matrix of words or "words"
    :return: The marginal matrix used to calculate PMI, in log space
    """
    marginal = sparse_tokens.sum(axis=0).astype(np.uint32)
    marginal = marginal / marginal.astype(np.float32).sum()
    marginal_log_matrix = np.log(np.outer(marginal, marginal))
    return marginal_log_matrix


def generate_pmi_matrix(co_occur_log_matrix, marginal_log_matrix, k=1.0):
    """
    Calculate PMI - p(AB)/(p(A)*p(B))
    I do not recommend doing this on the raw matrices if you used lots of one-hot encoded variables
    :param co_occur_log_matrix: A co-occurrence matrix of size nxn
    :param marginal_log_matrix: A marginal matrix of size nxn
    :param k: How much smoothing you want to do
    :return: A PMI matrix of nxn
    """
    if k < 0:
        k = 0.0
    pmi_matrix = co_occur_log_matrix - marginal_log_matrix - np.log(k)
    pmi_matrix[pmi_matrix < 0] = 0  # positive PMI: don't include negative associations
    np.fill_diagonal(pmi_matrix, 0)
    return pmi_matrix


def generate_matrices(sparse_tokens):
    """
    Convenience function for keeping matrices out of memory
    :param sparse_tokens:
    :return: Both co-occurrence and marginal matrices
    """
    co_occur_log_matrix = generate_co_occurrence_matrix(sparse_tokens)
    marginal_log_matrix = generate_marginal_matrix(sparse_tokens)
    return co_occur_log_matrix, marginal_log_matrix


def generate_vectors(pmi_matrix, n_components=100, normalize_flag=True):
    """
    Takes PMI matrix, does SVD, creates lower-dimensional representation
    :param pmi_matrix:
    :param n_components: Number of dimensions you want back from your data
    :param normalize_flag: If yes, vector lengths sum to 1
    :return:
    """
    pca = PCA(whiten=True, n_components=n_components)
    vecs = pca.fit_transform(pmi_matrix)
    # normalize vecs
    if normalize_flag:
        return vecs / np.c_[np.sqrt((vecs ** 2).sum(axis=1))]
    else:
        return vecs


def create_vector_df(vectors, colname, vocabulary):
    """
    Takes vectors and feature names and makes a dataframe
    :param vectors:
    :param colname:
    :param vocabulary:
    :return:
    """
    vocabularies_sorted = sorted(list(vocabulary))
    vector_df = pd.DataFrame(vectors, index=vocabularies_sorted)
    vector_df[colname] = vocabularies_sorted
    return vector_df



def concatenate_sparse_matrices(list_of_sparse_matrices):
    """
    Squish sparse matrices together horizontally
    :param list_of_sparse_matrices: Lots of sparse matrices with the same number of rows
    :return: An even larger sparse matrix
    """
    concatenated = hstack(list_of_sparse_matrices)
    return concatenated


def create_x_and_y_vectors(x_and_y_vectors, x_col, y_col, vectorizer_vocabularies):
    """
    :param x_and_y_vectors: vectors to break up
    :param x_col: Feature columns to reference for x category
    :param y_col: Feature columns to reference for y category
    :param vectorizer_vocabularies: Both of the vocabularies you want to reference columns by
    :return: the subsets of the original vectors
    """
    vectorized_x_len = len(vectorizer_vocabularies[0])
    vectorized_y_len = len(vectorizer_vocabularies[1])
    total_len = vectorized_x_len + vectorized_y_len

    vocabularies_sorted = []
    for vocabulary_list in vectorizer_vocabularies:
        vocabularies_sorted += sorted(list(vocabulary_list))

    x_y_vectors = pd.DataFrame(x_and_y_vectors, index=vocabularies_sorted)
    x_vectors = x_y_vectors.iloc[0:vectorized_x_len]
    x_vectors[x_col] = vocabularies_sorted[0:vectorized_x_len]
    y_vectors = x_y_vectors.iloc[vectorized_x_len: total_len]
    y_vectors[y_col] = vocabularies_sorted[vectorized_x_len: total_len]
    return x_vectors, y_vectors


def compute_pmi_vecs(m1, m2, n_pca_components, normalize_flag=True):
    # concatenate
    m_concatenated = concatenate_sparse_matrices([m1, m2])
    # generate co-occurence matrix and outer product of normalized marginal token counts
    # and return in log space
    co_occ_matrix, marginal_log_matrix = generate_matrices(m_concatenated)
    # Calculate PMI - p(AB)/(p(A)*p(B))
    pmi_matrix = generate_pmi_matrix(co_occ_matrix, marginal_log_matrix)
    # PCA on PMI
    return generate_vectors(pmi_matrix,
                            n_components=n_pca_components,
                            normalize_flag=normalize_flag)
