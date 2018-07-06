import scipy as sp
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from scipy.sparse import hstack


def generate_co_occurrence_matrix(sparse_tokens):
    """

    :param sparse_tokens:
    :return:
    """
    return construct_co_occurence_matrix(sparse_tokens)


def construct_co_occurence_matrix(sparse_tokens):
    """

    :param sparse_tokens:
    :return:
    """
    # transpose sparse matrices
    sparse_tokens_t = sparse_tokens.T
    # compute normalized log co-occurance counts
    co_occur = sparse_tokens_t.dot(sparse_tokens)
    dense_co_occur = co_occur.todense()
    co_occur_log_matrix = sp.special.xlogy(
        sp.sign(dense_co_occur), dense_co_occur
    ) - np.log(co_occur.sum())
    np.fill_diagonal(co_occur_log_matrix, 0)
    return co_occur_log_matrix


def construct_marginal_matrix(sparse_tokens):
    """
    Compute outer product of normalized marginal token counts
    :param sparse_tokens: Typically the output of vectorize, a sparse matrix of words or "words"
    :return: The marginal matrix used to calculate PMI, in log space
    """
    marginal = sparse_tokens.sum(axis=0).astype(np.uint32)
    marginal = marginal / marginal.astype(np.float32).sum()
    marginal_log_matrix = np.log(np.outer(marginal, marginal))
    return marginal_log_matrix


def generate_marginal_matrix(sparse_tokens):
    return construct_marginal_matrix(sparse_tokens)


def construct_pmi_matrix(
        co_occur_log_matrix, marginal_log_matrix,
        k=1.0, positive_only_flag=True
):
    """

    :param co_occur_log_matrix:
    :param marginal_log_matrix:
    :param k:
    :param positive_only_flag:
    :return:
    """
    if k < 0:
        k = 0.0
    pmi_matrix = co_occur_log_matrix - marginal_log_matrix - np.log(k)
    if positive_only_flag:
        pmi_matrix[pmi_matrix < 0] = 0  # positive PMI: don't include negative associations
        np.fill_diagonal(pmi_matrix, 0)
    else:
        np.fill_diagonal(pmi_matrix, 0)
    return pmi_matrix


def generate_pmi_matrix(
        co_occur_log_matrix, marginal_log_matrix,
        k=1.0, positive_only_flag=True
):
    return construct_pmi_matrix(
        co_occur_log_matrix, marginal_log_matrix,
        k, positive_only_flag
)


def generate_matrices(sparse_tokens):
    """

    :param sparse_tokens:
    :return:
    """
    return construct_matrices(sparse_tokens)


def construct_matrices(sparse_tokens):
    """

    :param sparse_tokens:
    :return:
    """
    co_occur_log_matrix = generate_co_occurrence_matrix(sparse_tokens)
    marginal_log_matrix = generate_marginal_matrix(sparse_tokens)
    return co_occur_log_matrix, marginal_log_matrix

def generate_vectors(m, n_components=100, normalize_flag=True):
    """

    :param m:
    :param n_components:
    :param normalize_flag:
    :return:
    """
    return compute_vectors(
        m, n_components, normalize_flag
    )

def compute_vectors(m, n_components=100, normalize_flag=True):
    """

    :param pmi_matrix:
    :param n_components:
    :param normalize_flag:
    :return:
    """
    pca = PCA(
        whiten=True, n_components=n_components
    )
    vecs = pca.fit_transform(m)
    # normalize vecs
    if normalize_flag:
        return normalize(vecs)
    else:
        return vecs


def normalize(vecs):
    return vecs / np.c_[np.sqrt((vecs ** 2).sum(axis=1))]


def create_vector_df(vectors, colname, sorted_vocabulary):
    """

    :param vectors:
    :param colname:
    :param sorted_vocabulary:
    :return:
    """
    vector_df = pd.DataFrame(vectors, index=sorted_vocabulary)
    vector_df[colname] = sorted_vocabulary
    return vector_df


def compute_pmi_vectors(
        m, n_pca_components, k=1.0, normalize_flag=True
):
    """

    :param m:
    :param n_pca_components:
    :param k:
    :param normalize_flag:
    :return:
    """
    # generate co-occurence matrix and outer product of normalized marginal token counts
    # and return in log space
    co_occ_matrix, marginal_log_matrix = generate_matrices(m)
    # Calculate PMI - p(AB)/(p(A)*p(B))
    pmi_matrix = construct_pmi_matrix(
        co_occ_matrix, marginal_log_matrix,
        k, normalize_flag
    )
    # PCA on PMI
    return compute_vectors(
        pmi_matrix, n_pca_components, normalize_flag
    )


def compute_glove_vectors(
        m, n_pca_components, normalize_flag=True
):
    """

    :param m:
    :param n_pca_components:
    :param normalize_flag:
    :return:
    """
    # generate co-occurence matrix and outer product of normalized marginal token counts
    # and return in log space
    co_occ_matrix = generate_matrices(m)
    # Calculate PMI - p(AB)/(p(A)*p(B))
    glove_matrix = generate_vectors(
        co_occ_matrix, n_pca_components, normalize_flag
    )
    # PCA on PMI
    return compute_vectors(
        glove_matrix, n_pca_components, normalize_flag
    )
