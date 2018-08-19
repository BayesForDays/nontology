import scipy as sp
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def generate_co_occurence_matrix(sparse_tokens):
    """
    Legacy name from 0.0.8. Replaced by construct_cooccurrence_matrix.
    :param sparse_tokens: A sparse matrix of counts
    :return: Co-occurrence matrix
    """
    return construct_co_occurrence_matrix(sparse_tokens)


def construct_co_occurrence_matrix(sparse_tokens):
    """
    Calculates a log-scaled co-occurrence matrix between all elements of a count matrix using a dot product.
    :param sparse_tokens: A sparse matrix of counts
    :return: Log-scaled co-occurrence matrix
    """
    # compute normalized log co-occurrence counts
    co_occur = (sparse_tokens.T).dot(sparse_tokens).todense()
    co_occur_log_matrix = sp.special.xlogy(
        sp.sign(co_occur), co_occur
    ) - np.log(co_occur.sum())
    np.fill_diagonal(co_occur_log_matrix, 0)
    return co_occur_log_matrix


def construct_marginal_matrix(sparse_tokens):
    """
    Compute outer product of normalized marginal token counts
    :param sparse_tokens: A sparse matrix of counts
    :return: The marginal matrix used to calculate PMI, in log probabilities
    """
    marginal = sparse_tokens.sum(axis=0).astype(np.uint32)
    marginal = marginal / marginal.astype(np.float32).sum()
    marginal_log_matrix = np.log(np.outer(marginal, marginal))
    return marginal_log_matrix


def generate_marginal_matrix(sparse_tokens):
    """
    Legacy name from 0.0.8. Replaced by construct_marginal_matrix.
    :param sparse_tokens: A sparse matrix of counts
    :return: The marginal matrix used to calculate PMI, in log probabilities
    """
    return construct_marginal_matrix(sparse_tokens)


def construct_pmi_matrix(
        co_occur_log_matrix, marginal_log_matrix,
        k=1.0, positive_only_flag=True
):
    """
    Calculates the PMI matrix
    :param co_occur_log_matrix: Sparse matrix of co-occurrences of categorical variables
    :param marginal_log_matrix: Matrix of marginal probabilities
    :param k: Smoothing parameter
    :param positive_only_flag: PMI matrix w/ negatives or PPMI without?
    :return: PMI matrix
    """
    if k < 0:
        k = 0.0
    pmi_matrix = co_occur_log_matrix - marginal_log_matrix - np.log(k)
    if positive_only_flag:
        pmi_matrix[pmi_matrix < 0] = 0  # positive PMI: don't include negative associations
    np.fill_diagonal(pmi_matrix, 0)
    return pmi_matrix


def generate_pmi_matrix(
        co_occur_log_matrix, marginal_log_matrix,
        k=1.0, positive_only_flag=True
):
    """
    Legacy name from 0.0.8. Replaced by construct_pmi_matrix.
    :param co_occur_log_matrix: Sparse matrix of co-occurrences of categorical variables
    :param marginal_log_matrix: Matrix of marginal probabilities
    :param k: Smoothing parameter
    :param positive_only_flag: PMI matrix w/ negatives or PPMI without?
    :return: PMI matrix
    """
    return construct_pmi_matrix(
        co_occur_log_matrix, marginal_log_matrix,
        k, positive_only_flag
)


def generate_matrices(sparse_tokens):
    """
    Legacy name from 0.0.8. Replaced by construct_matrices.
    :param sparse_tokens: A sparse matrix of counts
    :return: Co-occurrence and marginal matrices.
    """
    return construct_matrices(sparse_tokens)


def construct_matrices(sparse_tokens):
    """
    Convenience function for getting co-occurrence and marginal matrices.
    :param sparse_tokens: A sparse matrix of counts
    :return: Co-occurrence and marginal matrices.
    """
    co_occur_log_matrix = construct_co_occurrence_matrix(sparse_tokens)
    marginal_log_matrix = construct_marginal_matrix(sparse_tokens)
    return co_occur_log_matrix, marginal_log_matrix


def generate_vectors(m, n_components=100, normalize_flag=True):
    """
    Legacy name from 0.0.8. Replaced by compute_vectors.
    Computes low-dimensional representation of categorical (column) variables from a matrix m.
    :param m: Matrix (PMI, co-occurrence, etc.)
    :param n_components: Desired dimensionality of output vectors.
    :param normalize_flag: Whether to normalize vectors to length 1 or not.
    :return: Lower-dimensional representation of features.
    """
    return compute_vectors(
        m, n_components, normalize_flag
    )


def compute_vectors(m, n_components=100, normalize_flag=True):
    """
    Computes low-dimensional representation of categorical (column) variables from a matrix m.
    :param m: Matrix (PMI, co-occurrence, etc.)
    :param n_components: Desired dimensionality of output vectors.
    :param normalize_flag: Whether to normalize vectors to length 1 or not.
    :return: Lower-dimensional representation of features.
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
    """
    Makes vectors length 1
    :param vecs: Original vectors
    :return: Unit vectors
    """
    return vecs / np.c_[np.sqrt((vecs ** 2).sum(axis=1))]


def create_vector_df(vectors, colname, sorted_vocabulary):
    """
    Turns an array of learned vectors into a dataframe with an additional column that is the vocabulary.
    :param vectors: A 2d numpy array of n x k dimensions.
    :param colname: The name of the column (e.g. "entity_id" or "term")
    :param sorted_vocabulary: Vocabulary in the proper order. Typically vectorizer.get_feature_names().
    :return:
    """
    vector_df = pd.DataFrame(vectors, index=sorted_vocabulary)
    vector_df[colname] = sorted_vocabulary
    return vector_df


def compute_pmi_vectors(
        m, n_components, k=1.0, normalize_flag=True
):
    """
    Convenience function for taking a sparse matrix and learning vectors based on PPMI values.
    :param m: Sparse matrix of counts
    :param n_components: Desired dimensionality of output
    :param k: Smoothing parameter
    :param normalize_flag: Whether to make vectors length 1
    :return: Lower-dimensional representation of co-occurrence matrix trained on PPMI values.
    """
    # generate co-occurence matrix and outer product of normalized marginal token counts
    # and return in log space
    co_occ_matrix, marginal_log_matrix = construct_matrices(m)
    # Calculate PMI - p(AB)/(p(A)*p(B))
    pmi_matrix = construct_pmi_matrix(
        co_occ_matrix, marginal_log_matrix,
        k, normalize_flag
    )
    # PCA on PMI
    return compute_vectors(
        pmi_matrix, n_components, normalize_flag
    )


def compute_glove_vectors(
        m, n_components, normalize_flag=True
):
    """
    Convenience function for taking a sparse matrix and learning GloVe vectors based on co-occurrence values.
    :param m: Sparse matrix of counts
    :param n_components: Desired dimensionality of output
    :param normalize_flag: Whether to make vectors length 1
    :return: Lower-dimensional representation of co-occurrence matrix trained on log counts.
    """
    # generate co-occurence matrix and outer product of normalized marginal token counts
    # and return in log space
    co_occ_matrix = construct_co_occurrence_matrix(m)
    # PCA on co-occurrence matrix
    return compute_vectors(
        co_occ_matrix, n_components, normalize_flag
    )
