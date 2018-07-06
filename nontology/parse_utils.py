"""

"""

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize


def vectorize(df, colname, min_df=1):
    # transform into useful features
    df[colname] = df[colname].astype('unicode')
    vectorizer = CountVectorizer(min_df=min_df)
    vectorized_ = vectorizer.fit_transform(df[colname])
    return vectorizer, vectorized_


def preprocess(column):
    raise NotImplementedError


def concatenate_sparse_matrices(list_of_sparse_matrices):
    """

    :param list_of_sparse_matrices:
    :return:
    """
    concatenated = hstack(list_of_sparse_matrices)
    return concatenated
