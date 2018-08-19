"""

"""

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from nltk.tokenize import sent_tokenize, word_tokenize


def tokenize(docs, word_tokenize_flag=1):
    """

    :param docs:
    :param word_tokenize_flag:
    :return:
    """
    sent_tokenized = []
    for d_ in docs:
        sent_tokenized += sent_tokenize(d_)
    if word_tokenize_flag==1:
        word_tokenized = []
        for sent in sent_tokenized:
            word_tokenized.append(word_tokenize(sent))
        return word_tokenized
    elif word_tokenize_flag==0:
        return sent_tokenized


def no_tokenization(tokens):
    """

    :param tokens:
    :return:
    """
    return tokens


def chunkify_docs(docs, window=None):
    """
    Turn a list of sentences (as lists of words) into chunks with a window for narrower training.
    :param docs: A list of lists of words
    :param window: Context window size to consider for each word. Defaults to all.
    :return:
    """
    smaller_docs = []
    if window is None:
        #then don't do anything
        smaller_docs = docs
    else:
        for doc in docs:
            smaller_docs += chunkify_doc(doc, window)
    return smaller_docs


def chunkify_doc(doc, window):
    """
    Turn a sentence (a list of words) into chunks with a window.
    :param doc: A list of words
    :param window: Context window size to consider for each word. Defaults to all.
    :return:
    """
    taller_doc = []
    if len(doc) >= window:
        for ix, word in enumerate(doc): # iterate each index
            taller_doc.append(doc[ix - window : ix + window])
    return taller_doc


def make_sparse(
        docs_to_fit, min_df=50, stop_words=None,
        docs_to_transform=None
):
    """
    Take a pre-tokenized document and turn into a sparse matrix.
    :param docs_to_fit: A list of lists of tokenized words to build the vocabulary from.
    :param min_df: Number of records that a word should appear in to be stored as a feature.
    :param stop_words: List of words to exclude, if any.
    :param docs_to_transform: A list of lists of tokenized words to transform. If none, we transform the first argument.
    :return:
    """
    cv = CountVectorizer(
        tokenizer=no_tokenization, preprocessor=None,
        stop_words=stop_words, lowercase=False, min_df=min_df
    )
    if docs_to_transform is None:
        return cv, cv.fit_transform(docs_to_fit)
    elif docs_to_transform is not None:
        cv.fit(docs_to_fit)
        return cv, cv.transform(docs_to_transform)


def preprocess(column):
    raise NotImplementedError


def concatenate_sparse_matrices(list_of_sparse_matrices):
    """
    Combine matrices for e.g. matrix embeddings.
    :param list_of_sparse_matrices: A list of sparse (csr) matrices, output of CountVectorizer.
    :return:
    """
    concatenated = hstack(list_of_sparse_matrices)
    return concatenated
