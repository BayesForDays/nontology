# nontology

This repo provides some utilities for building low-dimensional vector representations (_embeddings_) from discrete data. This toolbox is intended to facilitate creation of e.g. GloVe and PPMI style vectors at varying degrees of granularity. Most python packages that create these embeddings make bespoke creation of embedding vectors difficult. This package allows users to specify the amount of context they need to learn latent word representations from text. Functions within `parse_utils`, for example, allow for users to select small chunks to whole-sentence and whole-document co-occurrence statistics as the input to the matrix factorization step, depending on their needs.

#### Why (P)PMI or GloVe over word2vec?

It is notoriously difficult to integrate contextual information into `word2vec`-like algorithms. Entity embeddings and document-level representations are not as coherently interpreted within the semantics of the objective (e.g. "predict the missing word" can't encompass secondary features as formulated). PPMI and GloVe vectors are derived from simple matrix factorization procedures (PCA) over co-occurrence matrices, which increases the interpretability of the derived vectors. You can learn more about this method in [Levy and Goldberg (2014)](http://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf).

_(P)PMI_ differs from _GloVe_ in that it uses one additional transformation step. Whereas _GloVe_ uses a log-transformed co-occurrence matrix _c_, _PMI_ subtracts out a marginal matrix _m_ from _c_, which produces a matrix corresponding to 

```P(AB) / (P(A)*P(B))```

So, each of the cells in the PMI matrix corresponds to the likelihood that two words co-occur beyond what would be expected by chance, in log space. PMI can capture longer-distance dependencies than `word2vec` and has been used to learn the meanings of collocations, or phrases, so relative to GloVe, PMI-based vectors may show slightly different behavior. Which one you use should be in line with your theoretical question or application.

#### Why use this package?

One goal of mine is to make learning embeddings using the GloVe and PPMI algorithms easier for end users, in the same manner that the `gensim` implementations of `skip-gram` and `CBOW` take much of the guesswork out of training these models. This package still allows the user to obtain intermediate representations that may be useful for other applications.

There are two modules within `nontology` that contain tools for the easy construction of these vectors, `parse_utils` and `ppmi_matrix_utils`. 

##### parse_utils

For those who are highly comfortable with `sklearn`, `numpy`, `scipy` and `nltk`, the tools in `parse_utils` may not be necessary. One notable contribution is the ability to shrink sentences down to smaller chunks with the function `parse_utils.chunkify_docs`. The function breaks down a sentence like 

```[["This", "is", "an", "example", "sentence", "for", "github", "."]]```

into multiple chunks with a given window size, e.g.:

```
[["this", "is", "an"],
["is", "an", "example"],
["an", "example", "sentence"]]
...
```

The `chunkify_docs` capability makes the learning objective more similar to that of conventional algorithms like `skip-gram` and contextual bag-of-words (`CBOW`), and especially the vanilla implementation (i.e. just `word2vec`). My (=Cassandra) experience with using smaller documents is that syntactic similarity between words can come through  more easily. One downside, however, is that learning over many small documents inflates co-occurrence statistics (by a constant factor) relative to sentence-level or document-level counts, so please be mindful of this. 

Additionally, previous versions of `nontology` relied on `sklearn.feature_extraction.text.CountVectorizer`'s off-the-shelf tokenization, which performs poorly in many common cases, such as contractions or emoji. The current version allows the user to pre-tokenize using their algorithm of choice, while still allowing `CountVectorizer` to do the work. The default implementation of `parse_utils.tokenize` uses `nltk.tokenize.word_tokenize` and `nltk.tokenize.sent_tokenize`. If you are working with languages other than English or wish to use a different tokenization scheme, you can simply pass in pre-tokenized data to `parse_utils.make_sparse`. Of course, if you wish to construct sparse matrices yourself you're welcome to do so, in which case you should move on to the next section :smiley:

##### ppmi_matrix_utils

The ability to add metadata and fine-tune parameters is another advantage of the PPMI method. Whereas off-the-shelf implementations of `word2vec` make it difficult to include metadata, in `numpy`/`scipy` it is trivial to add features to a matrix of observations where each row corresponds to some set of counts of categorical variables. Furthermore, using different cutoffs for frequency (either minimums or maximums) may be an important parameter to tune for your application -- for example you may want to set a higher frequency cutoff for words than for entities, bigrams, trigrams, etc. Working with the matrices going into the PCA algorithm directly provides greater flexibility.


#### Examples:

First, start by importing the modules:

```
from nontology import parse_utils as pu, ppmi_matrix_utils as ppmi, live_similarity_functions as lsf
import pandas as pd
```

Load in some data, and then optionally tokenize it as you'd like. I recommend this route over using scikit-learn's off-the-shelf tokenizer. As a reminder, `word_tokenize` depends on `nltk.tokenize.word_tokenize`.

```
docs = open('example_file.txt').readlines()
word_tokenized = [pu.word_tokenize(x) for x in docs]
```

And define the dimensionality of your output vectors.
```
n_components = 100
```

##### Example 1: GloVe embeddings without windows
You might not want windows and would prefer to use the whole document as a sort of "context" for computing word vectors. Whether you want to do this will depend on your application -- more context can create noise, but can also capture long-distance relationships between words.

```
v, sparse = pu.make_sparse(
    docs_to_fit = word_tokenized,
    ngram_range=(1, 2)
)
m = ppmi.construct_co_occurrence_matrix(sparse)
vecs = ppmi.compute_vectors(m, n_components)
```

and even more simply the last two lines can be replaced by

```
vecs = ppmi.compute_glove_vectors(m, n_components)
```

##### Example 2: GloVe embeddings with windows

To set a window but otherwise train on the documents as normal, assign `docs_to_fit` to your original dataset, and `docs_to_transform` to be some `pu.chunkify_docs` variant. You can set the window to any size you'd like, though you may see diminishing returns eventually.

```
v, sparse = pu.make_sparse(
    docs_to_fit = word_tokenized,
    docs_to_transform = pu.chunkify_docs(
        word_tokenized, window=4
    ),
    ngram_range=(1, 2)
)
m = ppmi.construct_co_occurrence_matrix(sparse)
vecs = ppmi.compute_vectors(m, n_components)
```

This results in an n x k numpy array with n being the size of your vocabulary and k being the size of the components (here, 100). The implementation here will work reasonably well until ~45k features, depending on the density of your data.

The last two lines can again be replaced by `ppmi.compute_glove_vectors`.

##### Example 3: PPMI embeddings without windows

Getting PPMI values is a matter of adding a single line. 

```
cooc_m = ppmi.construct_co_occurrence_matrix(sparse)
marg_m = ppmi.construct_marginal_matrix(sparse)
ppmi_m = ppmi.construct_pmi_matrix(cooc_m, marg_m)
vecs = ppmi.compute_vectors(ppmi_m, n_components)
```

The lines above can also be written as follows:

```
cooc_m, marg_m = ppmi.construct_matrices(sparse)
ppmi_m = ppmi.construct_pmi_matrix(cooc_m, marg_m)
vecs = ppmi.compute_vectors(ppmi_m, n_components)
```

And most simply as below:

```
vecs = compute_pmi_vectors(sparse, n_components)
```


##### Example 4: Entity embeddings -- Coming soon!
Entity embeddings are easy. Simply include metadata as a set of features that you can vectorize. Any categorical variable can be easily vectorized by `CountVectorizer`. Simply use `pu.concatenate_sparse_matrices` and then compute the entity-term co-occurrence matrix.
