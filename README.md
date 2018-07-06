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


#### Examples: Coming soon!