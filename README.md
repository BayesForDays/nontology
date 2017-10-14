# nontology

This repo provides some utilities for building "embeddings" from discrete data.
For example, we might be interested in learning latent representations of entities.
To do this, we use text associated with those entities and treat the entities and the words as features.

The bare bones contents you need to build embeddings are all in `nontology.ppmi_matrix_utils`:

* Observations containing discrete features (e.g. words, entities, concepts, etc.)
* Each row is an observation or a "document" containing words, entities, concepts, etc.
* `vectorize` takes a string representation of an observation and transforms it into a sparse vector of counts of words, entities, concepts, etc.
* `ppmi_matrix_utils` contains a few other functions
    * `generate_co_occurrence_matrix`
    * `generate_marginal_matrix`
    * `generate_pmi_matrix`
    * `generate_vectors`
* If you want to build your own features, you first need to vectorize your "text"
* There are optionally utilities that will let you do this for two sets of features that come from the same dataframe but with different criteria. For example, if you have a set of `participants`s associated with an experiment generating `utterance`s, you might not want the same frequency cutoff as you use for words from `utterance`.
    * You can then call `generate_pmi_matrix` on that output in the same way that you would in the single feature set case.
    * `generate_vectors` will take this PMI matrix and create vectors representing each of you two feature sets.
    * `create_x_and_y_vectors` will take the vectors you learned and output two numpy arrays for both feature sets
    * `create_vector_df` will take any set of features and output a human readable dataframe


Here is an example snippet for how to do this with a very simple dataframe with a column `text`:

```
from nontology import ppmi_matrix_utils as ppmi
import pandas as pd

df = pd.read_csv("example_df.txt")

```

To turn these notes into features, we can call `ppmi.vectorize`:

```
vectorizer, text = ppmi.vectorize(df, 'text')
# this returns two items:
#(CountVectorizer(analyzer=u'word', binary=False, decode_error=u'strict',
#         dtype=<type 'numpy.int64'>, encoding=u'utf-8', input=u'content',
#         lowercase=True, max_df=1.0, max_features=None, min_df=1,
#         ngram_range=(1, 1), preprocessor=None, stop_words=None,
#         strip_accents=None, token_pattern=u'(?u)\\b\\w\\w+\\b',
#         tokenizer=None, vocabulary=None),
# <5x80 sparse matrix of type '<type 'numpy.int64'>'
# 	with 105 stored elements in Compressed Sparse Row format>)
```

Now that we have these sparse features, we can call various functions to get dense vectors.

```
n_components = 4

co_occ_matrix = ppmi.generate_co_occurrence_matrix(text)
marg_matrix = ppmi.generate_marginal_matrix(text)
pmi_matrix = ppmi.generate_pmi_matrix(co_occ_matrix, marg_matrix)
vecs = ppmi.generate_vectors(pmi_matrix, n_components)
```

These vectors we can now turn into a dataframe to inspect them.

```
vocabulary = vectorizer.vocabulary_
colname = 'token'      # calling this token because these are words

vector_df = ppmi.create_vector_df(vecs, colname, vocabulary)
```

Which gives output like so:

```
                 0             1             2             3     token
another  -1.000000 -7.193832e-16  2.854118e-16  5.994860e-16   another
here     -0.206284 -5.649327e-01 -2.606189e-18  7.989355e-01      here
is        1.000000  1.121612e-16  5.059776e-17 -1.145400e-16        is
one      -0.206284 -1.883109e-01  7.989355e-01 -5.326236e-01       one
sentence -0.206284  9.415545e-01  4.433294e-17  2.663118e-01  sentence
this     -0.206284 -1.883109e-01 -7.989355e-01 -5.326236e-01      this
```


The module `live_similarity_functions` will also give you token-token or any kind of x-y similarity you want. You can create a matrix like this:


```
from nontology import live_similarity_functions as lsf

x_x_matrix = lsf.make_live_x_y_matrix(vector_df, colname, vector_df, colname, n_components)
```

Which gives output like so (a symmetric similarity matrix):

```
           another      here        is       one  sentence      this
another   1.000000  0.206284 -1.000000  0.206284  0.206284  0.206284
here      0.206284  1.000000 -0.206284 -0.276596 -0.276596 -0.276596
is       -1.000000 -0.206284  1.000000 -0.206284 -0.206284 -0.206284
one       0.206284 -0.276596 -0.206284  1.000000 -0.276596 -0.276596
sentence  0.206284 -0.276596 -0.206284 -0.276596  1.000000 -0.276596
this      0.206284 -0.276596 -0.206284 -0.276596 -0.276596  1.000000
```