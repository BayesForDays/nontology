This repo provides some utilities for building "embeddings" from discrete data. For example, in a knowledge graph context, we are interested in learning latent representations of style variants. To do this, we use text when clients leave feedback about their purchases and the styles that they are leaving feedback about and treat those as features.

The bare bones contents you need to build embeddings are all in `nontology.ppmi_matrix_utils`:

* Observations containing discrete features (e.g. words, stylist ids, sku ids)
* Each row is an observation or a "document" containing words, stylist ids, skus, etc.
* `vectorize` takes a string representation of an observation and transforms it into a sparse vector of counts of words, stylist ids, sku ids, etc.
* `ppmi_matrix_utils` contains a few other functions
    * `generate_co_occurrence_matrix`
    * `generate_marginal_matrix`
    * `generate_pmi_matrix`
    * `generate_vectors`
* If you want to build your own features, you first need to vectorize your "text"
* There are optionally utilities that will let you do this for two sets of features that come from the same dataframe but with different criteria. For example, if you have a set of `style_variant_id`s associated with a fix, you might not want the same frequency cutoff as you use for words from `request_notes`.
    * `load_create_x_and_y_matrix` will read in two text columns and create two sets of features for you.
    * You can then call `generate_pmi_matrix` on that output in the same way that you would in the single feature set case.
    * `generate_vectors` will take this PMI matrix and create vectors representing each of you two feature sets.
    * `create_x_and_y_vectors` will take the vectors you learned and output two numpy arrays for both feature sets
    * `create_vector_df` will take any set of features and output a human readable dataframe
    * You can now call `write_vector_df` to put those features you learned on S3 and in hive.


Here is an example snippet for how to do this with a very simple dataframe:

```
from nontology import ppmi_matrix_utils as ppmi
from r2d2 import query_presto

qry = """
select client_comments
from prod.shipmentitem
where shipment_id=19368790
"""

df = query_presto(qry)

```

Which returns a dataframe that looks like this:

```
0  I wanted to exchange for an 8 but it looks like I'd be charged for it - I don't know if an 8 will work or not so I am not going to do the exchange. 10 fits in thighs, too big in the waist.
1  I have well developed arms LOL and I can't wear this with the sleeves up as it is too tight but I like the blouse itself.
2  The purple is pretty. It's not a pattern I'd usually pick out but I like it when it is on.
3  To me, this looks too costume jewelery-y for the price.
4  My only question about this and the other blouses in my shipment is to ask what the best thing is to wear underneath for job interviews. I haven't ever worn partially transparent/ see through blouses.
```

To turn these notes into features, we can call `ppmi.vectorize`:

```
vectorizer, vectorized_comments = ppmi.vectorize(df, 'client_comments')
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

co_occ_matrix = ppmi.generate_co_occurrence_matrix(vectorized_comments)
marg_matrix = ppmi.generate_marginal_matrix(vectorized_comments)
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
-----  ---------  ---------  ---------  ---------  -----
10     -0.295082   0.622295   0.66966   -0.277906  10
about  -0.311992  -0.410593  -0.505738  -0.691596  about
am     -0.295082   0.622295   0.66966   -0.277906  am
an      0.203656   0.713257  -0.540499   0.397051  an
and     0.658738  -0.716074  -0.11135   -0.202246  and
-----  ---------  ---------  ---------  ---------  -----
```


You can also do:

```
from nontology.example import word_example, svid_example

word_vector_df = word_example()
svid_vector_df = svid_example()
```