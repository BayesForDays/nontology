import numpy as np
import pandas as pd


def make_live_x_y_matrix(x_data, xcol, y_data, ycol, n_components):
    x_numeric = np.array(x_data[x_data.columns[0:n_components]])
    y_numeric = np.array(y_data[y_data.columns[0:n_components]])
    x_y_matrix = pd.DataFrame(y_numeric.dot(x_numeric.T),
                              columns=[x.encode('utf-8') for x in x_data[xcol].values],
                              index=[y.encode('utf-8') for y in y_data[ycol].values])
    return x_y_matrix


def get_explanation(x1, x2, x_y_matrix):
    x1_dict = set(show_n_most_similar_x_to_y(x1, x_y_matrix, top_n=10).to_dict().keys())
    x2_dict = set(show_n_most_similar_x_to_y(x2, x_y_matrix, top_n=10).to_dict().keys())
    shared = list(x1_dict.intersection(x2_dict))
    return shared


def show_n_most_similar_x_to_y(x_name, x_y_matrix, top_n=5):
    return x_y_matrix.loc[x_name].sort_values(ascending=False).head(top_n)

