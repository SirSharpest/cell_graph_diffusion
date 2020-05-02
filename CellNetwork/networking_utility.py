import numpy as np
import pandas as pd


def check_negative_values(A):
    if np.any(A < 0):
        raise ValueError(f"Matrix cannot contain negative values! {A}")


def enforce_matrix_shape(I, O):
    if type(I) != np.ndarray or I.shape != O.shape:
        I = (I*O)
    return I


def G_to_pd(G, shape, D, rep_id):
    data = dict(G.nodes(data=True))
    for k, v in data.items():
        if 'pos' in data[k]:
            del data[k]['pos']
    df = pd.DataFrame(data).T
    df['shape'] = shape
    df['D'] = D
    df['rep'] = rep_id
    return df
