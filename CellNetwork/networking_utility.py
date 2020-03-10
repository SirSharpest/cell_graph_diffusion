import numpy as np
from numpy import pi, log, sqrt

def multi_escp(r, D, N, ep):
    D = (sqrt(D) / r)**2
    ep  =  ep / r
    f = lambda ep: ep - ep**2/pi * log(ep) + ep**2/pi * log(2)
    k = lambda sig: (4*sig) / (pi - 4 * sqrt(sig))
    sig = (N * ep**2)/4
    return (f(ep)/(3*D*k(sig))) + 1/(15*D)


def check_negative_values(A):
    if np.any(A < 0):
        raise ValueError(f"Matrix cannot contain negative values! {A}")


def enforce_matrix_shape(I, O):
    if type(I) != np.ndarray or I.shape != O.shape:
        I = (I*O)
    return I
