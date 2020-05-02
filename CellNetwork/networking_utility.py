import numpy as np
from numpy import pi, log, sqrt


def multi_escp(r, D, N, ep, ignore_error=False):
    D0 = D
    Ep0 = ep
    r0 = r

    D = (sqrt(D) / r)**2
    ep = (sqrt(ep) / r)**2
    def f(ep): return ep - ep**2/pi * log(ep) + ep**2/pi * log(2)
    def k(sig): return (4*sig) / (pi - 4 * sqrt(sig))
    sig = (N * ep**2)/4
    t = (f(ep)/(3*D*k(sig))) + 1/(15*D)
    if t < 0:
        if ignore_error == False:
            print(f"r:{r}  D: {D0}, N:{N}, ep:{Ep0}")
            print(f"r:{r}  D: {D}, N:{N}, ep:{ep}")
            raise ValueError(
                'Check parameters - escape time cannot be negative')
    return t


def check_negative_values(A):
    if np.any(A < 0):
        raise ValueError(f"Matrix cannot contain negative values! {A}")


def enforce_matrix_shape(I, O):
    if type(I) != np.ndarray or I.shape != O.shape:
        I = (I*O)
    return I


def calc_D_eff(r, D, N, ep, ignore_error=False):
    tau = multi_escp(r, D, N, ep)
    x2 = r**2
    Deff = x2 / (2*tau)
    return Deff
