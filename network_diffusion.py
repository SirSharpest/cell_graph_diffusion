import numpy as np
import image_to_network
import argparse
from random import choice
import networkx as nx
from scipy.optimize import curve_fit


class CellNetwork(nx.classes.graph.Graph):
    def __init__(self, incoming_graph_data=None, **args):
        super(CellNetwork, self).__init__(incoming_graph_data=None, **args)


def check_negative_values(A):
    if np.any(A < 0):
        raise ValueError(f"Matrix cannot contain negative values! {A}")


def enforce_matrix_shape(I, O):
    if type(I) != np.ndarray or I.shape != O.shape:
        I = (I*O)
    return I


def diffusion(A, C, E, dt, Mx=1/10, rules=[], rules_args=[]):
    """
    A  is the adj matrix
    C  is the concentration matrix
    E  is an edge matrix (or scalar) which dictates the flow between connections
    dt is delta time
    Mx is a maximum percentage which can flow between any two edges, either as a matrix or scalar
    rules and rules_args are additional functions which when given alter C prior to other operations
    """
    check_negative_values(C)
    A = np.array(A)
    C = np.copy(C)
    E = enforce_matrix_shape(E, A) * dt
    Mx = enforce_matrix_shape(Mx, A) * dt
    Ex = E*np.diag(C)
    Ex[Ex > Mx] = Mx[Mx < Ex]
    I = np.sum(Ex, axis=1)
    O = np.sum(Ex, axis=0)
    C = np.diag(np.diag(C)-O+I)
    for f, args in zip(rules, rules_args):
        C = f(C, *args)
    return C


def update_edge_weights(g, weights):
    W = weights_to_A(g, weights)
    idx_to_node = {idx: n for idx, (n, _) in enumerate(g.nodes(data=True))}
    edges = []
    for i, j in np.ndindex(W.shape):
        if W[j, i] > 0:
            edges.append((idx_to_node[j], idx_to_node[i],
                          {"weight": W[j, i]}))
    g.update(edges=edges)


def weights_to_A(g, weights):
    A = nx.to_numpy_array(g)
    W1 = np.triu(A)
    W2 = np.tril(A)
    W = np.zeros(A.shape)
    for w in (W1, W2):
        list_of_coords = np.where(w == 1)
        w[list_of_coords] = weights
        W += w
    return W


def rand_connection(G):
    r = [0, 1]
    s = [-2, -1, 0, +1, +2]
    while True:
        n0 = choice(list(G.nodes))
        idx = choice(r)
        n1 = list(n0)
        n1[idx] = (choice(s))
        edge = (n0, tuple(n1))
        yield edge


def extract_graph_info(G, vname='intensity'):
    A = nx.to_numpy_array(G)
    A[A > 0] = 1
    C = np.diag(np.array([d[vname] for n, d in G.nodes(data=True)]))
    return A, C


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input")
    ap.add_argument("-d", "--draw", default='False')
    args = vars(ap.parse_args())
    return args


def update_G_attribute(G, attr, new_attrs):
    nx.set_node_attributes(G, {n: v for (n, d), v in zip(
        G.nodes(data=True), new_attrs)}, attr)
    print(G.nodes(data=True))
    return True


def fit_G_to_data(G, ydata, tt, dt=60):
    def decay(A, gamma):
        # A is a 2D array
        B = np.diag(A).copy()
        B = B*(1-gamma)
        B[B < 0] = 0
        return np.diag(B)

    tmp = np.zeros(len(ydata)+2)
    tmp[1:-1] = ydata
    ydata = tmp
    xdata = np.arange(len(G.nodes)+2)
    tt = tt/dt  # Number of model estimations to make

    def f(x, *xargs):
        A, C = extract_graph_info(G)
        Y = np.zeros(len(A)+2)
        E = weights_to_A(G, np.array(xargs[:-1]))
        for _ in range(int(tt)):
            C = diffusion(A, C, E, dt, rules=[
                          decay], rules_args=[[xargs[-1]]], Mx=1)
        Y[1:-1] = np.diag(C)
        return Y
    p0 = np.array([1e-6 for _ in range(len(G.nodes))])
    p0[-1] = p0[-1]/100
    popt, pcov = curve_fit(f, xdata, ydata,
                           p0=p0,
                           bounds=(0, 0.5/dt))
    return popt, pcov


def main():
    args = get_args()
    g = image_to_network.make_network(args['input'], draw=args['draw'])


if __name__ == '__main__':
    main()
