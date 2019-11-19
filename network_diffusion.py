import numpy as np
import image_to_network
import argparse
from random import choice
import networkx as nx
from scipy.optimize import curve_fit
from networkx.generators.lattice import grid_2d_graph, hexagonal_lattice_graph, triangular_lattice_graph


class CellNetwork(nx.classes.graph.Graph):
    def __init__(self,  incoming_graph_data=None, **args):
        super(CellNetwork, self).__init__(incoming_graph_data=None, **args)
        self.node_attr = 'C'
        self.edge_attr = 'E'
        self.edge_lim = 1/10

    def generate_shape(self, shape, n=1, m=1):
        # Warning, will erase currently held information in the network
        func_dict = {'rectangle': grid_2d_graph,
                     'hexagon': hexagonal_lattice_graph,
                     'triangle': triangular_lattice_graph}
        f = func_dict[shape]
        G = f(m, n)
        self.__dict__.update(G.__dict__)
        self.set_edges()
        self.set_concentration()

    def update_node_attribute(self, attr, new_attrs):
        nx.set_node_attributes(self, {n: v for (n, d), v in zip(
            self.nodes(data=True), new_attrs)}, attr)

    def update_edge_attribute(self, attr, new_attrs):
        nx.set_edge_attributes(self, {(u, v): va for (u, v, a), va in zip(
            self.edges(data=True), new_attrs)}, attr)

    def set_concentration(self, C=None):
        if C is not None:
            self.update_node_attribute(self.node_attr, C)
            return
        centre = nx.algorithms.distance_measures.center(self)[0]
        idx = list(self.nodes()).index(centre)
        IC = np.zeros(self.number_of_nodes())
        IC[idx] = 1
        self.update_node_attribute(self.node_attr, IC)

    def set_edges(self, E=None):
        if E is not None:
            self.update_edge_attribute(self.edge_attr, E)
            return
        IC = np.ones(self.number_of_edges()) * self.edge_lim
        self.update_edge_attribute(self.edge_attr, IC)

    def get_concentration(self):
        return list(nx.get_node_attributes(self, self.node_attr).values())

    def get_weights(self):
        return list(nx.get_edge_attributes(self, self.edge_attr).values())

    def diffuse(self, D, dt, epochs, rules=[], rules_args=[]):
        A, C = self.extract_graph_info()
        E = self.weights_to_A()
        check_negative_values(C)
        self.A = np.array(A)
        C = np.copy(C)
        E = enforce_matrix_shape(E, A) * dt
        Mx = enforce_matrix_shape(self.edge_lim, A) * dt
        Ex = E*np.diag(C)
        Ex[Ex > Mx] = Mx[Mx < Ex]
        I = np.sum(Ex, axis=1)
        O = np.sum(Ex, axis=0)
        C = np.diag(np.diag(C)-O+I)

        for f, args in zip(rules, rules_args):
            C = f(C, *args)

        return C

    def extract_graph_info(self):
        A = nx.to_numpy_array(self)
        A[A > 0] = 1
        C = np.diag(np.array(self.get_concentration()))
        return A, C

    def check_negative_values(self, A):
        if np.any(A < 0):
            raise ValueError(f"Matrix cannot contain negative values! {A}")

    def weights_to_A(self):
        A = nx.to_numpy_array(self)
        W1 = np.triu(A)
        W2 = np.tril(A)
        W = np.zeros(A.shape)
        for w in (W1, W2):
            list_of_coords = np.where(w == 1)
            w[list_of_coords] = self.get_weights()
            W += w
        return W

    def enforce_matrix_shape(self, I, O):
        if type(I) != np.ndarray or I.shape != O.shape:
            I = (I*O)
        return I


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
        Y[1: -1] = np.diag(C)
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
