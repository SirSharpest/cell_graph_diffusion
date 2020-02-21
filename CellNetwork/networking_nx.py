import sys
from scipy.spatial import distance
import scipy as sp
import numpy as np
import networkx as nx
from networkx.generators.lattice import grid_2d_graph
from networkx.generators.lattice import hexagonal_lattice_graph
from networkx.generators.lattice import triangular_lattice_graph


DEFAULT_ATTR = 'weight'
DEFAULT_C = 'C'


def get_ego_graph(G, r=1, C=None):
    if C is None:
        C = get_centre_node(G)
    Gn = nx.generators.ego.ego_graph(G, C, radius=r, center=True)
    return Gn


def update_node_attribute(G, attr, new_attrs):
    nx.set_node_attributes(G, {n: v for (n, d), v in zip(
        G.nodes(data=True), new_attrs)}, attr)


def get_centre_node(G):
    centre = nx.algorithms.distance_measures.center(G)[0]
    return centre


def get_centre_c(G):
    return G.nodes()[get_centre_node(G)][DEFAULT_C]


def set_edge_attribute(G, attr, new_attrs):
    nx.set_edge_attributes(G, {(u, v): va for (u, v, a), va in zip(
        G.edges(data=True), new_attrs)}, attr)


def set_random_edge_weights(G, mu, sigma):
    E = np.random.normal(mu, sigma, G.number_of_edges())
    E[E < 0] = 1
    E = np.around(E, 2)
    set_edge_attribute(G, DEFAULT_ATTR, E)


def set_concentration(G, C=None):
    if C is None:
        centre = nx.algorithms.distance_measures.center(G)[0]
        idx = list(G.nodes()).index(centre)
        IC = np.zeros(G.number_of_nodes())
        IC[idx] = 1
        update_node_attribute(G, DEFAULT_C, IC)
    else:
        update_node_attribute(G, DEFAULT_C, C)


def generate_shape(shape, n=1, m=1):
    func_dict = {'rectangle': grid_2d_graph,
                 'hexagon': hexagonal_lattice_graph,
                 'triangle': triangular_lattice_graph}
    f = func_dict[shape]
    G = f(m, n)
    set_edge_attribute(G, DEFAULT_ATTR, np.ones(G.number_of_edges()))
    set_concentration(G)
    return G


def extract_graph_info(G):
    A = nx.to_numpy_array(G)
    A[A > 0] = 1
    C = np.diag(np.array(get_concentration(G)))
    return A, C


def get_concentration(G, names=False):
    if names:
        return nx.get_node_attributes(G, DEFAULT_C)
    return list(nx.get_node_attributes(G, DEFAULT_C).values())


def weights_to_A(G, attr=None):
    if attr is None:
        attr = DEFAULT_ATTR
    A = nx.to_numpy_array(G)
    W1 = np.triu(A)
    W2 = np.tril(A)
    W = np.zeros(A.shape)
    try:
        for w in (W1, W2):
            list_of_coords = np.where(w == 1)
            w[list_of_coords] = get_weights(G, attr=attr)
            W += w
    except ValueError:
        print('Incorrect setup')
        return 0
    return W


def get_weights(G, attr=None):
    if attr is None:
        attr = DEFAULT_ATTR
    return list(nx.get_edge_attributes(G, attr).values())
