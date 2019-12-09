import numpy as np
import image_to_network
import argparse
from random import choice
import networkx as nx
from scipy.optimize import curve_fit
from networkx.generators.lattice import grid_2d_graph
from networkx.generators.lattice import hexagonal_lattice_graph
from networkx.generators.lattice import triangular_lattice_graph


class CellNetwork(nx.classes.graph.Graph):
    def __init__(self, A=None, d=0, n=0, incoming_graph_data=None, **args):
        super(CellNetwork, self).__init__(incoming_graph_data=None,
                                          **args)
        self.node_attr = 'C'
        self.edge_attr = 'E'
        self.edge_lim = 1/10
        self.PD = 1  # Number of PD per unit of cellwall
        self.units = 5  # micrometers
        self.q = 1/10  # amount of C a single PD could push through itself to neighbours

        if n > 0:
            if d == 2:
                self.add_existing_shape(self.make_2N(n))
            else:
                self.add_existing_shape(
                    nx.generators.random_regular_graph(d, n, seed=4))
        elif A is not None:
            self.add_existing_shape(nx.from_numpy_matrix(A))

    def make_2N(self, N):
        A = np.zeros((N, N))
        for i in range(N):
            if i > 0 and i < N-1:
                A[i, i+1] = 1
                A[i, i-1] = 1
            elif i == 0:
                A[i, i+1] = 1
                A[i, N-1] = 1
            else:
                A[i, 0] = 1
        return nx.from_numpy_matrix(A)

    def get_ego_graph(self, r=1, C=None):
        if C is None:
            C = self.get_centre_node()
        Gn = nx.generators.ego.ego_graph(self, C, radius=r, center=True)
        return Gn

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

    def add_existing_shape(self, G):
        self.__dict__.update(G.__dict__)
        self.set_edges()
        self.set_concentration()

    def update_node_attribute(self, attr, new_attrs):
        nx.set_node_attributes(self, {n: v for (n, d), v in zip(
            self.nodes(data=True), new_attrs)}, attr)

    def set_random_edge_weights(self, mu, sigma, seed=1):
        np.random.seed(seed)
        E = np.random.normal(mu, sigma, self.number_of_edges())
        E[E < 0] = 0
        E = np.around(E, 2)
        self.set_edges(E)

    def update_edge_attribute(self, attr, new_attrs):
        nx.set_edge_attributes(self, {(u, v): va for (u, v, a), va in zip(
            self.edges(data=True), new_attrs)}, attr)

    def get_centre_node(self):
        centre = nx.algorithms.distance_measures.center(self)[0]
        return centre

    def get_centre_c(self):
        return self.nodes()[self.get_centre_node()][self.node_attr]

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
        for i in range(epochs):
            A, C = self.extract_graph_info()
            self.check_negative_values(C)
            E = (self.enforce_matrix_shape(
                self.weights_to_A(), A) * D)*np.diag(C)
            q_hat = self.PD * self.q
            E_hat = E * q_hat
            I = np.sum(E_hat, axis=1)
            O = np.sum(E_hat, axis=0)
            Cn = np.diag(np.diag(C) + dt*D*(I-O))
            for f, args in zip(rules, rules_args):
                Cn = f(Cn, *args)
            self.update_node_attribute(self.node_attr, np.diag(Cn))

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

    def generate_G_with_D_degrees(self, D):
        g = nx.Graph()
        for n in range(D):
            pass
