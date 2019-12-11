from scipy.spatial import distance
import sys
import scipy.spatial
import scipy as sp
import numpy as np
import image_to_network
import networkx as nx
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
        self.PD_attr = 'PD'
        self.unit = 5  # micrometers
        self.q = 1/10  # amount of C a single PD could push through itself to neighbours

        if n > 0:
            if d == 2:
                self.add_existing_shape(self.make_2N(n))
            else:
                self.add_existing_shape(
                    nx.generators.random_regular_graph(d, n))
        elif A is not None:
            self.add_existing_shape(nx.from_numpy_matrix(A))

    def set_PD(self, PD):
        self.PD = PD

    def set_q(self, q):
        self.q = q

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

    def set_random_edge_weights(self, mu, sigma):
        E = np.random.normal(mu, sigma, self.number_of_edges())
        E[E < 0] = 0
        E = np.around(E, 2)
        self.set_edges(E)

    def update_edge_attribute(self, attr, new_attrs):
        nx.set_edge_attributes(self, {(u, v): va for (u, v, a), va in zip(
            self.edges(data=True), new_attrs)}, attr)

    def set_random_PD_ratios(self, mu, sigma):
        P = np.random.normal(mu, sigma, self.number_of_edges())
        P[P < 0] = 0
        P = np.around(P, 2)
        self.update_edge_attribute(self.PD_attr, P)

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

    def get_concentration(self, names=False):
        if names:
            return nx.get_node_attributes(self, self.node_attr)
        return list(nx.get_node_attributes(self, self.node_attr).values())

    def get_weights(self):
        return list(nx.get_edge_attributes(self, self.edge_attr).values())

    def diffuse(self, D, dt, epochs, rules=[], rules_args=[]):

        A, C = self.extract_graph_info()
        E = np.floor(self.enforce_matrix_shape(
            self.weights_to_A(), A)) / self.unit
        q_hat = self.PD * self.q * D
        for i in range(epochs):
            E_hat = E*np.diag(C)*q_hat
            I = np.sum(E_hat, axis=1)
            O = np.sum(E_hat, axis=0)
            C = np.diag(np.diag(C) + dt*(I-O))
        self.update_node_attribute(self.node_attr, np.diag(C))

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

    def PD_to_A(self):
        A = nx.to_numpy_array(self)
        W1 = np.triu(A)
        W2 = np.tril(A)
        W = np.zeros(A.shape)
        for w in (W1, W2):
            list_of_coords = np.where(w == 1)
            w[list_of_coords] = list(
                nx.get_edge_attributes(self, self.PD_attr).values())
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

    def generate_voronoi(self, ncells, bboxsize=1):
        def centeroidnp(arr):
            length, dim = arr.shape
            return np.array([np.sum(arr[:, i])/length for i in range(dim)])
        cells = np.around(np.random.rand(ncells, 2), 2)*bboxsize
        cells[cells < 0.01] = cells[cells < 0.01] + 0.01
        cells[cells > bboxsize-0.01] = cells[cells > bboxsize-0.01] - 0.01

        bounding_box = np.array([0., bboxsize, 0., bboxsize])
        vor = self.voronoi(cells, bounding_box)

        G = CellNetwork()

        for idx, region in enumerate(vor.filtered_regions):
            G.add_node(idx)
            vertices = vor.vertices[region + [region[0]], :]
            P = 0
            for idxy in range(len(vertices)-1):
                P += distance.euclidean(vertices[idxy], vertices[idxy+1])
            G.nodes[idx]['P'] = P

            G.nodes[idx]['x'], G.nodes[idx]['y'] = centeroidnp(vertices)

            for idy, r in enumerate(vor.filtered_regions):
                if idy == idx:
                    continue
                m = list(set(set(region) & set(r)))
                if len(m) > 1:
                    G.add_edge(idx, idy,
                               E=np.around(distance.euclidean(vor.vertices[m[0]],
                                                              vor.vertices[m[1]]), 2))
        self.__dict__.update(G.__dict__)

    def voronoi(self, cells, bounding_box):
        """
        Solution based on
        https://stackoverflow.com/a/33602171
        with only slight modification
        """
        def in_box(cells, bounding_box):
            return np.logical_and(np.logical_and(bounding_box[0] <= cells[:, 0],
                                                 cells[:, 0] <= bounding_box[1]),
                                  np.logical_and(bounding_box[2] <= cells[:, 1],
                                                 cells[:, 1] <= bounding_box[3]))
        eps = sys.float_info.epsilon
        # Select cells inside the bounding box
        i = in_box(cells, bounding_box)
        # Mirror points
        points_center = cells[i, :]
        points_left = np.copy(points_center)
        points_left[:, 0] = bounding_box[0] - \
            (points_left[:, 0] - bounding_box[0])
        points_right = np.copy(points_center)
        points_right[:, 0] = bounding_box[1] + \
            (bounding_box[1] - points_right[:, 0])
        points_down = np.copy(points_center)
        points_down[:, 1] = bounding_box[2] - \
            (points_down[:, 1] - bounding_box[2])
        points_up = np.copy(points_center)
        points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
        points = np.append(points_center,
                           np.append(np.append(points_left,
                                               points_right,
                                               axis=0),
                                     np.append(points_down,
                                               points_up,
                                               axis=0), axis=0), axis=0)
        # Compute Voronoi
        vor = sp.spatial.Voronoi(points)
        # Filter regions
        regions = []
        for region in vor.regions:
            flag = True
            for index in region:
                if index == -1:
                    flag = False
                    break
                else:
                    x = vor.vertices[index, 0]
                    y = vor.vertices[index, 1]
                    if not(bounding_box[0] - eps <= x and x <= bounding_box[1] + eps and
                           bounding_box[2] - eps <= y and y <= bounding_box[3] + eps):
                        flag = False
                        break
            if region != [] and flag:
                regions.append(region)
        vor.filtered_points = points_center
        vor.filtered_regions = regions
        return vor
