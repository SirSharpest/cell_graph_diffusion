import numpy as np
from .networking_nx import extract_graph_info, update_node_attribute, weights_to_A, DEFAULT_C
from .networking_utility import enforce_matrix_shape, check_negative_values

def diffuse(G, D, dt, epochs, q=1, rules=[], rules_args=[]):
    A, C = extract_graph_info(G)
    E = (enforce_matrix_shape(
        weights_to_A(G), A))
    q_hat = E * q * D * dt
    for i in range(epochs):
        check_negative_values(C)
        E_hat = np.diag(C) * q_hat
        I = np.sum(E_hat, axis=1)
        O = np.sum(E_hat, axis=0)
        C = np.diag(np.diag(C) + (I-O))
        for f, args in zip(rules, rules_args):
            C = f(C, *args)
        update_node_attribute(G, DEFAULT_C, np.diag(C))
