import numpy as np
from tqdm import tqdm
from .networking_nx import extract_graph_info, update_node_attribute, weights_to_A, DEFAULT_C
from .networking_utility import enforce_matrix_shape, check_negative_values


def d_eff(D, P, L):
    with np.errstate(divide='ignore', invalid='ignore'):
        return D / ((D/(P*L)) + 1)


def apply_dead_cells(G, E):
    for cell in G.nodes(data=True):
        if 'deadcell' in cell[1]:
            if cell[1]['deadcell']:
                E[cell[0]] = 0


def diffuse(G, D, dt, epochs, deadcells=False, progress=True):
    E, C = extract_graph_info(G)
    if deadcells:
        apply_dead_cells(G, E)
    q_hat = E * D * dt

    if progress:
        for i in tqdm(range(epochs)):
            check_negative_values(C)
            E_hat = np.diag(C) * q_hat
            I = np.sum(E_hat, axis=1)
            O = np.sum(E_hat, axis=0)
            C = np.diag(np.diag(C) + (I-O))
    else:
        for i in range(epochs):
            check_negative_values(C)
            E_hat = np.diag(C) * q_hat
            I = np.sum(E_hat, axis=1)
            O = np.sum(E_hat, axis=0)
            C = np.diag(np.diag(C) + (I-O))
    update_node_attribute(G, DEFAULT_C, np.diag(C))
