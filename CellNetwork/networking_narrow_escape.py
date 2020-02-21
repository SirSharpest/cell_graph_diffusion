from .networking_utility import multi_escp
import numpy as np


def escape_through_network(G, D, tt, Rn, Eps, particles=1000):

    # Rn = array of sizes
    # Eps = array of PD radius

    # Calculating neighbours is intensive, lets do it once just.
    nbrs = {n[0]: [ns for ns in G.neighbors(n[0])] for n in G.nodes(data=True)}
    weights = G.weights_to_A()
    Cn = np.zeros(G.number_of_nodes())
    for cell in G.nodes(data=True):
        for p0 in range(int(cell[1]['C']*particles)):
            cur_pos = cell[0]
            te = 0
            while te < tt:
                N = len(nbrs[cur_pos])
                # calculate escape time
                # Assume each PD field is equal ~
                w = weights[cur_pos][nbrs[cur_pos]]
                r = Rn[cur_pos]
                ep = Eps[cur_pos]
                te += multi_escp(r, D, N, ep)
                cur_pos = np.random.choice(nbrs[cur_pos], p=w/w.sum())
            Cn[cur_pos] += 1
