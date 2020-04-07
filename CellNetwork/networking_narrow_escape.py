from .networking_utility import multi_escp
from .networking_nx import weights_to_A, update_node_attribute, DEFAULT_C
import numpy as np
import multiprocessing
from os import cpu_count


def move_particles(G, particles, nbrs, Rn, Eps, D, weights, tt, pc_stay, lim_moves=1000):
    np.random.seed()
    Cn = np.zeros(G.number_of_nodes())
    for cell in G.nodes(data=True):
        for _ in range(int(cell[1]['C']*particles)):
            cur_pos = cell[0]
            te = 0
            moves = 0

            while te < tt:
                N = len(nbrs[cur_pos])
                # calculate escape time
                # Assume each PD field is equal ~
                w = weights[cur_pos][nbrs[cur_pos]]
                te += multi_escp(Rn[cur_pos], D, N, Eps[cur_pos])
                if te > tt:
                    break
                odds = [n for n in nbrs[cur_pos]]
                odds.append(cur_pos)
                w = [wi for wi in (w/w.sum())]
                w = [wi - pc_stay/len(w) for wi in w]
                w.append(pc_stay)
                w = np.array(w)
                cur_pos = np.random.choice(odds, p=w)
                if moves == lim_moves:
                    raise ValueError(
                        'Too many movements made - check parameters!')
                moves += 1
            Cn[cur_pos] += 1
    Cn /= particles
    return Cn


def escape_through_network(G, D, tt, Rn, Eps, particles=1000, pc_stay=0.5, processes=1):
    # Rn = array of sizes
    # Eps = array of PD radius
    # Calculating neighbours is intensive, lets do it once just.
    nbrs = {n[0]: [ns for ns in G.neighbors(n[0])] for n in G.nodes(data=True)}
    weights = weights_to_A(G)
    rep_args = [[G, int(particles//processes), nbrs, Rn, Eps, D, weights, tt, pc_stay]
                for _ in range(processes)]
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.starmap(move_particles, rep_args)
    Cn = np.mean(np.array(results), axis=0)
    update_node_attribute(G, DEFAULT_C, Cn)
