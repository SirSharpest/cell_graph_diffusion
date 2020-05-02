from .networking_utility import multi_escp
from .networking_nx import weights_to_A, update_node_attribute, DEFAULT_C
import numpy as np
import multiprocessing


def move_particles(G, particles, nbrs, Rn, Eps, D, weights, tt,
                   pc_stay, sigma, lim_moves=1000, ignore_error=True):
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
                u_t = abs(multi_escp(Rn[cur_pos], D, N,
                                     Eps[cur_pos], ignore_error=ignore_error))
                st = abs(np.random.normal(u_t, sigma*u_t))
                te += st
                if te > tt:
                    break
                odds = [n for n in nbrs[cur_pos]]
                odds.append(cur_pos)
                if w.sum() == 0:
                    break
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


def escape_through_network(G, D, tt, Rn, Eps, particles=1000, pc_stay=0.5, sigma=0,
                           processes=1, deadcells=False):
    # Rn = array of sizes
    # Eps = array of PD radius
    # Calculating neighbours is intensive, lets do it once just.
    nbrs = {n[0]: [ns for ns in G.neighbors(n[0])] for n in G.nodes(data=True)}
    weights = weights_to_A(G)
    if deadcells:
        for cell in G.nodes(data=True):
            if 'deadcell' in cell[1]:
                if cell[1]['deadcell']:
                    weights[cell[0]] = 0
    rep_args = [[G, int(particles//processes), nbrs, Rn, Eps, D, weights,
                 tt, pc_stay, sigma]
                for _ in range(processes)]
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.starmap(move_particles, rep_args)
    Cn = np.mean(np.array(results), axis=0)
    update_node_attribute(G, DEFAULT_C, Cn)


def multi_escp(r, D, N, ep, ignore_error=False):
    D0 = D
    Ep0 = ep
    r0 = r

    D = (np.sqrt(D) / r)**2
    ep = (np.sqrt(ep) / r)**2
    def f(ep): return ep - ep**2/np.pi * np.log(ep) + ep**2/np.pi * np.log(2)
    def k(sig): return (4*sig) / (np.pi - 4 * np.sqrt(sig))
    sig = (N * ep**2)/4
    t = (f(ep)/(3*D*k(sig))) + 1/(15*D)
    if t < 0:
        if ignore_error == False:
            print(f"r:{r}  D: {D0}, N:{N}, ep:{Ep0}")
            print(f"r:{r}  D: {D}, N:{N}, ep:{ep}")
            raise ValueError(
                'Check parameters - escape time cannot be negative')
    return t
