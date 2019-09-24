import numpy as np
import image_to_network
import argparse
from random import choice


def diffuse(A, C, D, R, dt, rules=[]):
    A = np.array(A)
    C = np.copy(C)
    R = R/dt
    while True:
        I = np.diag(np.sum((A * (np.sum(C*R, axis=1))),
                           axis=1))
        O = C*R*D
        for f in rules:
            C = f(C)
        C = C-O+I
        yield C


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


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input")
    ap.add_argument("-d", "--draw", default='False')
    args = vars(ap.parse_args())
    return args


def main():
    args = get_args()
    g = image_to_network.make_network(args['input'], draw=args['draw'])


if __name__ == '__main__':
    main()
