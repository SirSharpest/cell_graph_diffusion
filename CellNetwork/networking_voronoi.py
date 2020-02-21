import numpy as np
import networkx as nx
import sys 
from scipy.spatial import distance, Voronoi


def generate_voronoi(G, ncells, bboxsize=1):
    def centeroidnp(arr):
        length, dim = arr.shape
        return np.array([np.sum(arr[:, i])/length for i in range(dim)])
    cells = np.around(np.random.rand(ncells, 2), 2)*bboxsize
    cells[cells < 0.01] = cells[cells < 0.01] + 0.01
    cells[cells > bboxsize-0.01] = cells[cells > bboxsize-0.01] - 0.01

    bounding_box = np.array([0., bboxsize, 0., bboxsize])
    vor = G.voronoi(cells, bounding_box)

    G = nx.Graph()

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
    G = nx.relabel_nodes(G, {i: idx for idx, i in enumerate(G.nodes)})
    G.__dict__.update(G.__dict__)




def voronoi(cells, bounding_box):
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
    vor = Voronoi(points)
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
