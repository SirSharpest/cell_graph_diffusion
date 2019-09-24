import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import dilation, disk
from skimage.measure import label, regionprops
from skimage.color import label2rgb, rgb2gray
from skimage.future import graph
from skimage.io import imread
import networkx as nx

cutoff = 2000


def read_img_pair(img_loc):
    image = imread(img_loc)[:, :, :3]
    w = image.shape[1]//2
    mask_image = image[:, :w, :]
    input_image = image[:, w:, :]
    return input_image, mask_image


def get_img_labels_edge_map(img):
    if isinstance(img, str):
        orig = imread(img)
    else:
        orig = img[:, :, :3]

    image = np.copy(orig)
    mask = np.where(image[:, :, 0] > 200)
    edge_map = np.zeros(image.shape)
    edge_map[mask] = [1, 1, 1]
    edge_map[np.where(edge_map > 0)] = 1
    edge_map = rgb2gray(edge_map)

    image[mask] = [0, 0, 0]
    binary = image > 10
    image = rgb2gray(binary)
    image[np.where(image > 0)] = 1

    image_labels = label(image)
    image_labels = dilation(image_labels, disk(5))

    props = regionprops(image_labels)
    for p in props:
        if p.area > cutoff:
            continue
        x, y = np.split(p.coords, [-1], axis=1)
        image_labels[x, y] = 0
    return orig, image_labels, edge_map


def get_network_from_paired(paired):
    inp, mask = read_img_pair(paired)
    g = make_network(mask)
    _, image_labels, edge_map = get_img_labels_edge_map(mask)
    grayvals = calc_grayvalues(inp, image_labels, g)
    for n, data in g.nodes(data=True):
        data['intensity'] = grayvals[n]
    return g


def calc_grayvalues(inp, image_labels, g):
    props = regionprops(image_labels)
    mask2d = np.zeros(image_labels.shape)
    grayvals = {}
    for (idx, _), p in zip(g.nodes(data=True), props):
        mask2d = np.zeros(image_labels.shape)
        x, y = np.split(p.coords, [-1], axis=1)
        mask2d[x, y] = 1
        cell = np.where(mask2d, rgb2gray(inp), 0)
        grayvals[idx] = np.around(np.sum(cell)/p.area, 2)
    return grayvals


def make_network(img_loc, draw=False, save_loc=None):
    orig, image_labels, edge_map = get_img_labels_edge_map(img_loc)
    g = graph.rag_boundary(image_labels, edge_map)
    g.remove_node(0)
    if draw:
        fig, ax = plt.subplots(2, 2, figsize=(15, 15), dpi=200)
        ax = ax.ravel()
        ax[0].imshow(orig, cmap='gray')
        ax[2].imshow(label2rgb(image_labels, image=orig))
        ax[1].imshow(edge_map)
        lc = graph.show_rag(image_labels, g, edge_map,
                            ax=ax[3], edge_width=5, edge_cmap='Blues')

        fig.colorbar(lc, fraction=0.03, ax=ax[3])
        pos = {}
        for idx in list(g.nodes):
            pos[idx] = (np.array(g.nodes[idx]['centroid'])[::-1])
        nx.draw(g, pos, ax=ax[3])
        for a in ax:
            a.grid('off')
        fig.tight_layout()

        if save_loc is not None:
            fig.savefig(save_loc)
    else:
        # because if we don't draw then these features aren't added to the graph
        props = regionprops(image_labels)
        for (n, data), region, idx in zip(g.nodes(data=True), props, range(len(props))):
            data['centroid'] = tuple(map(int, region['centroid']))
            data['uid'] = idx
    return g
