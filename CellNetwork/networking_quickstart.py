import numpy as np
from .CellNetwork.networking_nx import set_concentration
from .CellNetwork.networking_nx import set_default_edge_weights
from .CellNetwork.networking_nx import generate_shape
from .CellNetwork.networking_nx import get_centre_node
from .CellNetwork.networking_diffuse import calc_D_eff, diffuse
from .CellNetwork.networking_utility import G_to_pd


class LatticeNetworkModel:
    def __init__(self, shape, sizeN, sizeM, IC='default'):
        self.G = generate_shape(shape, n=int(np.sqrt(sizeN)),
                                m=int(np.sqrt(sizeM)))
        set_default_edge_weights(G)
        self.N = np.array([len([_ for _ in self.G.neighbors(n)])
                           for n in self.G.nodes()])
        if IC is 'default':
            set_concentration(self.G)
        else:
            set_concentration(self.G, IC)
        self.totalTime = 0

    def set_model_parameters(self, D, avgCellR=50, PDR=5e-3, PDN=1e3,
                             cellSigmaPC=0, yGradientPC=0, deadCellPC=0):
        self.D = D
        self.cellSigmaPC = cellSigmaPC
        self.yGradientPC = yGradientPC
        self.deadCellPC = deadCellPC
        self.avgCellR = avgCellR
        self.PDR = PDR
        self.PDN = PDN
        self.PDArea = np.pi*(self.PDR**2)
        self.avgCellSA = (4*np.pi*(self.avgCellR**2))
        self.PD_per_um2 = PDN / self.avgCellSA
        self.apply_deadcells()
        self.apply_radius()

    def apply_deadcells(self):
        for i, cell in self.G.nodes(data=True):
            cell['deadcell'] = np.random.choice(
                [True, False], p=[self.deadCellPC, 1-self.deadCellPC])

    def apply_radius_G(self, upperLim=100, lowerLim=1):
        _, centreY = (self.G.nodes()[get_centre_node(self.G)]['x'],
                      self.G.nodes()[get_centre_node(self.G)]['y'])
        for k, v in self.G.nodes(data=True):
            noisy_size = np.random.normal(
                self.avgCellR, self.cellSigmaPC*self.avgCellR)
            r = noisy_size * (1 + self.yGradientPC * (v['y'] - centreY))
            if r < lowerLim or r > upperLim:
                r = self.avgCellR
            v['r'] = r

    def apply_radius(self):
        self.apply_radius_G()
        self.Rn = np.array([v['r'] for k, v in self.G.nodes(data=True)])
        self.PD_per_cell = np.around(self.PD_per_um2 * (4*np.pi*(self.Rn**2)))
        self.Ep = self.PD_per_cell * self.PDA
        self.Eps = self.Ep/self.PD_per_cell
        self.set_effective_diffusion()

    def run(self, seconds, dt=1e-4, reapply_randomDC=False, reapply_randomR=False):
        self.epochs = int(seconds/dt)
        self.totalTime += seconds
        if reapply_randomR:
            self.apply_radius()
        if reapply_randomDC:
            self.apply_deadcells()
        diffuse(self.G, self.Deff, dt,
                self.epochs, deadcells=True)

    def get_df(self, rep=1):
        df = G_to_pd(self.G, self.shape, self.Deff, rep)
        df['sigma'] = self.cellSigmaPC
        df['gradient'] = self.yGradientPC
        df['DC'] = self.deadCellPC
        df['time'] = self.totalTime
        return df

    def set_effective_diffusion(self):
        self.N = np.array([len([_ for _ in self.G.neighbors(n)])
                           for n in self.G.nodes()])
        self.Deff = np.array([calc_D_eff(r, self.D, n, ep)
                              for r, ep, n in zip(self.Rn,
                                                  self.Eps,
                                                  self.PD_per_cell)])
