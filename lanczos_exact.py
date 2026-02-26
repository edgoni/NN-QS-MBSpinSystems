#Lanczos exact
import numpy as np
from utils import KitaevTransverse_H
import netket as nk
import pandas as pd

##Declaring KITAEV 
kitaev_graph= nk.graph.KitaevHoneycomb(extent=[3,3], pbc = True)
N = kitaev_graph.n_nodes
adj_list = kitaev_graph.adjacency_list()
direcciones = kitaev_graph.edge_colors
bonds = kitaev_graph.edges()

hi = nk.hilbert.Spin(s=1 / 2, N=kitaev_graph.n_nodes)
kitaev_graph.hi=hi
h = 1


jz_values = np.linspace(0, 1, 11)
path_energies = 'energies_eigenvecs'

for i, jz in enumerate(jz_values):
    jx = jy = (1 - jz) / 2
    H = KitaevTransverse_H(direcciones, bonds, Jx=jx, Jy=jy, Jz=jz, h=0, hi=hi)
    eigenvals, eigenvecs = nk.exact.lanczos_ed(H,k=50,compute_eigenvectors=True)

    filename = f"{path_energies}.npz"
    np.savez(filename, energies=eigenvals, vecs=eigenvecs)
