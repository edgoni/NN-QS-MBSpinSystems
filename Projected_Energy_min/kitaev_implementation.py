
import netket as nk
import numpy as np

from cycles_Kitaev import obtener_plaquetas_kitaev
from cycles_Kitaev import build_Wp_operators
from utils import BestIterKeeper,make_extract_metrics, KitaevTransverse_H, make_extract_metrics_plaquete
from utils import identify_irreps


def declare_kitaev(extent=[3,3], pbc=True):
    kitaev_graph= nk.graph.KitaevHoneycomb(extent=extent, pbc = pbc)
    N = kitaev_graph.n_nodes
    adj_list = kitaev_graph.adjacency_list()
    direcciones = kitaev_graph.edge_colors
    bonds = kitaev_graph.edges()

    hi = nk.hilbert.Spin(s=1 / 2, N=kitaev_graph.n_nodes)
    kitaev_graph.hi=hi
    
    if extent[0] == 1 or extent[1] == 1:
        Wp_list = None
        Wp_total = None
        print("Warning: the Kitaev model with one of the dimensions equal to 1 is not well defined, since it does not have plaquettes. Consider using a larger extent for a more meaningful simulation.")
        return kitaev_graph, hi, Wp_list, Wp_total, N
    
    else:
        plaquetas, operadores = obtener_plaquetas_kitaev(kitaev_graph)
        Wp_list = build_Wp_operators(hi, plaquetas, operadores)
        Wp_total = sum(Wp_list)/len(Wp_list)

    
        return kitaev_graph, hi, Wp_list, Wp_total, N


def obtain_kitaev_symmetries(kitaev_graph, hi, use_symmetry=False):
    canonical_representation = nk.symmetry.canonical_representation(
        hilbert=hi,
        group=kitaev_graph.translation_group()
    )

    translations = kitaev_graph.translation_group()
    automorphisms = kitaev_graph.automorphisms()
    point_group = nk.utils.group.PointGroup(kitaev_graph.point_group(), ndim=2)
    space_group = kitaev_graph.space_group()
    irreps_matrices = space_group.irrep_matrices()
    character_table = space_group.character_table()
    
    return point_group, space_group, canonical_representation, irreps_matrices, character_table, translations,automorphisms, use_symmetry