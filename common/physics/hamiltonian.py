from typing import Sequence, Tuple

import netket as nk


def build_kitaev_lattice(
    extent: Sequence[int] = (3, 3), pbc: bool = True
) -> Tuple[nk.graph.Graph, nk.hilbert.Spin]:
    """Build the Kitaev honeycomb graph and its spin-1/2 Hilbert space.

    :param extent: number of unit cells along each lattice direction.
    :param pbc: whether to use periodic boundary conditions.
    :return: (graph, hilbert)
    """
    graph = nk.graph.KitaevHoneycomb(extent=list(extent), pbc=pbc)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)
    graph.hi = hilbert
    return graph, hilbert


def KitaevTransverse_H(colores, enlaces, Jx, Jy, Jz, h, hi):
    """Build a Kitaev Hamiltonian with an optional transverse field.

    :param colores: direction/color of each bond in the graph (0=x, 1=y, 2=z)
    :param enlaces: bonds (edges) of the graph
    :param Jx: coupling on x-bonds
    :param Jy: coupling on y-bonds
    :param Jz: coupling on z-bonds
    :param h: uniform transverse field strength
    :param hi: NetKet Hilbert space
    """
    sigmax, sigmay, sigmaz = (
        nk.operator.spin.sigmax,
        nk.operator.spin.sigmay,
        nk.operator.spin.sigmaz,
    )
    couplings = {0: (Jx, sigmax), 1: (Jy, sigmay), 2: (Jz, sigmaz)}

    H = nk.operator.LocalOperator(hi, dtype=complex)
    for color, bond in zip(colores, enlaces):
        if color not in couplings:
            raise ValueError(f"Unsupported bond color {color!r}; expected 0, 1 or 2.")
        J, sigma = couplings[color]
        H -= J * sigma(hi, bond[0]) @ sigma(hi, bond[1])
        H -= h * (
            sigmax(hi, bond[0]) + sigmay(hi, bond[0]) + sigmaz(hi, bond[0])
        )

    return H
