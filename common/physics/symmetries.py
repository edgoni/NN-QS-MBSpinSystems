from typing import NamedTuple

import numpy as np
import netket as nk
from netket.utils import HashableArray


class KitaevSymmetries(NamedTuple):
    point_group: nk.utils.group.PointGroup
    space_group: nk.utils.group.PermutationGroup
    canonical_representation: object
    irreps_matrices: object
    character_table: np.ndarray
    translations: nk.utils.group.PermutationGroup
    automorphisms: nk.utils.group.PermutationGroup
    """Raw permutation group (iterable of `Permutation` elements). Use
    `symmetry_projector_inputs` to get a plain (n_g, N) index array instead."""


def get_kitaev_symmetries(graph, hi) -> KitaevSymmetries:
    """Collect the symmetry group data of a Kitaev honeycomb graph.

    :param graph: a `nk.graph.KitaevHoneycomb` graph
    :param hi: the matching NetKet Hilbert space
    """
    canonical_representation = nk.symmetry.canonical_representation(
        hilbert=hi, group=graph.translation_group()
    )
    space_group = graph.space_group()

    return KitaevSymmetries(
        point_group=nk.utils.group.PointGroup(graph.point_group(), ndim=2),
        space_group=space_group,
        canonical_representation=canonical_representation,
        irreps_matrices=space_group.irrep_matrices(),
        character_table=space_group.character_table(),
        translations=graph.translation_group(),
        automorphisms=graph.automorphisms(),
    )


def get_projection_group(symmetries: KitaevSymmetries, group: str = "translation"):
    """Return the `(sg, character_table)` pair for the requested symmetry
    group, so every caller (ED sector detection, ansatz projection) picks
    the group the same way.

    :param symmetries: output of `get_kitaev_symmetries`
    :param group: "translation" (default) — every irrep of this lattice's
        translation group is 1-dimensional, so each irrep index names a
        single, well-defined momentum sector. Or "space" — the full space
        group (translations + point group), which has 2-dimensional irreps
        at 3x3 and up; a character projector onto one of those lands on a
        2D isotypic subspace rather than a single state.
        `docs/proyeccion_simetria_rbm_kitaev.md` found experimentally that
        projecting on the full space group can collapse training onto an
        excited eigenstate instead of the ground state, so "translation" is
        the recommended default — "space" is kept available for comparison
        experiments.
    """
    if group == "translation":
        sg = symmetries.translations
        return sg, sg.character_table()
    elif group == "space":
        return symmetries.automorphisms, symmetries.character_table
    else:
        raise ValueError(f"Unknown group {group!r}, expected 'translation' or 'space'")


def symmetry_projector_inputs(symmetries: KitaevSymmetries, irrep_index: int, group: str = "translation"):
    """Build the (permutations, characters) HashableArrays for a given irrep,
    ready to pass to a symmetry-projected ansatz such as `ProjectedRBM` or
    `QuantumSelfAttention`.

    :param symmetries: output of `get_kitaev_symmetries`
    :param irrep_index: row index into the chosen group's character table
    :param group: which symmetry group to project onto; see `get_projection_group`
    """
    sg, character_table = get_projection_group(symmetries, group)
    perms = HashableArray(np.array(sg))
    characters = HashableArray(np.array(character_table[irrep_index]))
    return perms, characters
