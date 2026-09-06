import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import netket as nk

from .hamiltonian import build_kitaev_lattice, KitaevTransverse_H
from .symmetries import get_kitaev_symmetries, get_projection_group


_PERM_SPARSE_CACHE: Dict[tuple, Any] = {}


def permutation_matrices(hi, sg):
    """Sparse matrices of the group elements `sg` acting on `hi`, memoized.

    Building one is O(dim(hi)) and the same handful of group elements gets
    reused for every eigenvector of every manifold at every Jz (9 elements
    for the 3x3 translation group, 18 for the space group), so without a
    cache a Jz scan rebuilds the same 262144x262144 permutation matrices
    thousands of times. Keyed on the Hilbert space and the permutation's
    index array, so two different lattices never share an entry.
    """
    site_perms = np.asarray(sg.to_array()) if hasattr(sg, "to_array") else None

    matrices = []
    for i, g in enumerate(sg):
        if site_perms is not None:
            key_perm = site_perms[i]
        else:
            key_perm = np.asarray(g.permutation_array)
        key = (hi, tuple(np.asarray(key_perm).ravel().tolist()))
        mat = _PERM_SPARSE_CACHE.get(key)
        if mat is None:
            mat = nk.operator.permutation.PermutationOperator(hi, g).to_sparse()
            _PERM_SPARSE_CACHE[key] = mat
        matrices.append(mat)
    return matrices


def identify_irreps(eigvec, hi, sg, character_table) -> Dict[int, float]:
    """Decompose `eigvec` into the irreps of a symmetry group.

    :param eigvec: eigenvector of the Hamiltonian
    :param hi: NetKet Hilbert space
    :param sg: iterable of permutations (symmetry group elements)
    :param character_table: character table matching `sg`, shape (n_irreps, n_g)
    :return: {irrep_index: weight} for every irrep in the character table
    """
    n_g = len(sg)
    n_irreps = character_table.shape[0]

    expect_vals = np.array([
        np.vdot(eigvec, mat @ eigvec) for mat in permutation_matrices(hi, sg)
    ])

    weights = {}
    for i in range(n_irreps):
        d_mu = np.real(character_table[i, 0])
        weight = (d_mu / n_g) * np.sum(np.conj(character_table[i, :]) * expect_vals)
        weights[i] = float(np.real(weight))

    return weights


def dominant_irrep(irrep_contributions: Dict[int, float]) -> int:
    """Index of the irrep with the largest weight."""
    return max(irrep_contributions, key=irrep_contributions.get)


def _convert_for_json(obj: Any) -> Any:
    """Recursively convert NumPy/complex values to plain JSON-safe types."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, complex):
        return {"re": float(obj.real), "im": float(obj.imag)}
    if isinstance(obj, dict):
        return {str(k): _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_convert_for_json(x) for x in obj]
    return obj


def run_exact_diagonalization(
    extent=(3, 3),
    jz_steps: int = 11,
    k_eigenvals: int = 1,
    save_path: str | None = None,
    save_debug_json: bool = True,
    sector_group: str = "translation",
) -> Dict[float, Dict[str, Any]]:
    """Lanczos-diagonalize the Kitaev Hamiltonian over a Jz scan and record,
    for each Jz, the ground-state energy, low-lying spectrum, eigenvectors,
    and the irrep decomposition of the ground state.

    :param extent: lattice extent passed to `build_kitaev_lattice`
    :param jz_steps: number of Jz points in `np.linspace(0, 1, jz_steps)`
    :param k_eigenvals: number of low-lying eigenpairs to compute per Jz
    :param save_path: where to save the full (pickle-backed) results; defaults
        to `data/raw/energies_eigenvecs_dict_{k_eigenvals}.npz`. A companion
        `.json` with only energies/irrep weights is saved alongside it when
        `save_debug_json` is True
    :param sector_group: which group's irreps to label sectors with, see
        `src.physics.symmetries.get_projection_group` ("translation", the
        default, or "space")
    """
    if save_path is None:
        save_path = f"data/raw/energies_eigenvecs_dict_{k_eigenvals}.npz"

    graph, hilbert = build_kitaev_lattice(extent=extent, pbc=True)
    symmetries = get_kitaev_symmetries(graph, hilbert)
    sector_sg, sector_character_table = get_projection_group(symmetries, sector_group)
    jz_values = np.linspace(0, 1, jz_steps)

    exact_results: Dict[float, Dict[str, Any]] = {}
    json_debug_results: Dict[float, Dict[str, Any]] = {}

    for jz in jz_values:
        jx = jy = (1.0 - jz) / 2.0
        H = KitaevTransverse_H(
            graph.edge_colors, graph.edges(), Jx=jx, Jy=jy, Jz=jz, h=0, hi=hilbert
        )
        eigenvals, eigenvecs = nk.exact.lanczos_ed(
            H, k=k_eigenvals, compute_eigenvectors=True
        )
        order = np.argsort(eigenvals.real)
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]

        irrep_contributions = identify_irreps(
            eigenvecs[:, 0], hilbert, sector_sg, sector_character_table
        )

        deg_idx = degenerate_manifold(eigenvals.real)
        mani_irrep_weights = manifold_irrep_weights(eigenvecs, deg_idx, hilbert, sector_sg, sector_character_table)
        host_sectors = sectors_hosting_manifold(mani_irrep_weights)


        jz_key = round(float(jz), 4)
        exact_results[jz_key] = {
            "E0": float(eigenvals[0].real),
            "energies": eigenvals.real,
            "eigenvectors": eigenvecs,
            "irrep_contributions": irrep_contributions,
            "degenerate_idx": deg_idx,
            "manifold_irrep_weights": mani_irrep_weights,
            "hosting_sectors": host_sectors,
        }
        json_debug_results[jz_key] = {
            "E0": float(eigenvals[0].real),
            "irrep_contributions": irrep_contributions,
            "degenerate_idx": deg_idx,
            "manifold_irrep_weights": mani_irrep_weights,
            "hosting_sectors": host_sectors,
        }

        print(f"[ED] Jz = {jz_key:.4f} done -> E0 = {float(eigenvals[0].real):.6f}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, data_dict=exact_results)
        print(f"[SUCCESS] Full results saved to: {save_path}")

        if save_debug_json:
            json_path = save_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(_convert_for_json(json_debug_results), f, indent=2, ensure_ascii=False)
            print(f"[DEBUG] Energies/irreps summary saved to: {json_path}")

    return exact_results


def load_exact_results(
    file_path: str = "data/raw/energies_eigenvecs_dict.npz",
) -> Dict[float, Dict[str, Any]]:
    """Load the dictionary produced by `run_exact_diagonalization`."""
    npz_file = np.load(file_path, allow_pickle=True)
    return npz_file["data_dict"].item()


def degenerate_manifold(eigenvals, tol=1e-6):
    '''
    Detecta el numero de niveles energéticos degenerados con una pequeña tolerancia.
    devuelve el indice en la matriz de autovalores de estados del subespacio
    degnerado del gs.
    '''
    e0 = eigenvals[0]
    return [i for i, e in enumerate(eigenvals) if abs(e - e0) < tol]

def manifold_irrep_weights(eigvecs, manifold_idx, hi, sg, character_table):
    '''
    Sumamos las contribuciones de los pesos de todos los vectores pertenecientes al
    subespacio degenerado, devolvemos la contribución en peso de la suma de todos
    los eigenvecs.
    '''
    n_irreps = character_table.shape[0]
    totals = {i: 0.0 for i in range(n_irreps)}
    for idx in manifold_idx:
        for k, w in identify_irreps(eigvecs[:, idx], hi, sg, character_table).items():
            totals[k] += w
    return totals

def sectors_hosting_manifold(manifold_weights, tol=0.5):
    '''
    Ordenamos los pesos en orden mayores a 0.5.
    '''
    return sorted(k for k, w in manifold_weights.items() if w > tol)

def manifold_energy_gaps(eigenvals, manifold_idx):
    """Diferencias de energía (respecto a E0) de cada estado del manifold.
    Ayuda a distinguir degeneración real (gaps ~ precisión de máquina, sin
    saltos) de una tolerancia demasiado floja (núcleo apretado + cola cerca
    del límite de `tol`)."""
    e0 = eigenvals.real[0]
    return [float(eigenvals.real[i] - e0) for i in manifold_idx]


def detect_manifold_tail(gaps, jump_ratio=100.0):
    """Busca el mayor salto relativo entre gaps consecutivos (ordenados),
    ignorando el gap trivial de índice 0 (=0.0, siempre presente). Si el
    salto más grande supera `jump_ratio`, devuelve
    (posición, tamaño_del_salto, gap_antes, gap_despues) — señal de que hay
    un núcleo degenerado + una cola de casi-degenerados colada por la
    tolerancia. Si no hay salto grande, devuelve None (bloque homogéneo)."""
    non_trivial = sorted(g for g in gaps if g > 0)
    if len(non_trivial) < 2:
        return None
    best = None
    for i in range(1, len(non_trivial)):
        prev, curr = non_trivial[i - 1], non_trivial[i]
        ratio = curr / prev
        if best is None or ratio > best[1]:
            best = (i, ratio, prev, curr)
    return best if best[1] > jump_ratio else None

def include_sector_analysis(npz_path, hi, sg, character_table, save_path=None):
    '''
    Añade a cada entrada del diccionario cargado desde `npz_path` el análisis
    del manifold degenerado del estado fundamental (índices degenerados, pesos
    de irrep sumados sobre el manifold y sectores que lo hospedan), y guarda el
    resultado en disco (sobrescribiendo `npz_path` salvo que se indique `save_path`).
    '''
    npz = load_exact_results(file_path=npz_path)

    for entry in npz.values():
        eigvals = entry['energies']
        eigvecs = entry['eigenvectors']

        order = np.argsort(eigvals.real)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        deg_idx = degenerate_manifold(eigvals.real)
        gaps = manifold_energy_gaps(eigvals, deg_idx)
        tail = detect_manifold_tail(gaps)

        entry['manifold_gaps'] = gaps
        entry['manifold_tail_warning'] = tail is not None

        print(f"  gaps = {['%.2e' % g for g in gaps]}")
        if tail is not None:
            pos, ratio, prev, curr = tail
            print(f"  [WARN] salto de x{ratio:.1f} en el gap #{pos}: {prev:.2e} -> {curr:.2e} "
                f"— posible cola de casi-degenerados colada por la tolerancia")
        mani_irrep_weights = manifold_irrep_weights(eigvecs, deg_idx, hi, sg, character_table)
        host_sectors = sectors_hosting_manifold(mani_irrep_weights)

        entry['degenerate_idx'] = deg_idx
        entry['manifold_irrep_weights'] = mani_irrep_weights
        entry['hosting_sectors'] = host_sectors

    save_path = Path(save_path) if save_path else Path(npz_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, data_dict=npz)
    print(f"[SUCCESS] Sector analysis added and saved to: {save_path}")

    return npz

def project_state_to_irrep(psi, hi, sg, character_table, irrep_index):
    """Aplica P_k al vector (no solo calcula el peso escalar como identify_irreps).
    Extrae el vector proyectado y normaliza.
    Actua sobre todos los vectores de un subespacio degenerado para obtener el vector proyectado
    :param psi: eigenvector of the Hamiltonian
    :param hi: NetKet Hilbert space
    :param sg: iterable of permutations (symmetry group elements)
    :param character_table: character table matching `sg`, shape (n_irreps, n_g)
    :param irrep_index: index of the irrep to project onto
    :return: projected vector and its norm"""

    n_g = len(sg)
    d_mu = character_table[irrep_index, 0].real
    proj = np.zeros_like(psi)
    for mat, chi in zip(permutation_matrices(hi, sg), character_table[irrep_index]):
        proj = proj + np.conj(chi) * (mat @ psi)
    proj = proj * (d_mu / n_g)
    norm = np.linalg.norm(proj)
    return proj / norm, norm

def pick_source_for_sector(eigvecs, manifold_idx, hi, sg, character_table, irrep_index, min_norm=0.1):
    """Prueba autovectores del manifold hasta encontrar uno con proyección
    suficientemente grande sobre `irrep_index`; si ninguno la tiene, devuelve
    el mejor encontrado (con un warning implícito por la norma baja).

    OJO — esto sólo controla el sector *espacial* (irrep). No dice nada sobre
    el sector de *flujo*: si el manifold aloja más de un patrón de vórtices,
    los `eigvecs[:, idx]` crudos son superposiciones arbitrarias de esos
    patrones y el estado proyectado no es autoestado de ningún `W_p`. Para
    construir un target que sea autoestado simultáneo de H, de los `W_p` y
    del proyector de irrep, usa `pick_target_for_sector`, que resuelve
    primero los sectores de flujo y llama a esta función dentro de uno de
    ellos. Ver el bloque "Vortex (plaquette flux) sector" más abajo."""
    best = None
    for idx in manifold_idx:
        g_k, norm = project_state_to_irrep(eigvecs[:, idx], hi, sg, character_table, irrep_index)
        if best is None or norm > best[1]:
            best = (g_k, norm)
        if norm > min_norm:
            return best
    return best

def manifold_fidelity(eigvecs, manifold_idx, psi):
    """Rotation-invariant fidelity of `psi` against the full ground-state
    manifold: F = <psi|P|psi>, with P the projector onto the whole
    degenerate eigenspace. Unlike the overlap with a single (possibly
    arbitrarily rotated) eigenvector, it does not change if the solver
    returns a different basis of the manifold.

    The obvious `sum_i |<g_i|psi>|^2` over the stored eigenvectors is NOT
    that projector unless the `g_i` are **orthonormal**, and across a
    degenerate level they are not: Lanczos normalizes each vector but loses
    orthogonality between them. Measured on `energies_eigenvecs_dict_k40`,
    |<g_i|g_j>| for i != j reaches 0.108 (Jz=0.5), 0.068 (Jz=0.8) and 0.038
    (Jz=0.6) while every norm is 1 to 5e-13. The naive sum then misses the
    projector by ~1e-3 **in either direction** -- enough to report
    F_manifold < F_sector, which is impossible since the phase-2 target
    lies inside the manifold. Applying it to the target itself, which is
    100% inside by construction, returned 0.9968 (Jz=0.6) and 1.0049
    (Jz=0.5) instead of 1. See docs/DEVLOG.md 2026-08-29.
    """
    idx = list(manifold_idx)
    if not idx:
        return 0.0
    basis = _orthonormal_manifold_basis(eigvecs, idx)
    return float(np.sum(np.abs(basis.conj().T @ np.asarray(psi)) ** 2))


def _orthonormal_manifold_basis(eigvecs, manifold_idx):
    """Orthonormal basis of the manifold as a (dim, m) column matrix.

    Lanczos normalizes each column but does NOT keep them mutually
    orthogonal across a degenerate level -- |<g_i|g_j>| reaches 0.108 on the
    stored 3x3 spectra -- so this is a real re-orthonormalization and not a
    touch-up to the solver's tolerance. It costs O(dim*m^2) and makes the
    restricted operator below exactly Hermitian instead of
    Hermitian-to-1e-8, which matters because the +-1 purity check is
    measured at 1e-6.
    """
    V = np.asarray(eigvecs)[:, list(manifold_idx)]
    Q, _ = np.linalg.qr(V)
    return Q


def restrict_operator(op_sparse, basis):
    """Matrix of `op_sparse` restricted to the span of `basis`.

    :param op_sparse: sparse operator on the full Hilbert space
    :param basis: (dim, m) matrix with orthonormal columns
    :return: the (m, m) matrix ``basis^dag @ op @ basis``
    """
    return basis.conj().T @ (op_sparse @ basis)


def diagonalize_wp_in_manifold(
    eigvecs,
    manifold_idx,
    wilson_loops,
    seed: int = 0,
    purity_tol: float = 1e-6,
):
    """Find the basis of simultaneous eigenstates of {H, W_p_1, ..., W_p_n}
    inside a degenerate manifold, and label each one by its flux pattern.

    The W_p commute with each other and with H, so the manifold can always be
    resolved into common eigenvectors. Rather than diagonalizing the W_p one
    after another (which needs bookkeeping of nested degenerate blocks), this
    diagonalizes a single random Hermitian combination
    ``M = sum_i c_i W_p_i`` restricted to the manifold: for generic real
    weights `c_i` two different sign patterns give different eigenvalues of
    M, so one `eigh` separates all of them at once. The weights come from a
    seeded generator, so the routine is reproducible.

    Vectors sharing a flux pattern stay degenerate under M and come back as
    an arbitrary basis of their common block. That is not a failure: they
    have the same <W_p_i> regardless of how the block is rotated, which is
    exactly the statement that the pattern -- not the individual vector --
    is the physical label.

    :param eigvecs: (dim, n_states) eigenvector matrix, energy-ordered
    :param manifold_idx: column indices of the degenerate manifold
    :param wilson_loops: plaquette operators from `build_wilson_loops`;
        accepts NetKet operators or already-sparse matrices
    :param seed: seed for the random weights `c_i`
    :param purity_tol: how far |<W_p_i>| may sit from 1 before a vector is
        flagged as impure
    :return: dict with

        - ``vectors``: (dim, m) matrix of simultaneous eigenvectors
        - ``sign_patterns``: list of m tuples of +-1, one entry per plaquette
        - ``wp_expectations``: (m, n_plaq) array of the raw <W_p_i>, before
          rounding to +-1 -- the evidence behind each pattern
        - ``max_impurity``: the largest ||<W_p_i>| - 1| over all vectors
        - ``pure``: whether that stayed within `purity_tol`. False means the
          random weights failed to separate two patterns; retry with another
          seed (`diagonalize_wp_in_manifold_robust` does that for you)
        - ``combination_weights``: the `c_i` actually used

    :raises ValueError: if `manifold_idx` is empty
    """
    manifold_idx = list(manifold_idx)
    if not manifold_idx:
        raise ValueError("manifold_idx is empty: nothing to diagonalize.")

    wp_sparse = [w if hasattr(w, "nnz") else w.to_sparse() for w in wilson_loops]
    basis = _orthonormal_manifold_basis(eigvecs, manifold_idx)

    rng = np.random.default_rng(seed)
    weights = rng.normal(size=len(wp_sparse))

    wp_restricted = [restrict_operator(W, basis) for W in wp_sparse]
    M = sum(c * Wr for c, Wr in zip(weights, wp_restricted))
    M = 0.5 * (M + M.conj().T)

    _, U = np.linalg.eigh(M)
    vectors = basis @ U

    wp_expectations = np.array(
        [np.real(np.einsum("ji,jk,ki->i", U.conj(), Wr, U)) for Wr in wp_restricted]
    ).T

    max_impurity = float(np.max(np.abs(np.abs(wp_expectations) - 1.0)))
    sign_patterns = [tuple(1 if v >= 0 else -1 for v in row) for row in wp_expectations]

    return {
        "vectors": vectors,
        "sign_patterns": sign_patterns,
        "wp_expectations": wp_expectations,
        "max_impurity": max_impurity,
        "pure": max_impurity < purity_tol,
        "combination_weights": weights,
    }


def diagonalize_wp_in_manifold_robust(
    eigvecs, manifold_idx, wilson_loops, seeds=(0, 1, 2), purity_tol: float = 1e-6
):
    """`diagonalize_wp_in_manifold` retried over several seeds.

    A random combination fails to separate two sign patterns only when the
    weights happen to give them the same eigenvalue -- a measure-zero
    coincidence, but a cheap one to rule out. Returns the first pure result,
    with the seed that produced it under key ``seed``. If no seed is pure,
    returns the purest attempt with `pure` still False, so the caller reports
    the failure instead of acting on a basis that is not made of W_p
    eigenstates.
    """
    best = None
    for seed in seeds:
        result = diagonalize_wp_in_manifold(
            eigvecs, manifold_idx, wilson_loops, seed=seed, purity_tol=purity_tol
        )
        result["seed"] = seed
        if result["pure"]:
            return result
        if best is None or result["max_impurity"] < best["max_impurity"]:
            best = result
    return best


def plaquette_permutations(plaquettes, sg):
    """Permutation of plaquette *indices* induced by each site permutation.

    A vortex pattern that is a translated copy of another is the same physics
    seen from a different origin, so counting "distinct patterns" without
    quotienting by the lattice symmetry reports nine patterns where there is
    one. This builds the group action that quotient needs.

    Group elements that do not map plaquettes onto plaquettes are skipped
    rather than raising: where that happens the quotient just uses the
    subgroup that does act.

    Whether the site arrays are the group elements or their inverses does not
    matter here: a group is closed under inversion, so both choices generate
    the same set of induced permutations and hence the same orbits.

    :param plaquettes: list of site-index lists, from `get_kitaev_plaquettes`
    :param sg: a NetKet `PermutationGroup` (anything exposing `to_array()`),
        or a plain iterable of site-permutation arrays
    :return: list of tuples, each a permutation of `range(len(plaquettes))`
    """
    site_perms = sg.to_array() if hasattr(sg, "to_array") else sg

    index_of = {frozenset(p): i for i, p in enumerate(plaquettes)}
    induced = []
    for perm in np.asarray(site_perms):
        perm = np.asarray(perm).ravel()
        mapped = []
        for p in plaquettes:
            target = index_of.get(frozenset(int(perm[s]) for s in p))
            if target is None:
                mapped = None
                break
            mapped.append(target)
        if mapped is not None and len(set(mapped)) == len(plaquettes):
            induced.append(tuple(mapped))
    return induced


def _pattern_orbit(pattern, plaquette_perms):
    """Every image of a sign pattern under the induced plaquette group."""
    return {tuple(pattern[i] for i in perm) for perm in plaquette_perms}


def vortex_pattern_summary(sign_patterns, plaquette_perms=None):
    """Reduce the per-vector flux patterns of a manifold to one statement
    about the level's vortex content.

    :param sign_patterns: the `sign_patterns` list from
        `diagonalize_wp_in_manifold`
    :param plaquette_perms: optional output of `plaquette_permutations`. When
        given, patterns related by a lattice symmetry count as one class,
        represented by the lexicographically smallest image in the orbit.
        When None, patterns are compared literally, so translated copies
        count separately.
    :return: dict with

        - ``n_minus``: number of plaquettes at -1, one entry per class. A
          single-element list is the clean case: the level sits in one
          vortex sector
        - ``n_distinct_patterns``: how many genuinely different classes
        - ``vortex_plaquette_idx``: for each class, the plaquette indices at
          -1 in that class's representative
        - ``representatives``: the canonical pattern of each class
        - ``multiplicity``: how many manifold vectors fall in each class
        - ``all_same_n_minus``: whether every class carries the same vortex
          count. A level can host several *placements* of the same number of
          vortices, which is a much weaker statement than several different
          vortex numbers
        - ``quotiented_by_symmetry``: whether `plaquette_perms` was used
    """
    patterns = [tuple(int(s) for s in p) for p in sign_patterns]

    classes: Dict[tuple, dict] = {}
    for pattern in patterns:
        key = min(_pattern_orbit(pattern, plaquette_perms)) if plaquette_perms else pattern
        entry = classes.setdefault(key, {"count": 0, "members": []})
        entry["count"] += 1
        entry["members"].append(pattern)

    ordered = sorted(classes)
    n_minus = [int(sum(1 for s in key if s < 0)) for key in ordered]

    return {
        "n_minus": n_minus,
        "n_distinct_patterns": len(ordered),
        "vortex_plaquette_idx": [[i for i, s in enumerate(k) if s < 0] for k in ordered],
        "representatives": [list(k) for k in ordered],
        "multiplicity": [classes[k]["count"] for k in ordered],
        "all_same_n_minus": len(set(n_minus)) <= 1,
        "quotiented_by_symmetry": bool(plaquette_perms),
    }


def vortex_resolved_manifold(
    eigvecs, manifold_idx, wilson_loops, plaquette_perms=None, seeds=(0, 1, 2)
):
    """Split a degenerate manifold into its flux (vortex) sectors.

    This is the bridge between the W_p analysis and target selection: it
    diagonalizes the W_p inside the manifold, groups the resulting vectors by
    flux pattern, and hands back each group as a column block ready to be
    projected onto a spatial irrep.

    :param eigvecs: (dim, n_states) eigenvector matrix, energy-ordered
    :param manifold_idx: column indices of the degenerate manifold
    :param wilson_loops: plaquette operators, NetKet or sparse
    :param plaquette_perms: optional `plaquette_permutations` output; merges
        patterns related by a lattice symmetry into one block
    :param seeds: seeds tried for the random W_p combination
    :return: dict with

        - ``blocks``: list of dicts, largest first, each with ``vectors``
          (dim, n_j), ``pattern`` (the canonical +-1 tuple), ``n_minus``,
          ``vortex_plaquette_idx`` and ``size``
        - ``summary``: the `vortex_pattern_summary` of the manifold
        - ``diagonalization``: the raw `diagonalize_wp_in_manifold` result
        - ``ambiguous``: True when more than one block exists, i.e. the level
          genuinely hosts several vortex configurations and no single one of
          them is *the* target
        - ``pure``: whether the W_p basis came out clean; False means the
          manifold is not a complete energy level and none of this is
          trustworthy
    """
    diagonalization = diagonalize_wp_in_manifold_robust(
        eigvecs, manifold_idx, wilson_loops, seeds=seeds
    )
    patterns = diagonalization["sign_patterns"]
    summary = vortex_pattern_summary(patterns, plaquette_perms)

    columns_by_pattern: Dict[tuple, list] = {}
    for column, pattern in enumerate(patterns):
        columns_by_pattern.setdefault(tuple(pattern), []).append(column)

    def class_key(pattern):
        return min(_pattern_orbit(pattern, plaquette_perms)) if plaquette_perms else tuple(pattern)

    vectors = diagonalization["vectors"]
    blocks = []
    for pattern, columns in columns_by_pattern.items():
        blocks.append({
            "pattern": list(pattern),
            "class_key": list(class_key(pattern)),
            "n_minus": int(sum(1 for s in pattern if s < 0)),
            "vortex_plaquette_idx": [i for i, s in enumerate(pattern) if s < 0],
            "vectors": vectors[:, columns],
            "size": len(columns),
        })
    blocks.sort(key=lambda b: (-b["size"], b["pattern"]))

    n_classes = summary["n_distinct_patterns"]
    placements: Dict[tuple, int] = {}
    for block in blocks:
        key = tuple(block["class_key"])
        placements[key] = placements.get(key, 0) + 1

    n_plaquettes = len(patterns[0]) if patterns else 0
    classes = []
    for key in sorted({tuple(b["class_key"]) for b in blocks}):
        members = [b for b in blocks if tuple(b["class_key"]) == key]
        n_minus = members[0]["n_minus"]
        classes.append({
            "class_key": list(key),
            "n_minus": n_minus,
            "total_flux": n_plaquettes - 2 * n_minus,
            "vectors": np.concatenate([b["vectors"] for b in members], axis=1),
            "size": sum(b["size"] for b in members),
            "n_placements": len(members),
            "pattern": list(key) if len(members) == 1 else None,
            "placements": [list(b["pattern"]) for b in members],
        })
    classes.sort(key=lambda c: (-c["size"], c["class_key"]))

    return {
        "blocks": blocks,
        "classes": classes,
        "n_plaquettes": n_plaquettes,
        "summary": summary,
        "diagonalization": diagonalization,
        "ambiguous": n_classes > 1,
        "n_distinct_patterns": n_classes,
        "placements_per_class": placements,
        "pure": diagonalization["pure"],
    }


def pick_target_for_sector(
    eigvecs,
    manifold_idx,
    hi,
    sg,
    character_table,
    irrep_index,
    wilson_loops,
    plaquette_perms=None,
    min_norm=0.1,
    flux_pattern=None,
    n_minus=None,
    class_index=None,
    seeds=(0, 1, 2),
    verbose=True,
):
    """Phase-2 target for one spatial irrep, built inside the level's *actual*
    flux sector rather than an assumed vortex-free one.

    `pick_source_for_sector` projects the raw Lanczos eigenvectors, which is
    correct only as long as every vector in the manifold carries the same
    flux pattern. When the level hosts more than one, a raw eigenvector is an
    arbitrary superposition of flux sectors and so is the state projected out
    of it -- a target that is an eigenstate of H and of the spatial projector
    but of no W_p. This routine resolves the flux sectors first and projects
    within one of them, so the target is a simultaneous eigenstate of H, the
    W_p and the irrep projector.

    On the 3x3 torus the level at Jz >= 0.5 turns out to be a *single*
    non-trivial flux sector (four vortices for Jz in [0.5, 0.7], two for
    Jz in [0.8, 0.9]), so here this mostly changes what gets *reported*: the
    target's vortex content becomes an explicit, recorded label instead of an
    unstated assumption. The correctness argument still matters, because
    nothing guarantees that stays true at other sizes or couplings -- and the
    2x2 lattice at Jz=1 is a level that really does mix 0-, 2- and 4-vortex
    patterns.

    :param wilson_loops: plaquette operators, NetKet or sparse
    :param plaquette_perms: optional `plaquette_permutations` output, so
        translated copies of one pattern are not *counted* as separate
        sectors. Targets are still built from a single literal pattern --
        see `vortex_resolved_manifold`.
    :param flux_pattern: pin the target to the vortex sector containing this
        exact +-1 pattern. None picks the largest sector.
    :param n_minus: pin the target to the sector with this many vortices, a
        looser and usually more useful handle than a full pattern.
    :param class_index: pin the target to the i-th vortex sector, in the order
        `vortex_resolved_manifold` returns them (largest span first, ties
        broken by the canonical pattern, so the order is deterministic across
        runs). This is the handle to use when several sectors carry the *same*
        vortex number and `n_minus` therefore cannot tell them apart -- the
        3x3 manifold at Jz >= 0.48 is two such sectors, mutually orthogonal
        and each hosting every irrep, so (class_index, irrep) is what names a
        target uniquely there.
    :param min_norm: passed to `pick_source_for_sector`
    :param verbose: print the chosen sector and any warnings
    :return: dict with ``vector`` (the normalized target), ``norm`` (its
        projection norm onto `irrep_index`), ``n_minus`` and ``total_flux``
        (the conserved flux labels, ready for
        `run_infidelity_projection(target_n_minus=...)`), ``pattern`` (the
        literal +-1 pattern, or **None** when the sector has several
        symmetry-equivalent placements and no momentum eigenstate in it has a
        definite per-plaquette pattern), ``placements``,
        ``n_equivalent_placements``, ``vortex_plaquette_idx``, ``ambiguous``,
        ``pure``, ``block_size``, ``n_distinct_patterns`` and ``summary``

    :raises ValueError: if `flux_pattern` or `n_minus` is given but no sector
        of this manifold matches it
    """
    resolved = vortex_resolved_manifold(
        eigvecs, manifold_idx, wilson_loops,
        plaquette_perms=plaquette_perms, seeds=seeds,
    )
    classes = resolved["classes"]

    if flux_pattern is not None:
        wanted = [int(v) for v in flux_pattern]
        chosen = next(
            (c for c in classes if wanted in c["placements"] or c["class_key"] == wanted),
            None,
        )
        if chosen is None:
            available = [c["placements"] for c in classes]
            raise ValueError(
                f"No block of this manifold carries flux pattern {wanted}. "
                f"Available placements: {available}"
            )
    elif class_index is not None:
        candidates = (
            classes if n_minus is None
            else [c for c in classes if c["n_minus"] == n_minus]
        )
        if not 0 <= class_index < len(candidates):
            raise ValueError(
                f"class_index={class_index} is out of range: this manifold has "
                f"{len(candidates)} vortex sector(s)"
                + (f" with N_minus={n_minus}" if n_minus is not None else "")
                + f" (N_minus={[c['n_minus'] for c in candidates]})."
            )
        chosen = candidates[class_index]
    elif n_minus is not None:
        chosen = next((c for c in classes if c["n_minus"] == n_minus), None)
        if chosen is None:
            raise ValueError(
                f"No vortex sector with N_minus={n_minus} in this manifold. "
                f"Available: {[c['n_minus'] for c in classes]}"
            )
    else:
        chosen = classes[0]

    n_placements = chosen["n_placements"]

    if verbose:
        if not resolved["pure"]:
            print(
                f"  [WARN] the W_p basis of this manifold is impure "
                f"({resolved['diagonalization']['max_impurity']:.1e}): the level is "
                f"probably truncated (too few eigenpairs stored). The target's "
                f"flux label is not trustworthy."
            )
        if resolved["ambiguous"] and flux_pattern is None and n_minus is None:
            if resolved["summary"]["all_same_n_minus"]:
                print(
                    f"  [OPEN CASE, mild] {resolved['n_distinct_patterns']} "
                    f"symmetry-inequivalent sectors with the same vortex number "
                    f"N_minus={chosen['n_minus']} (multiplicity="
                    f"{resolved['summary']['multiplicity']}). The vortex number is "
                    f"well defined either way, but these sectors are mutually "
                    f"ORTHOGONAL, so a target from the wrong one is unreachable: "
                    f"using class_index={classes.index(chosen)} of "
                    f"{len(classes)}. Pin it with `class_index=` to compare them."
                )
            else:
                print(
                    f"  [OPEN CASE] this level mixes vortex NUMBERS (n_minus="
                    f"{resolved['summary']['n_minus']}, multiplicity="
                    f"{resolved['summary']['multiplicity']}). Defaulting to the largest "
                    f"(N_minus={chosen['n_minus']}) picks one of them, which IS a "
                    f"physics choice, not a default. Pin it with `n_minus=` once the "
                    f"intended sector is decided."
                )
        print(
            f"  target flux sector: N_minus={chosen['n_minus']}, "
            f"<sum W_p>={chosen['total_flux']:+d}/{resolved['n_plaquettes']}, "
            f"{n_placements} symmetry-equivalent placement(s), "
            f"span {chosen['size']}/{len(manifold_idx)}"
        )
        if n_placements > 1:
            print(
                f"  (per-plaquette pattern undefined for a momentum target here; "
                f"the conserved label is the vortex number N_minus={chosen['n_minus']})"
            )

    class_vectors = chosen["vectors"]
    g_k, norm = pick_source_for_sector(
        class_vectors, range(class_vectors.shape[1]), hi, sg, character_table,
        irrep_index, min_norm=min_norm,
    )

    wp_sparse_checked = [w if hasattr(w, "nnz") else w.to_sparse() for w in wilson_loops]
    measured_total = float(
        sum(np.real(np.vdot(g_k, W @ g_k)) for W in wp_sparse_checked)
    )
    if abs(measured_total - chosen["total_flux"]) > 1e-6 * max(1, len(wp_sparse_checked)):
        raise ValueError(
            f"Projecting onto irrep {irrep_index} left the vortex sector: the target "
            f"has <sum W_p> = {measured_total:+.6f} but the sector is "
            f"{chosen['total_flux']:+d} (N_minus={chosen['n_minus']} over "
            f"{len(wp_sparse_checked)} plaquettes).\n"
            f"This means the manifold handed in is not a complete energy level: it "
            f"holds {len(manifold_idx)} of the level's states, so the flux sector is "
            f"missing some of its symmetry-equivalent placements and is not closed "
            f"under the projector.\n"
            f"Fix the input, not this call -- rerun the exact diagonalization with a "
            f"larger k so the level fits strictly inside the stored spectrum "
            f"(`common/analysis/map_vortex_sectors.py` escalates k automatically and will "
            f"report the dimension needed at this Jz)."
        )

    if verbose and norm < min_norm:
        print(
            f"  [WARN] irrep {irrep_index} has projection norm {norm:.4f} < {min_norm} "
            f"inside this flux sector: the sector and the irrep may be incompatible, "
            f"and the target is mostly projector noise."
        )

    return {
        "vector": g_k,
        "norm": float(norm),
        "pattern": chosen["pattern"],
        "placements": chosen["placements"],
        "n_minus": chosen["n_minus"],
        "total_flux": chosen["total_flux"],
        "n_plaquettes": resolved["n_plaquettes"],
        "vortex_plaquette_idx": (
            [i for i, v in enumerate(chosen["pattern"]) if v < 0]
            if chosen["pattern"] is not None
            else None
        ),
        "block_size": chosen["size"],
        "class_index": classes.index(chosen),
        "n_classes": len(classes),
        "n_equivalent_placements": n_placements,
        "ambiguous": resolved["ambiguous"],
        "pure": resolved["pure"],
        "n_distinct_patterns": resolved["n_distinct_patterns"],
        "summary": resolved["summary"],
    }


def flux_pattern_of(psi, wilson_loops, tol=1e-6):
    """Measure the flux pattern of an arbitrary state.

    Used to check that a variational state ended up in the flux sector its
    target was built in. Unlike the manifold routines this makes no purity
    assumption -- a variational state is generally *not* a W_p eigenstate,
    which is exactly what the caller wants to find out.

    :return: (signs, expectations, pure) -- `signs` is the rounded +-1
        pattern, `expectations` the raw <W_p_i>, and `pure` whether every
        one of them is within `tol` of +-1
    """
    wp_sparse = [w if hasattr(w, "nnz") else w.to_sparse() for w in wilson_loops]
    expectations = np.array(
        [float(np.real(np.vdot(psi, W @ psi))) for W in wp_sparse]
    )
    signs = tuple(1 if v >= 0 else -1 for v in expectations)
    pure = bool(np.all(np.abs(np.abs(expectations) - 1.0) < tol))
    return signs, expectations, pure
