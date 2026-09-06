import numpy as np

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)

SPIN_ROT_120 = np.exp(1j * np.pi / 3) * 0.5 * (_I2 - 1j * (_SX + _SY + _SZ))


def c3_site_permutation(graph, center_site: int = 0) -> np.ndarray:
    """Site permutation of the 120-degree lattice rotation about `center_site`.

    Built from the lattice geometry rather than from `graph.automorphisms()`,
    which cannot contain it: NetKet's automorphisms preserve edge colours and
    a C3 permutes them.

    Each site position is rotated by 120 degrees about the centre and matched
    back to a site modulo the supercell vectors (`extent * basis_vectors`), so
    it respects the periodic boundary conditions.

    :param graph: a `nk.graph.KitaevHoneycomb` built with `pbc=True`
    :param center_site: site to rotate about; any site of the same sublattice
        gives a permutation differing by a translation, and translations are
        already handled by the translation group
    :return: array `perm` of length `n_nodes`, in NetKet's permutation
        convention (verified against `PermutationOperator` in the tests)
    :raises ValueError: if the rotation does not map the lattice onto itself,
        which happens for supercells incompatible with 3-fold rotation
    """
    pos = np.asarray(graph.positions, dtype=float)
    basis = np.asarray(graph.basis_vectors, dtype=float)
    extent = np.asarray(graph.extent, dtype=float)
    supercell = (basis.T * extent).T

    theta = 2.0 * np.pi / 3.0
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    centre = pos[center_site]

    perm = np.full(len(pos), -1, dtype=int)
    for i, p in enumerate(pos):
        target = centre + rot @ (p - centre)
        for j, pj in enumerate(pos):
            frac = np.linalg.solve(supercell.T, target - pj)
            if np.allclose(frac, np.round(frac), atol=1e-8):
                perm[i] = j
                break
        if perm[i] < 0:
            raise ValueError(
                f"the 120-degree rotation sends site {i} outside the lattice; "
                f"extent={list(graph.extent)} is not compatible with C3"
            )
    if sorted(perm.tolist()) != list(range(len(pos))):
        raise ValueError("the C3 site map is not a bijection")
    return perm


def bond_colour_map(graph, perm) -> dict:
    """How `perm` acts on the edge colours: `{colour: set(image colours)}`.

    A correct C3 gives a clean 3-cycle, `{0: {1}, 1: {2}, 2: {0}}`. Anything
    else means the permutation is not the rotation it claims to be.
    """
    colours = np.asarray(graph.edge_colors)
    by_edge = {
        frozenset((int(u), int(v))): int(c)
        for (u, v), c in zip(graph.edges(), colours)
    }
    mapping: dict = {}
    for (u, v), c in zip(graph.edges(), colours):
        image = by_edge[frozenset((int(perm[u]), int(perm[v])))]
        mapping.setdefault(int(c), set()).add(image)
    return mapping


def apply_site_permutation(vec: np.ndarray, perm) -> np.ndarray:
    """Apply a site permutation to a state vector, matching NetKet's
    `PermutationOperator` convention (asserted in the tests).

    Site 0 is the most significant bit of the basis index (checked against
    `nk.operator.spin.sigmaz(hi, 0)` in the tests), so sites are axes of a
    `(2,) * N` tensor and a site permutation is a transpose of those axes.

    The transpose uses `argsort(perm)`, not `perm`: NetKet's convention makes
    site i of the output read site `perm[i]` of the input, which as a tensor
    transpose is the inverse permutation. Both choices are "a" C3 -- they are
    each other's inverse, and the pair (`perm`, U^dagger) commutes with H just
    as well as (`argsort(perm)`, U) -- but mixing one with the wrong spin
    direction gives an operator that does *not* commute, so the convention is
    pinned by `test_apply_site_permutation_matches_netket`.
    """
    perm = np.asarray(perm, dtype=int)
    n = len(perm)
    tensor = np.asarray(vec).reshape((2,) * n)
    return np.transpose(tensor, axes=np.argsort(perm)).reshape(-1)


def apply_onsite_unitary(vec: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Apply `u` (2x2) to every site of a state vector: the action of
    `u^{tensor N}`, in O(N 2^N) instead of the O(4^N) a dense matrix costs.

    This is the whole reason the commutation test is affordable at 3x3, where
    the dense operator would be 262144 x 262144.
    """
    vec = np.asarray(vec)
    n = int(round(np.log2(vec.size)))
    tensor = vec.reshape((2,) * n)
    for axis in range(n):
        tensor = np.tensordot(u, tensor, axes=([1], [axis]))
        tensor = np.moveaxis(tensor, 0, axis)
    return tensor.reshape(-1)


def apply_combined_c3(vec: np.ndarray, perm, dagger: bool = False) -> np.ndarray:
    """Apply the combined lattice+spin C3 to a state vector.

    The two factors commute (the same 2x2 acts on every site, so relabelling
    sites does not change it), so the order is immaterial; the tests assert
    that too.

    :param dagger: apply the inverse rotation, i.e. the other direction of the
        3-cycle (sigma^x -> sigma^z -> sigma^y)
    """
    u = SPIN_ROT_120.conj().T if dagger else SPIN_ROT_120
    return apply_site_permutation(apply_onsite_unitary(vec, u), perm)


def combined_c3_dense(graph, dagger: bool = False) -> np.ndarray:
    """Dense matrix of the combined C3, by applying it to each basis vector.

    Only for small lattices: this is 2^N x 2^N. At 2x2 (N=8) it is a 256x256
    matrix; at 3x3 (N=18) it would be 262144x262144, so use
    `apply_combined_c3` there instead.
    """
    n = graph.n_nodes
    dim = 2 ** n
    perm = c3_site_permutation(graph)
    out = np.empty((dim, dim), dtype=complex)
    basis = np.zeros(dim, dtype=complex)
    for col in range(dim):
        basis[:] = 0.0
        basis[col] = 1.0
        out[:, col] = apply_combined_c3(basis, perm, dagger=dagger)
    return out


DEFAULT_SPIN_AXIS = (1.0, 1.0, 1.0)


def spin_basis_rotation(axis=DEFAULT_SPIN_AXIS) -> np.ndarray:
    """2x2 unitary V with `V^dagger (n.sigma) V = sigma^z`, n = axis/|axis|.

    Built explicitly rather than by `np.linalg.eig` on `SPIN_ROT_120`: an
    eigen-decomposition fixes the eigenvectors only up to phase and ordering,
    so it would give a different (though equivalent) frame on every call and
    on every numpy version, which is a poor thing to hang saved checkpoints on.

    The dagger convention matters and is easy to get backwards. Everything
    downstream conjugates as `W^dagger X W` with `W = V^{tensor N}` -- the
    rotated state is `W^dagger psi`, the rotated Hamiltonian `W^dagger H W`,
    the rotated symmetry `W^dagger R W = P (V^dagger U V)^{tensor N}`. So the
    factor that has to come out diagonal is `V^dagger U V`, which is why V is
    the rotation carrying n *onto* z in that direction and not its inverse.
    Choosing the opposite one leaves the Hamiltonian perfectly valid (any
    unitary conjugation preserves the spectrum) while silently leaving the
    symmetry operator dense -- measured: 256 connections per configuration
    instead of 1, at 2x2.

    :param axis: which spin axis to carry onto z. `(1,1,1)` (the default) is
        the C3's rotation axis; `(1,1,0)` is the xy mirror's, needed by the
        C2v path. Each symmetry needs the frame that diagonalises ITS OWN
        spin factor -- the C3 frame does not make `SPIN_ROT_XY_SWAP`
        diagonal and vice versa, so a run projecting onto one group must use
        that group's axis throughout (Hamiltonian included).
    """
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(n, z)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(n @ z, -1.0, 1.0))
    pauli = axis[0] * _SX + axis[1] * _SY + axis[2] * _SZ
    rotation = np.cos(angle / 2.0) * _I2 - 1j * np.sin(angle / 2.0) * pauli
    return rotation.conj().T


def pauli_rotation_matrix(axis=DEFAULT_SPIN_AXIS) -> np.ndarray:
    """Real 3x3 matrix `m` with `V^dagger sigma^a V = sum_b m[a, b] sigma^b`.

    This is what turns a Kitaev bond term into its rotated-frame form: the
    single term `sigma^c_i sigma^c_j` becomes the nine-term product
    `(sum_a m[c,a] sigma^a_i)(sum_b m[c,b] sigma^b_j)`.

    :param axis: passed through to `spin_basis_rotation`.
    """
    v = spin_basis_rotation(axis)
    paulis = (_SX, _SY, _SZ)
    m = np.empty((3, 3), dtype=float)
    for a, pa in enumerate(paulis):
        rotated = v.conj().T @ pa @ v
        for b, pb in enumerate(paulis):
            coeff = np.trace(pb.conj().T @ rotated) / 2.0
            assert abs(coeff.imag) < 1e-12, "rotated Pauli is not Hermitian"
            m[a, b] = coeff.real
    return m


def rotated_diagonal_phase(axis=DEFAULT_SPIN_AXIS, unitary=None) -> np.ndarray:
    """The diagonal `V^dagger U V` of a spin rotation in its own frame.

    Defaults reproduce the C3: `diag(1, omega)` with omega = exp(2 pi i / 3).
    With `axis=(1,1,0), unitary=SPIN_ROT_XY_SWAP` it gives `diag(1, -1)`,
    the Z2 case (measured).

    :param axis: spin axis of the frame, see `spin_basis_rotation`
    :param unitary: the 2x2 spin rotation to diagonalise; defaults to
        `SPIN_ROT_120`. It must be the rotation about `axis`, otherwise the
        result is simply not diagonal -- which is exactly the failure mode
        this function exists to let the tests catch.
    """
    v = spin_basis_rotation(axis)
    u = SPIN_ROT_120 if unitary is None else unitary
    return v.conj().T @ u @ v


def rotated_kitaev_hamiltonian(colours, edges, Jx, Jy, Jz, hi, axis=DEFAULT_SPIN_AXIS):
    """`KitaevTransverse_H` expressed in the rotated spin frame.

    Same operator, same spectrum, different basis: this is
    `V^dagger^{tensor N} H V^{tensor N}` written directly as a NetKet
    `LocalOperator` rather than built as a dense matrix, so it works at 3x3.

    Sign convention follows `src.physics.hamiltonian.KitaevTransverse_H`,
    which *subtracts* each bond term.

    :param axis: the spin frame to express H in. MUST match the axis of the
        symmetry group being projected onto -- `(1,1,1)` for the C3,
        `(1,1,0)` for the xy mirror / C2v. Mixing them leaves H perfectly
        valid (any unitary conjugation preserves the spectrum) while the
        symmetry operator stays dense, which is a silent performance cliff
        rather than a crash.
    """
    import netket as nk
    from netket.operator.spin import sigmax, sigmay, sigmaz

    m = pauli_rotation_matrix(axis)
    spin_ops = (sigmax, sigmay, sigmaz)
    couplings = (Jx, Jy, Jz)

    h = nk.operator.LocalOperator(hi, dtype=complex)
    for edge, colour in zip(edges, colours):
        colour = int(colour)
        if colour not in (0, 1, 2):
            raise ValueError(f"unexpected edge colour {colour}")
        i, j = int(edge[0]), int(edge[1])
        rotated_i = sum(
            m[colour, b] * spin_ops[b](hi, i) for b in range(3) if abs(m[colour, b]) > 1e-14
        )
        rotated_j = sum(
            m[colour, b] * spin_ops[b](hi, j) for b in range(3) if abs(m[colour, b]) > 1e-14
        )
        h -= couplings[colour] * (rotated_i @ rotated_j)
    return h


def c3_translation_group(graph):
    """The group generated by the lattice translations and the combined C3.

    At 3x3 this is 9 translations x 3 rotations = 27 elements, against the 18
    of `graph.space_group()` (9 translations x a C2). It is only a symmetry
    group of the *isotropic* Hamiltonian.

    In the rotated spin frame every element acts monomially: permute the
    sites, then multiply by a phase `omega**(k * n_minus)` where k is the C3
    power and `n_minus` counts the sites in the second local state. Because
    that count is permutation-invariant, the phase depends on the
    configuration and on k alone -- not on which translation -- which is what
    lets a SymmExpSum-style sum absorb it into an effective character.

    :return: `(perms, powers)` with `perms` of shape (|G|, N) in NetKet's
        permutation convention and `powers[g]` the C3 power of element g
    """
    translations = np.asarray(graph.translation_group())
    c3 = c3_site_permutation(graph)
    identity = np.arange(len(c3))
    c3_powers = [identity, c3, c3[c3]]

    perms, powers = [], []
    for k, rotation in enumerate(c3_powers):
        for t in translations:
            perms.append(rotation[t])
            powers.append(k)

    perms = np.asarray(perms, dtype=int)
    powers = np.asarray(powers, dtype=int)

    unique = {tuple(p) for p in perms.tolist()}
    if len(unique) != len(perms):
        raise ValueError(
            f"the C3/translation product is degenerate: {len(perms)} elements "
            f"collapse to {len(unique)} distinct permutations"
        )
    return perms, powers


def c3_group_is_closed(perms) -> bool:
    """Whether `perms` is closed under composition -- i.e. really a group.

    Checked numerically rather than argued: the semidirect product only closes
    if the C3 maps translations to translations, which is a property of the
    supercell, not something to assume.
    """
    table = {tuple(p) for p in np.asarray(perms).tolist()}
    for p in np.asarray(perms):
        for q in np.asarray(perms):
            if tuple(p[q].tolist()) not in table:
                return False
    return True


def c3_character_table(graph, perms=None):
    """Character table of `c3_translation_group`, via NetKet.

    The map from group element to site permutation is injective (a C3 is not
    a translation), so the abstract group is isomorphic to this permutation
    group and NetKet can compute the table directly.
    """
    import netket as nk
    from netket.utils.group import Permutation, PermutationGroup

    if perms is None:
        perms, _ = c3_translation_group(graph)
    elements = [Permutation(permutation_array=np.asarray(p)) for p in perms]
    group = PermutationGroup(elements, degree=int(np.asarray(perms).shape[1]))
    return group, group.character_table()


def c3_configuration_phase(sigma, power: int) -> np.ndarray:
    """The monomial phase `omega**(power * n_minus(sigma))` in the rotated
    frame, for spin configurations in {+1, -1}.

    `n_minus` counts sites in the second local state, which for NetKet's
    `Spin` Hilbert space (`local_states == [1, -1]`) means sigma == -1.
    """
    sigma = np.asarray(sigma)
    n_minus = np.sum(sigma < 0, axis=-1)
    omega = np.exp(2j * np.pi / 3.0)
    return omega ** ((power * n_minus) % 3)


def rotate_state_to_frame(psi, inverse: bool = False, axis=DEFAULT_SPIN_AXIS) -> np.ndarray:
    """Move a state vector between the computational and the rotated frame.

    Forward (`inverse=False`) is `W^dagger psi` with `W = V^{tensor N}`: takes
    a state written in the computational basis into the rotated frame the C3
    is monomial in. Useful for comparing against ED eigenvectors, plaquette
    expectation values, or checkpoints produced in the other frame.

    :param axis: the spin frame, see `spin_basis_rotation`.
    """
    v = spin_basis_rotation(axis)
    return apply_onsite_unitary(psi, v if inverse else v.conj().T)


def c3_irrep_weights(psi, perms, powers, character_table) -> dict:
    """Decompose a rotated-frame state into the irreps of the C3 group.

    The analogue of `src.physics.exact_diag.identify_irreps` for this group.
    It cannot reuse that one: `identify_irreps` builds `PermutationOperator`
    matrices, and these group elements are permutations *times a phase*, so
    the permutational version would compute the wrong expectation values.

    Everything is matrix-free -- one permutation and one elementwise multiply
    per group element, O(|G| 2^N) in total -- so it runs at 3x3 where the
    27 dense operators never could.

    :param psi: state vector in the ROTATED frame (see `rotate_state_to_frame`)
    :return: `{irrep_index: weight}`, weights summing to 1 for a normalised psi
    """
    psi = np.asarray(psi)
    n_sites = int(round(np.log2(psi.size)))
    omega = np.exp(2j * np.pi / 3.0)

    bits = ((np.arange(psi.size)[:, None] >> np.arange(n_sites)[::-1]) & 1)
    n_minus = bits.sum(axis=1)
    phases = [omega ** ((k * n_minus) % 3) for k in range(3)]

    expectations = []
    for perm, power in zip(np.asarray(perms), np.asarray(powers)):
        rotated = apply_site_permutation(phases[int(power) % 3] * psi, perm)
        expectations.append(np.vdot(psi, rotated))
    expectations = np.asarray(expectations)

    weights = {}
    n_g = len(expectations)
    for mu in range(character_table.shape[0]):
        d_mu = float(np.real(character_table[mu, 0]))
        weight = (d_mu / n_g) * np.sum(
            np.conj(character_table[mu, :]) * expectations
        )
        weights[mu] = float(np.real(weight))
    return weights


SPIN_ROT_XY_SWAP = (_SX + _SY) / np.sqrt(2.0)


def c2_xy_site_permutation(graph, center_site: int = 0) -> np.ndarray:
    """Site permutation of the mirror that fixes z-bonds and swaps x with y.

    Built from the lattice geometry for the same reason as
    `c3_site_permutation`: `graph.automorphisms()` cannot contain it, because
    NetKet requires automorphisms to preserve edge colours and this one
    exchanges two of them.

    The mirror is about the vertical axis, `diag(-1, 1)`, because on this graph
    the z-bonds (colour 2) run vertically while colours 0 and 1 sit at +30 and
    -30 degrees; reflecting about the vertical therefore fixes the first and
    exchanges the other two. Bond directions are unoriented, which is why +30
    maps to -30 rather than to 150.

    :param graph: a `nk.graph.KitaevHoneycomb` built with `pbc=True`
    :param center_site: site the mirror axis passes through. Any site works
        (measured: all 8 at 2x2 and all 18 at 3x3 give a valid order-2
        permutation with the right colour map); different choices differ by a
        translation, and translations are handled by the translation group.
    :return: array `perm` of length `n_nodes`, in NetKet's permutation
        convention -- the same one `apply_site_permutation` documents
    :raises ValueError: if the mirror does not map the lattice onto itself
    """
    pos = np.asarray(graph.positions, dtype=float)
    basis = np.asarray(graph.basis_vectors, dtype=float)
    extent = np.asarray(graph.extent, dtype=float)
    supercell = (basis.T * extent).T

    mirror = np.array([[-1.0, 0.0], [0.0, 1.0]])
    centre = pos[center_site]

    perm = np.full(len(pos), -1, dtype=int)
    for i, p in enumerate(pos):
        target = centre + mirror @ (p - centre)
        for j, pj in enumerate(pos):
            frac = np.linalg.solve(supercell.T, target - pj)
            if np.allclose(frac, np.round(frac), atol=1e-8):
                perm[i] = j
                break
        if perm[i] < 0:
            raise ValueError(
                f"the xy mirror sends site {i} outside the lattice; "
                f"extent={list(graph.extent)} is not compatible with it"
            )
    if sorted(perm.tolist()) != list(range(len(pos))):
        raise ValueError("the xy mirror site map is not a bijection")
    return perm


def check_c2xy_applicable(Jx, Jy, Jz=None, h=0.0) -> None:
    """Raise unless the Jx=Jy Z2 really is a symmetry of this Hamiltonian.

    Two independent conditions, and the second is the easy one to forget:

    - `Jx == Jy`, because the mirror exchanges x-bonds with y-bonds. `Jz` is
      free -- that is the whole point of this Z2 as against the C3.
    - `h == 0`. The spin half sends sigma^z to *minus* sigma^z (see
      `SPIN_ROT_XY_SWAP`: the sign is forced, no SU(2) element implements the
      bare transposition). Bond terms are bilinear in a single colour so the
      sign cancels, but a field term is linear in sigma^a and it does not.

    :raises ValueError: naming which condition failed
    """
    if not np.isclose(Jx, Jy, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"the xy mirror is only a symmetry when Jx == Jy, got "
            f"Jx={Jx}, Jy={Jy}"
        )
    if not np.isclose(h, 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"the xy mirror is not a symmetry with a transverse field "
            f"(h={h}): its spin half sends sigma^z to -sigma^z, which cancels "
            f"in the bilinear bond terms but not in the field term, which is "
            f"linear in sigma^a"
        )


def apply_combined_c2xy(vec: np.ndarray, perm) -> np.ndarray:
    """Apply the combined mirror+spin Z2 to a state vector.

    No `dagger` argument, unlike `apply_combined_c3`: `SPIN_ROT_XY_SWAP` is
    Hermitian as well as unitary, and the site permutation has order 2, so the
    whole operator is its own inverse.

    As for the C3 the two factors commute -- the same 2x2 acts on every site,
    so relabelling sites cannot change it -- and the tests assert that.
    """
    return apply_site_permutation(
        apply_onsite_unitary(vec, SPIN_ROT_XY_SWAP), perm
    )


def combined_c2xy_dense(graph) -> np.ndarray:
    """Dense matrix of the combined Z2, by applying it to each basis vector.

    2^N x 2^N, so only for small lattices: 256x256 at 2x2. At 3x3 use
    `apply_combined_c2xy` on vectors instead.
    """
    n = graph.n_nodes
    dim = 2 ** n
    perm = c2_xy_site_permutation(graph)
    out = np.empty((dim, dim), dtype=complex)
    basis = np.zeros(dim, dtype=complex)
    for col in range(dim):
        basis[:] = 0.0
        basis[col] = 1.0
        out[:, col] = apply_combined_c2xy(basis, perm)
    return out


def c2xy_translation_group(graph) -> np.ndarray:
    """The group generated by the lattice translations and the xy mirror.

    This exists for a *different* job than everything above it in this
    module: quotienting vortex-pattern equivalence classes
    (`src.physics.exact_diag.plaquette_permutations` +
    `vortex_pattern_summary`), which is purely combinatorial bookkeeping on
    plaquette indices. It needs no spin rotation at all -- unlike projecting
    an ansatz onto this symmetry, which is why this returns plain site
    PERMUTATIONS rather than going through the monomial-frame machinery the
    C3 needed for `c3_translation_group`. No phases, no `powers` array.

    Closes on its own (measured: 18 elements at 3x3, 8 at 2x2) without
    needing `graph.space_group()`'s other order-2 automorphism (the one
    `try_Norman.py`'s `projection_group="space"` uses): the mirror
    normalizes the translation group by itself, so translations + mirror is
    already a subgroup, not merely a generating set that happens to need
    closing with something else.

    Why this matters: `Jx == Jy` holds identically along the whole
    production scan (`jx = jy = (1 - jz) / 2` in both `try_Norman.py` and
    `Supervised_Infid_min/run_vmc.py`), so the xy mirror is an exact symmetry of H at
    every point that scan visits -- not a special case at one Jz. Measured
    on the 3x3 lattice (`data/results/vortex_sectors_by_jz.csv`, computed
    with `--group space`, which does NOT include this mirror): the level
    hosting the ground manifold at `Jz in [0.48, 0.90]` was reported as
    TWO vortex classes of multiplicity 9 each. Quotienting with this group
    instead collapses them into ONE class of size 18 -- the two "classes"
    are mirror images of each other, not physically distinct configurations
    (see `docs/física/simetrias_y_optimizacion.md`, sec. 10.3).

    :param graph: a `nk.graph.KitaevHoneycomb` graph
    :return: array `perms` of shape (2*|translations|, N), in NetKet's
        permutation convention
    :raises ValueError: if the generated set is not closed or degenerate,
        which would mean the mirror does not normalize this lattice's
        translation group -- not expected for any extent this graph
        supports, since `c2_xy_site_permutation` already requires the
        mirror to be a bijection of the same periodic lattice
    """
    translations = np.asarray(graph.translation_group())
    mirror = c2_xy_site_permutation(graph)
    identity = np.arange(len(mirror))

    perms = []
    for reflection in (identity, mirror):
        for t in translations:
            perms.append(reflection[t])
    perms = np.asarray(perms, dtype=int)

    unique = {tuple(p) for p in perms.tolist()}
    if len(unique) != len(perms):
        raise ValueError(
            f"the mirror/translation product is degenerate: {len(perms)} "
            f"elements collapse to {len(unique)} distinct permutations"
        )
    if not c3_group_is_closed(perms):
        raise ValueError(
            "translations + xy mirror do not close into a group for this "
            "graph -- the mirror does not normalize the translation group"
        )
    return perms


def c2_lattice_site_permutation(graph, center_edge: int = 0) -> np.ndarray:
    """Site permutation of the pi rotation of the lattice: the C2 that
    preserves bond colours.

    Unlike `c3_site_permutation` and `c2_xy_site_permutation`, the centre must
    be the MIDPOINT OF A BOND, not a site. Measured: centring on a site sends
    points off the lattice and the map is not a bijection, while all 27 bond
    midpoints at 3x3 (12 at 2x2) give a valid order-2 permutation that
    preserves every colour. That is a property of the honeycomb: its C2
    centres sit on bond midpoints and plaquette centres, not on vertices.

    Being colour-preserving, this one IS among `graph.automorphisms()` and
    needs no spin rotation to be a symmetry -- it commutes with H for ANY
    couplings, isotropic or not. It is included here anyway so the C2v group
    can be built from explicit geometry rather than by fishing an unnamed
    order-2 element out of `automorphisms()` and hoping it is the right one.

    :param graph: a `nk.graph.KitaevHoneycomb` built with `pbc=True`
    :param center_edge: index into `graph.edges()`; the rotation centre is
        that edge's midpoint. Different edges give permutations differing by
        a translation.
    :return: array `perm` of length `n_nodes`, NetKet's convention
    :raises ValueError: if the rotation does not map the lattice onto itself
    """
    pos = np.asarray(graph.positions, dtype=float)
    basis = np.asarray(graph.basis_vectors, dtype=float)
    extent = np.asarray(graph.extent, dtype=float)
    supercell = (basis.T * extent).T

    edges = list(graph.edges())
    u, v = edges[center_edge]
    centre = 0.5 * (pos[int(u)] + pos[int(v)])
    rotation = -np.eye(2)

    perm = np.full(len(pos), -1, dtype=int)
    for i, p in enumerate(pos):
        target = centre + rotation @ (p - centre)
        for j, pj in enumerate(pos):
            frac = np.linalg.solve(supercell.T, target - pj)
            if np.allclose(frac, np.round(frac), atol=1e-8):
                perm[i] = j
                break
        if perm[i] < 0:
            raise ValueError(
                f"the pi rotation about edge {center_edge}'s midpoint sends "
                f"site {i} outside the lattice"
            )
    if sorted(perm.tolist()) != list(range(len(pos))):
        raise ValueError("the lattice C2 site map is not a bijection")
    return perm


def c2v_translation_group(graph):
    """Translations combined with the full C2v point group (lattice C2 + xy
    mirror), with the spin grade of each element.

    At 3x3: 9 translations x 4 point-group cosets = **36 elements**, against
    `space`'s 18 and `c3`'s 27. Only a symmetry group when `Jx == Jy` (and
    `h == 0`), which `check_c2xy_applicable` enforces.

    The grade is derived from `bond_colour_map`, not from how the element was
    built: grade 1 means the element swaps x with y and therefore needs
    `SPIN_ROT_XY_SWAP`; grade 0 means it preserves colours and is a pure
    permutation. Deriving it rather than assigning it makes the construction
    self-checking -- if the geometric composition were wrong, some element
    would come back with a colour map that is neither, and this raises.

    :return: `(perms, grades)` with `perms` of shape (|G|, N) in NetKet's
        permutation convention and `grades[g]` in {0, 1}
    :raises ValueError: if the elements are not distinct, the set is not
        closed, or any element acts on colours as neither the identity nor
        the x<->y swap
    """
    translations = np.asarray(graph.translation_group())
    mirror = c2_xy_site_permutation(graph)
    lattice_c2 = c2_lattice_site_permutation(graph)
    identity = np.arange(graph.n_nodes)

    perms = []
    for reflection in (identity, mirror):
        for rotation in (identity, lattice_c2):
            for t in translations:
                perms.append(reflection[rotation[t]])
    perms = np.asarray(perms, dtype=int)

    unique = {tuple(p) for p in perms.tolist()}
    if len(unique) != len(perms):
        raise ValueError(
            f"the C2v/translation product is degenerate: {len(perms)} "
            f"elements collapse to {len(unique)} distinct permutations"
        )
    if not c3_group_is_closed(perms):
        raise ValueError(
            "translations + C2v do not close into a group for this graph"
        )

    preserving = {0: {0}, 1: {1}, 2: {2}}
    swapping = {0: {1}, 1: {0}, 2: {2}}
    grades = []
    for p in perms:
        colour_map = bond_colour_map(graph, p)
        if colour_map == preserving:
            grades.append(0)
        elif colour_map == swapping:
            grades.append(1)
        else:
            raise ValueError(
                f"a C2v element acts on bond colours as {colour_map}, which is "
                f"neither the identity nor the x<->y swap -- the geometric "
                f"composition is wrong"
            )
    return perms, np.asarray(grades, dtype=int)


def c2v_character_table(graph, perms=None):
    """Character table of `c2v_translation_group`, via NetKet.

    Same trick as `c3_character_table`: the map from group element to site
    permutation is injective (verified by the distinctness check in
    `c2v_translation_group`), so the abstract group is isomorphic to this
    permutation group.

    Measured at 3x3: 9 irreps with dims [1,1,1,1,2,2,2,2,4]. Note only FOUR
    are one-dimensional, against nine of the eleven for the C3 group -- a
    bigger group is not automatically more usable, since a projector onto a
    d>1 irrep lands on a d-dimensional isotypic subspace rather than naming a
    state (fine for energy minimization, not for a supervised target).
    """
    from netket.utils.group import Permutation, PermutationGroup

    if perms is None:
        perms, _ = c2v_translation_group(graph)
    elements = [Permutation(permutation_array=np.asarray(p)) for p in perms]
    group = PermutationGroup(elements, degree=int(np.asarray(perms).shape[1]))
    return group, group.character_table()


def c2v_irrep_weights(psi, perms, grades, character_table) -> dict:
    """Decompose a rotated-frame state into the irreps of the C2v group.

    The `root_order=2` analogue of `c3_irrep_weights`: the monomial phase is
    `(-1)**(grade * n_minus)` instead of `omega**(power * n_minus)`. Same
    reason it cannot reuse `src.physics.exact_diag.identify_irreps` -- that
    one builds pure `PermutationOperator` matrices and these elements carry a
    sign.

    :param psi: state vector in the (1,1,0) ROTATED frame, i.e.
        `rotate_state_to_frame(psi, axis=(1,1,0))`
    :return: `{irrep_index: weight}`, summing to 1 for a normalised psi
    """
    psi = np.asarray(psi)
    n_sites = int(round(np.log2(psi.size)))

    bits = ((np.arange(psi.size)[:, None] >> np.arange(n_sites)[::-1]) & 1)
    n_minus = bits.sum(axis=1)
    phases = [np.ones_like(n_minus, dtype=float), (-1.0) ** n_minus]

    expectations = []
    for perm, grade in zip(np.asarray(perms), np.asarray(grades)):
        rotated = apply_site_permutation(phases[int(grade) % 2] * psi, perm)
        expectations.append(np.vdot(psi, rotated))
    expectations = np.asarray(expectations)

    weights = {}
    n_g = len(expectations)
    for mu in range(character_table.shape[0]):
        d_mu = float(np.real(character_table[mu, 0]))
        weight = (d_mu / n_g) * np.sum(
            np.conj(character_table[mu, :]) * expectations
        )
        weights[mu] = float(np.real(weight))
    return weights
