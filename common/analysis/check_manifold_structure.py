#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np
import netket as nk

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from common.physics.hamiltonian import build_kitaev_lattice, KitaevTransverse_H
from common.physics.observables import get_kitaev_plaquettes, build_wilson_loops
from common.physics.symmetries import get_kitaev_symmetries, get_projection_group
from common.physics.exact_diag import (
    load_exact_results,
    degenerate_manifold,
    diagonalize_wp_in_manifold,
    plaquette_permutations,
    identify_irreps,
)
from common.physics.isotropic_symmetry import (
    c2_xy_site_permutation,
    c2v_translation_group,
    apply_site_permutation,
    apply_combined_c2xy,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--jz", type=float, default=0.6)
    p.add_argument("--extent", type=int, nargs=2, default=[3, 3])
    p.add_argument("--k-eigenvals", type=int, default=40)
    p.add_argument("--ncv", type=int, default=120,
                   help="subespacio de Krylov de ARPACK, solo para "
                        "--recompute. El defecto de scipy (2k+1 = 51 para "
                        "k=25) NO basta aqui: devuelve 14 de las 18 copias del "
                        "nivel fundamental y rellena con estados de arriba.")
    p.add_argument("--exact-data", type=str,
                   default="data/raw/energies_eigenvecs_dict_k40.npz",
                   help="npz con un espectro ya calculado. El de k=40 es el "
                        "cacheado que si recupera las 18 copias a Jz=0.6.")
    p.add_argument("--recompute", action="store_true",
                   help="rediagonaliza con Lanczos en vez de leer --exact-data.")
    p.add_argument("--manifold-size", type=int, default=None,
                   help="fuerza el tamano del manifold en vez de leerlo del "
                        "corte por tolerancia.")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="tolerancia de degenerate_manifold.")
    p.add_argument("--force", action="store_true",
                   help="sigue con los tests 1-3 aunque el test 0 falle.")
    return p.parse_args()


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def load_spectrum(args, hilbert, H):
    """(eigenvals, eigenvecs) para este Jz, de cache o de Lanczos."""
    if args.exact_data and not args.recompute:
        data = load_exact_results(args.exact_data)
        key = min(data, key=lambda k: abs(float(k) - args.jz))
        if abs(float(key) - args.jz) > 1e-6:
            raise SystemExit(
                "%s no contiene Jz=%s (mas cercano: %s)"
                % (args.exact_data, args.jz, key)
            )
        entry = data[key]
        ev = np.asarray(entry["energies"]).real
        evec = np.asarray(entry["eigenvectors"])
        order = np.argsort(ev)
        n_unsorted = int(np.sum(order != np.arange(ev.size)))
        print("[cache] %s, Jz=%s, %d autovalores%s"
              % (args.exact_data, key, ev.size,
                 "  (venian SIN ordenar: %d posiciones movidas)" % n_unsorted
                 if n_unsorted else ""))
        return ev[order], evec[:, order]

    print("[lanczos] k=%d, ncv=%d, dim=%d, tol=0 (precision de maquina) -- "
          "tarda unos minutos" % (args.k_eigenvals, args.ncv, H.hilbert.n_states))
    ev, evec = nk.exact.lanczos_ed(
        H, k=args.k_eigenvals, compute_eigenvectors=True,
        scipy_args={"tol": 0, "ncv": args.ncv},
    )
    order = np.argsort(ev.real)
    return ev.real[order], evec[:, order]


def manifold_integrity(H_sparse, eigvecs, manifold_idx, energy):
    """Restringe H al manifold y mide cuanto se parece a E * I.

    Esto valida que el nivel es plano, pero NO detecta que falten
    copias: cualquier subespacio de un autoespacio pasa este test igual de
    bien. Quien si detecta el truncamiento es la pureza de los W_p del test 2
    -- un subespacio que no es un autoespacio COMPLETO de H no es invariante
    bajo los W_p, y `diagonalize_wp_in_manifold` devuelve max_impurity ~ 1.
    """
    V = np.asarray(eigvecs)[:, list(manifold_idx)]
    sing = np.linalg.svd(V, compute_uv=False)
    Q, _ = np.linalg.qr(V)
    Hr = Q.conj().T @ (H_sparse @ Q)
    Hr = 0.5 * (Hr + Hr.conj().T)
    m = Hr.shape[0]
    return {
        "residual": float(np.linalg.norm(H_sparse @ Q - Q @ Hr)),
        "dev_from_EI": float(np.linalg.norm(Hr - energy * np.eye(m))),
        "eig_spread": float(np.ptp(np.linalg.eigvalsh(Hr))),
        "sigma_min": float(sing[-1]),
        "sigma_max": float(sing[0]),
    }


def test0(eigenvals, n_manifold):
    banner("TEST 0 -- estructura del nivel fundamental")

    n_show = min(len(eigenvals), 25)
    print("\n%d autovalores mas bajos (repr completo):" % n_show)
    for i in range(n_show):
        print("  E[%2d] = %r" % (i, eigenvals[i]))

    gaps = np.diff(eigenvals[:n_show])
    print("\ngaps consecutivos E[i+1]-E[i]:")
    for i, g in enumerate(gaps):
        mark = "   <-- salida del supuesto manifold" if i == n_manifold - 1 else ""
        print("  E[%2d]-E[%2d] = %r%s" % (i + 1, i, g, mark))

    intra = gaps[: n_manifold - 1]
    max_intra = float(np.max(intra))
    argmax_intra = int(np.argmax(intra))
    exit_gap = float(gaps[n_manifold - 1])

    print("\n--- veredicto test 0 ---")
    print("  gap MAXIMO dentro del manifold (indices 0..%d) = %r   (entre E[%d] y E[%d])"
          % (n_manifold - 1, max_intra, argmax_intra, argmax_intra + 1))
    print("  gap al primer excitado  E[%d]-E[%d]   = %r"
          % (n_manifold, n_manifold - 1, exit_gap))
    print("  ratio exit/intra = %.3e" % (exit_gap / max_intra))
    print("  |E0| = %.6f  ->  gap intra relativo = %.3e  (eps_maquina = %.2e)"
          % (abs(eigenvals[0]), max_intra / abs(eigenvals[0]), np.finfo(float).eps))

    if n_manifold >= 18:
        half = float(gaps[8])
        others = np.delete(intra, 8)
        print("\n  test de particion 9+9: gap E[9]-E[8] = %r" % half)
        print("    mediana de los otros gaps intra = %r" % float(np.median(others)))
        print("    maximo  de los otros gaps intra = %r" % float(np.max(others)))
        print("    ratio gap(9|9)/mediana(resto) = %.3f   (>>1 indicaria dos multipletes)"
              % (half / np.median(others)))

    absolute_ok = max_intra < 1e-12
    orders_ok = exit_gap / max_intra > 1e3
    passed = absolute_ok and orders_ok
    print("\n  criterio A (max gap intra < 1e-12)                     : %s"
          % ("OK" if absolute_ok else "NO"))
    print("  criterio B (>= 3 ordenes por debajo del gap al 1er exc): %s"
          % ("OK" if orders_ok else "NO"))
    print("  => degeneracion de %d %s"
          % (n_manifold,
             "EXACTA (dentro de precision numerica)" if passed else "DUDOSA"))
    return passed


def make_M(perm):
    """M = P_xy (x) U_xy^{(x)N} como funcion sobre vectores.

    Se aplica factor a factor (O(N 2^N)); la matriz densa seria 262144^2.
    """
    return lambda v: apply_combined_c2xy(np.asarray(v, dtype=complex), perm)


def op_norm_power(matvec, dim, n_iter=40, seed=0):
    """||C||_2 por iteracion de potencias sobre A = C^dag C = -C C.

    C = [M,H] es antihermitica (M y H son hermiticas), asi que C^dag C = -C^2
    y basta aplicar C dos veces por iteracion.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v /= np.linalg.norm(v)
    rayleigh = 0.0
    for _ in range(n_iter):
        av = -matvec(matvec(v))
        rayleigh = float(np.real(np.vdot(v, av)))
        nrm = np.linalg.norm(av)
        if nrm == 0.0:
            return 0.0
        v = av / nrm
    return float(np.sqrt(max(rayleigh, 0.0)))


def test1(graph, H_sparse, jz):
    banner("TEST 1 -- el espejo x<->y como permutacion de sitios")

    perm = c2_xy_site_permutation(graph)
    n = graph.n_nodes
    print("\nP_xy (convencion NetKet, centro = sitio 0):\n  %s" % perm.tolist())

    bijective = sorted(perm.tolist()) == list(range(n))
    involution = bool(np.array_equal(perm[perm], np.arange(n)))
    print("\n(a) biyeccion de los %d sitios : %s" % (n, bijective))
    print("    orden 2 (P_xy^2 = id)      : %s" % involution)

    edges = [(int(u), int(v)) for u, v in graph.edges()]
    colours = [int(c) for c in graph.edge_colors]
    by_colour = {c: {frozenset(e) for e, cc in zip(edges, colours) if cc == c}
                 for c in (0, 1, 2)}
    image = {c: {frozenset((int(perm[u]), int(perm[v])))
                 for (u, v), cc in zip(edges, colours) if cc == c}
             for c in (0, 1, 2)}
    names = {0: "x", 1: "y", 2: "z"}
    print("\n(b) accion sobre los CONJUNTOS de bonds (|x|=%d, |y|=%d, |z|=%d):"
          % (len(by_colour[0]), len(by_colour[1]), len(by_colour[2])))
    ok_b = True
    for c, target in ((0, 1), (1, 0), (2, 2)):
        same = image[c] == by_colour[target]
        ok_b = ok_b and same
        print("    P_xy( %s-bonds ) == %s-bonds : %s   (interseccion %d/%d)"
              % (names[c], names[target], same,
                 len(image[c] & by_colour[target]), len(by_colour[c])))
    fixed_z = sum(1 for e, cc in zip(edges, colours)
                  if cc == 2 and frozenset((int(perm[e[0]]), int(perm[e[1]]))) == frozenset(e))
    print("    z-bonds fijos UNO A UNO (no solo como conjunto): %d/%d"
          % (fixed_z, len(by_colour[2])))

    dim = H_sparse.shape[0]
    M = make_M(perm)

    rng = np.random.default_rng(1)
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v /= np.linalg.norm(v)
    print("\n(c) M = P_xy (x) U_xy^%d,  U_xy = (sx+sy)/sqrt(2)" % n)
    print("    ||M v|| - 1        = %.3e   (unitaria)" % (np.linalg.norm(M(v)) - 1.0))
    print("    ||M(M v) - v||     = %.3e   (M^2 = 1)" % np.linalg.norm(M(M(v)) - v))

    def comm(x):
        return M(H_sparse @ x) - H_sparse @ M(x)

    rand_ests = []
    for s in range(5):
        rr = np.random.default_rng(100 + s)
        w = rr.normal(size=dim) + 1j * rr.normal(size=dim)
        w /= np.linalg.norm(w)
        rand_ests.append(float(np.linalg.norm(comm(w))))
    norm_M = op_norm_power(comm, dim)
    print("\n    Jz=%s, Jx=Jy=%s" % (jz, (1 - jz) / 2))
    print("    ||[M,H] v|| sobre 5 vectores aleatorios unitarios:")
    print("      " + "  ".join("%.3e" % e for e in rand_ests))
    print("    ||[M,H]||_2 (iteracion de potencias, 40 iter) = %.6e" % norm_M)
    print("    escala de referencia: ||H|| ~ |E0| ~ 5.7, eps*||H|| ~ 1e-15")

    def perm_only(x):
        return apply_site_permutation(np.asarray(x, dtype=complex), perm)

    def comm_perm(x):
        return perm_only(H_sparse @ x) - H_sparse @ perm_only(x)

    norm_perm = op_norm_power(comm_perm, dim, n_iter=25, seed=7)
    print("\n    [control] ||[P_xy, H]||_2 SIN el factor de espin = %.6e" % norm_perm)
    print("              (debe ser O(1): mide que el test tiene poder)")

    ok_c = norm_M < 1e-10
    print("\n--- veredicto test 1 ---")
    print("  (a) permutacion valida      : %s" % ("OK" if bijective and involution else "NO"))
    print("  (b) x<->y, z invariante     : %s" % ("OK" if ok_b else "NO"))
    print("  (c) ||[M,H]|| < 1e-10       : %s" % ("OK" if ok_c else "NO"))
    return perm, M, (bijective and involution and ok_b and ok_c)


def orbit_of(pattern, plaq_perms):
    """Imagenes de un patron bajo un conjunto de permutaciones de plaquetas.

    Misma convencion que `_pattern_orbit` de exact_diag: new[i] = old[perm[i]].
    """
    return {tuple(pattern[i] for i in p) for p in plaq_perms}


def test2(graph, eigvecs, manifold_idx, wilson_loops, plaquettes, perm_xy, M):
    banner("TEST 2 -- M conecta las dos orbitas?")

    res = diagonalize_wp_in_manifold(eigvecs, manifold_idx, wilson_loops, seed=0)
    patterns = [tuple(int(s) for s in p) for p in res["sign_patterns"]]
    m = len(patterns)
    print("\ndiagonalize_wp_in_manifold: m=%d, pure=%s, max_impurity=%.3e"
          % (m, res["pure"], res["max_impurity"]))
    if not res["pure"]:
        print("\n    *** AVISO: los W_p NO son puros en este subespacio. ***")
        print("    Como [H, W_p] = 0, todo autoespacio COMPLETO de H es")
        print("    invariante bajo los W_p y da pureza ~1e-15. Impureza ~1")
        print("    significa que el manifold esta TRUNCADO: a ARPACK le")
        print("    faltan copias del nivel. Sube --k-eigenvals / --ncv, o usa")
        print("    data/raw/energies_eigenvecs_dict_k40.npz. Todo lo que sigue")
        print("    en los tests 2 y 3 es basura mientras esto no este limpio.")

    n_minus = [sum(1 for s in p if s < 0) for p in patterns]
    sums = [sum(p) for p in patterns]
    distinct = sorted(set(patterns))
    print("\n(a) N_- por estado : %s" % n_minus)
    print("    S = sum_p w_p  : %s" % sums)
    print("    todos con el mismo N_- : %s  (N_- = %s)"
          % (len(set(n_minus)) == 1, sorted(set(n_minus))))
    print("    todos con el mismo S   : %s  (S = %s)"
          % (len(set(sums)) == 1, sorted(set(sums))))
    print("    patrones DISTINTOS: %d  (de %d estados)" % (len(distinct), m))
    for i, p in enumerate(distinct):
        idx = [j for j, q in enumerate(patterns) if q == p]
        print("      #%2d  %s   vortices en %s   estados %s"
              % (i, list(p), [k for k, s in enumerate(p) if s < 0], idx))

    trans = np.asarray(graph.translation_group().to_array())
    tperms = plaquette_permutations(plaquettes, trans)
    print("\n(b) permutaciones de plaqueta inducidas por las traslaciones: %d"
          % len(tperms))
    remaining = set(distinct)
    orbits = []
    while remaining:
        rep = min(remaining)
        full_orb = orbit_of(rep, tperms)
        orb = full_orb & set(distinct)
        orbits.append((rep, sorted(orb), sorted(full_orb)))
        remaining -= orb
    print("    numero de orbitas: %d" % len(orbits))
    for i, (rep, orb, full) in enumerate(orbits):
        print("      orbita %d: tamano %d (orbita completa bajo T: %d)  representante %s"
              % (i, len(orb), len(full), list(rep)))

    if len(orbits) < 2:
        print("\n    (!) solo hay una orbita: (c) y (d) no aplican tal cual")

    vecs = res["vectors"]
    Mv = np.column_stack([M(vecs[:, i]) for i in range(m)])
    overlap = np.abs(vecs.conj().T @ Mv)
    orbit_of_state = []
    for p in patterns:
        found = -1
        for i, (rep, orb, full) in enumerate(orbits):
            if p in set(full):
                found = i
                break
        orbit_of_state.append(found)
    print("\n(c) |<v_i| M |v_j>| en la base de flujo (%dx%d)" % (m, m))
    print("    orbita de cada estado: %s" % orbit_of_state)
    print("      j:    " + " ".join("%4d" % j for j in range(m)))
    for i in range(m):
        row = " ".join("%4.2f" % overlap[i, j] for j in range(m))
        print("    i=%2d[o%d] %s" % (i, orbit_of_state[i], row))

    n_orb = len(orbits)
    block = np.zeros((n_orb, n_orb))
    for i in range(m):
        for j in range(m):
            if orbit_of_state[i] >= 0 and orbit_of_state[j] >= 0:
                block[orbit_of_state[i], orbit_of_state[j]] += overlap[i, j] ** 2
    print("\n    peso sum_ij |<v_i|M|v_j>|^2 por bloque de orbita:")
    for a in range(n_orb):
        print("      " + " ".join("%8.4f" % block[a, b] for b in range(n_orb)))

    if n_orb >= 2:
        iA = next(i for i in range(m) if orbit_of_state[i] == 0)
        iB = next(i for i in range(m) if orbit_of_state[i] == 1)
        wB = np.sqrt(sum(overlap[k, iA] ** 2 for k in range(m) if orbit_of_state[k] == 1))
        wA = np.sqrt(sum(overlap[k, iA] ** 2 for k in range(m) if orbit_of_state[k] == 0))
        print("\n    representantes: |A> = v_%d %s" % (iA, list(patterns[iA])))
        print("                    |B> = v_%d %s" % (iB, list(patterns[iB])))
        print("    |<B| M |A>|          = %.9f" % overlap[iB, iA])
        print("    ||P_orbitaB M |A>||  = %.9f" % wB)
        print("    ||P_orbitaA M |A>||  = %.9f" % wA)

    pperm = plaquette_permutations(plaquettes, np.asarray([perm_xy]))
    print("\n(d) permutacion de plaquetas inducida por P_xy: %s" % pperm)
    if pperm and orbits:
        A = orbits[0][0]
        imgA = tuple(A[i] for i in pperm[0])
        print("    patron A          = %s" % list(A))
        print("    P_xy . patron A   = %s" % list(imgA))
        print("    esta en la orbita de A ? %s" % (imgA in set(orbits[0][2])))
        if len(orbits) > 1:
            print("    esta en la orbita de B ? %s   (B = %s)"
                  % (imgA in set(orbits[1][2]), list(orbits[1][0])))
    return res, patterns, orbits, orbit_of_state


def test3(graph, hilbert, plaquettes, patterns, orbits, res):
    banner("TEST 3 -- estabilizador y multiplicidad")

    c2v_perms, grades = c2v_translation_group(graph)
    c2v_plaq = plaquette_permutations(plaquettes, c2v_perms)
    print("\n(a) grupo C2v espacial: |G| = %d  (grados de espin: %s)"
          % (len(c2v_perms), np.bincount(grades).tolist()))
    print("    permutaciones de plaqueta inducidas: %d (%d distintas)"
          % (len(c2v_plaq), len(set(c2v_plaq))))
    A = orbits[0][0]
    images = [tuple(A[i] for i in p) for p in c2v_plaq]
    orbit = set(images)
    stab = sum(1 for im in images if im == A)
    print("    patron A = %s" % list(A))
    print("    |orbita(A)| = %d   |estabilizador(A)| = %d" % (len(orbit), stab))
    print("    orbita x estabilizador = %d  (debe ser %d)  -> %s"
          % (len(orbit) * stab, len(c2v_plaq),
             "OK" if len(orbit) * stab == len(c2v_plaq) else "NO"))
    distinct = sorted(set(patterns))
    print("    la orbita C2v de A cubre los %d patrones del manifold? %s"
          % (len(distinct), set(distinct) <= orbit))
    if len(orbits) > 1:
        B = orbits[1][0]
        print("    patron B = %s en la orbita C2v de A ? %s" % (list(B), B in orbit))

    sym = get_kitaev_symmetries(graph, hilbert)
    sg_space, ct_space = get_projection_group(sym, "space")
    vecs = res["vectors"]
    m = vecs.shape[1]
    n_irreps = ct_space.shape[0]
    dims = [float(np.real(ct_space[i, 0])) for i in range(n_irreps)]
    print("\n(b) grupo espacial: |G| = %d, %d irreps, d_Gamma = %s"
          % (len(sg_space), n_irreps, [round(d, 6) for d in dims]))
    print("    (%d de dim 1, %d de dim 2; sum d^2 = %.1f)"
          % (sum(1 for d in dims if abs(d - 1) < 0.1),
             sum(1 for d in dims if abs(d - 2) < 0.1),
             sum(d * d for d in dims)))
    totals = {i: 0.0 for i in range(n_irreps)}
    for i in range(m):
        w = identify_irreps(vecs[:, i], hilbert, sg_space, ct_space)
        for k, val in w.items():
            totals[k] += val
    print("    n_Gamma = Tr(P_Gamma | manifold), sobre una base ortonormal de "
          "los %d estados:" % m)
    for k in range(n_irreps):
        print("      irrep %d: d_Gamma = %.4f   n_Gamma = %+.9f   m_Gamma = n/d = %.4f"
              % (k, dims[k], totals[k], totals[k] / dims[k]))
    print("    sum_Gamma n_Gamma = %.9f  (debe ser %d)" % (sum(totals.values()), m))

    trivial = [k for k in range(n_irreps)
               if np.allclose(ct_space[k, :], 1.0, atol=1e-8)]
    print("\n(c) irreps 1D del grupo espacial: %s"
          % [k for k in range(n_irreps) if abs(dims[k] - 1) < 0.1])
    print("    irrep TRIVIAL (todos los caracteres = 1): %s" % trivial)
    for k in range(n_irreps):
        if abs(dims[k] - 1) < 0.1:
            tag = "trivial" if k in trivial else "otra 1D"
            print("      n_(%s, irrep %d) = %+.9f" % (tag, k, totals[k]))

    sg_tr, ct_tr = get_projection_group(sym, "translation")
    tot_tr = {i: 0.0 for i in range(ct_tr.shape[0])}
    for i in range(m):
        w = identify_irreps(vecs[:, i], hilbert, sg_tr, ct_tr)
        for k, val in w.items():
            tot_tr[k] += val
    print("\n    grupo de TRASLACIONES puro (|G| = %d, %d irreps 1D):"
          % (len(sg_tr), ct_tr.shape[0]))
    for k in range(ct_tr.shape[0]):
        print("      momento k=%d: n_k = %+.9f" % (k, tot_tr[k]))
    print("      sum_k n_k = %.9f" % sum(tot_tr.values()))
    print("\n    -> n_(k=0) del grupo de traslaciones = %.6f" % tot_tr[0])


def main():
    args = parse_args()
    jz = args.jz
    jx = jy = (1.0 - jz) / 2.0

    graph, hilbert = build_kitaev_lattice(extent=tuple(args.extent), pbc=True)
    print("lattice %s  N=%d  dim=%d" % (tuple(args.extent), graph.n_nodes,
                                        hilbert.n_states))
    print("Jz=%s  Jx=Jy=%s" % (jz, jx))

    H = KitaevTransverse_H(
        graph.edge_colors, graph.edges(), Jx=jx, Jy=jy, Jz=jz, h=0, hi=hilbert
    )
    eigenvals, eigenvecs = load_spectrum(args, hilbert, H)

    detected = degenerate_manifold(eigenvals, tol=args.tol)
    n_manifold = args.manifold_size or len(detected)
    manifold_idx = list(range(n_manifold))
    if not args.manifold_size and detected != manifold_idx:
        raise SystemExit(
            "el manifold detectado %s no son los primeros %d indices tras "
            "ordenar -- revisa el espectro" % (detected, n_manifold)
        )
    print("[manifold] degenerate_manifold(tol=%g) -> %d estados"
          % (args.tol, len(manifold_idx)))

    ok0 = test0(eigenvals, n_manifold)
    if not ok0 and not args.force:
        print("\n*** TEST 0 NEGATIVO: la degeneracion no es exacta. ***")
        print("*** Se para aqui: invalida la premisa de los tests 1-3. ***")
        print("*** Usa --force para continuar de todos modos.          ***")
        return

    H_sparse = H.to_sparse()

    integ = manifold_integrity(H_sparse, eigenvecs, manifold_idx, eigenvals[0])
    print("\n--- integridad del manifold (H restringido al subespacio) ---")
    print("  ||H Q - Q (Q^dag H Q)||_F = %.3e" % integ["residual"])
    print("  ||Q^dag H Q - E0 * I||_F  = %.3e" % integ["dev_from_EI"])
    print("  spread de autovalores     = %.3e" % integ["eig_spread"])
    print("  sigma_min/max del bloque crudo de Lanczos = %.3e / %.3e"
          % (integ["sigma_min"], integ["sigma_max"]))

    perm_xy, M, ok1 = test1(graph, H_sparse, jz)

    plaquettes, ops = get_kitaev_plaquettes(graph)
    wilson_loops = [w.to_sparse() for w in build_wilson_loops(hilbert, plaquettes, ops)]
    res, patterns, orbits, orbit_of_state = test2(
        graph, eigenvecs, manifold_idx, wilson_loops, plaquettes, perm_xy, M
    )

    test3(graph, hilbert, plaquettes, patterns, orbits, res)

    banner("FIN")


if __name__ == "__main__":
    main()
