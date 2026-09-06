from collections import defaultdict

from netket.operator.spin import sigmax, sigmay, sigmaz

_SIGMAS = {"x": sigmax, "y": sigmay, "z": sigmaz}


def get_kitaev_plaquettes(graph):
    """Enumerate the hexagonal plaquettes of a Kitaev honeycomb graph.

    Each hexagon contains exactly two bonds of each color. Starting from
    every node, this walks the six-site cycle z-x-y-z-x-y (or its mirror),
    alternating sublattices, and records the site list plus the operator
    color acting on each site (the color of the bond *not* touching it).

    :return: (plaquettes, operators) where plaquettes[p] is a list of 6 site
        indices and operators[p] is the matching list of 'x'/'y'/'z' labels.
    """
    edges = list(graph.edges())
    colors = list(graph.edge_colors)
    cmap = {0: "x", 1: "y", 2: "z"}
    all_colors = {"x", "y", "z"}

    neighbors_by_color = defaultdict(lambda: defaultdict(list))
    for (u, v), c in zip(edges, colors):
        col = cmap[c]
        neighbors_by_color[u][col].append(v)
        neighbors_by_color[v][col].append(u)

    plaquettes = []
    operators = []
    visited = set()

    for start_node in range(graph.n_nodes):
        for c0 in ("x", "y", "z"):
            if not neighbors_by_color[start_node][c0]:
                continue
            for seq in _hexagon_color_sequences(c0):
                for _ in neighbors_by_color[start_node][seq[0]]:
                    cycle = _build_cycle(start_node, seq, neighbors_by_color)
                    if cycle is None:
                        continue
                    key = tuple(sorted(cycle))
                    if key in visited:
                        continue
                    ops = _cycle_operators(cycle, seq, all_colors)
                    visited.add(key)
                    plaquettes.append(cycle)
                    operators.append(ops)

    return plaquettes, operators


def _hexagon_color_sequences(c0):
    """The two valid alternating color sequences for a Kitaev hexagon."""
    others = [c for c in ("x", "y", "z") if c != c0]
    return [
        [c0, others[0], others[1], c0, others[0], others[1]],
        [c0, others[1], others[0], c0, others[1], others[0]],
    ]


def _build_cycle(start, seq_colors, neighbors_by_color):
    """Walk a 6-site cycle following seq_colors; None if it doesn't close."""
    cycle = [start]
    current = start
    for i, col in enumerate(seq_colors):
        neighbors = neighbors_by_color[current][col]
        candidates = (
            [v for v in neighbors if v not in cycle]
            if i < 5
            else [v for v in neighbors if v == start]
        )
        if not candidates:
            return None
        current = candidates[0]
        if i < 5:
            cycle.append(current)
    return cycle if current == start else None


def _cycle_operators(cycle, seq_colors, all_colors):
    """Operator acting on each site = the color absent from its two bonds."""
    n = len(cycle)
    ops = []
    for i in range(n):
        c_in = seq_colors[(i - 1) % n]
        c_out = seq_colors[i]
        ops.append((all_colors - {c_in, c_out}).pop())
    return ops


def build_wilson_loops(hi, plaquettes, operators):
    """Build one Wilson-loop (plaquette) operator per plaquette.

    :param hi: NetKet Hilbert space
    :param plaquettes: list of 6-site index lists, from `get_kitaev_plaquettes`
    :param operators: matching list of 'x'/'y'/'z' operator labels
    """
    wilson_loops = []
    for sites, ops in zip(plaquettes, operators):
        Wp = _SIGMAS[ops[0]](hi, sites[0])
        for site, op in zip(sites[1:], ops[1:]):
            Wp = Wp @ _SIGMAS[op](hi, site)
        wilson_loops.append(Wp)
    return wilson_loops


def build_magnetization_observables(hi, N):
    """Staggered and uniform magnetization along z, normalized by N."""
    staggered = sum((-1) ** i * sigmaz(hi, i) / N for i in range(N))
    uniform = sum(sigmaz(hi, i) / N for i in range(N))
    return staggered, uniform
