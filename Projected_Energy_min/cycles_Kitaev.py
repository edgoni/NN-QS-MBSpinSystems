from collections import defaultdict
from netket.operator.spin import sigmaz, sigmax, sigmay

def obtener_plaquetas_kitaev(graph):
    """
    Estrategia: cada hexágono de Kitaev contiene exactamente 2 enlaces de
    cada color. Partimos de cada enlace z y construimos el hexágono
    siguiendo la estructura honeycomb: z→x→y→z→x→y (alternando subred A/B).
    """
    edges  = list(graph.edges())
    colors = list(graph.edge_colors)
    cmap   = {0:'x', 1:'y', 2:'z'}
    todos  = {'x','y','z'}

    
    vecinos_color = defaultdict(lambda: defaultdict(list))
    for (u,v), c in zip(edges, colors):
        col = cmap[c]
        vecinos_color[u][col].append(v)
        vecinos_color[v][col].append(u)

    plaquetas  = []
    operadores = []
    visitadas  = set()


    secuencia_colores = ['z', 'x', 'y', 'z', 'x', 'y']

    for start_node in range(graph.n_nodes):
        for c0 in ['x', 'y', 'z']:
            if not vecinos_color[start_node][c0]:
                continue

            seq = [c0,
                   sorted(todos - {c0})[0],
                   sorted(todos - {c0})[1],
                   c0,
                   sorted(todos - {c0})[0],
                   sorted(todos - {c0})[1]]

            for seq in _secuencias_hexagono(c0):
                for v0 in vecinos_color[start_node][seq[0]]:
                    ciclo = _construir_ciclo(start_node, seq, vecinos_color)
                    if ciclo is None:
                        continue
                    key = tuple(sorted(ciclo))
                    if key in visitadas:
                        continue
                    ops = _calcular_ops(ciclo, seq, todos)
                    visitadas.add(key)
                    plaquetas.append(ciclo)
                    operadores.append(ops)

    return plaquetas, operadores


def _secuencias_hexagono(c0):
    """Las dos secuencias válidas de colores para un hexágono de Kitaev."""
    todos = ['x','y','z']
    otros = [c for c in todos if c != c0]
    return [
        [c0, otros[0], otros[1], c0, otros[0], otros[1]],
        [c0, otros[1], otros[0], c0, otros[1], otros[0]],
    ]


def _construir_ciclo(start, seq_colores, vecinos_color):
    """
    Intenta construir un ciclo de 6 nodos siguiendo la secuencia de colores.
    Devuelve la lista de 6 nodos o None si no existe.
    """
    ciclo = [start]
    nodo_actual = start
    for i, col in enumerate(seq_colores):
        vecinos = vecinos_color[nodo_actual][col]
        if i < 5:
            candidatos = [v for v in vecinos if v not in ciclo]
        else:
            candidatos = [v for v in vecinos if v == start]
        if not candidatos:
            return None
        if len(candidatos) > 1:
            pass
        nodo_actual = candidatos[0]
        if i < 5:
            ciclo.append(nodo_actual)
    if nodo_actual != start:
        return None
    return ciclo


def _calcular_ops(ciclo, seq_colores, todos):
    """
    Para cada sitio i del ciclo:
      - enlace entrante: seq_colores[i-1]
      - enlace saliente: seq_colores[i]
      - operador exterior = el color restante
    """
    n = len(ciclo)
    ops = []
    for i in range(n):
        c_in  = seq_colores[(i - 1) % n]
        c_out = seq_colores[i]
        op    = (todos - {c_in, c_out}).pop()
        ops.append(op)
    return ops


def build_Wp_operators(hi, plaquetas, operadores_colores):
    """
    Construye una lista de operadores Wp, uno por plaqueta.

    - plaquetas: lista de listas de sitios [i1, i2, ..., i6]
    - operadores_colores: lista de listas de 'x','y','z' del mismo largo
    """
    mapa_sigmas = {
        'x': sigmax,
        'y': sigmay,
        'z': sigmaz,
    }

    Wp_list = []

    for p_idx, nodos in enumerate(plaquetas):
        ops_p = operadores_colores[p_idx]

        Wp = mapa_sigmas[ops_p[0]](hi, nodos[0])

        for site, op_char in zip(nodos[1:], ops_p[1:]):
            Wp = Wp @ mapa_sigmas[op_char](hi, site)

        Wp_list.append(Wp)
        print(f"Construido Wp para plaqueta {p_idx} con nodos {nodos} y ops {ops_p}")

    return Wp_list