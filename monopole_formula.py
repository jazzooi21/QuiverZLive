import itertools
import functools
import collections
from typing import Dict, Iterable, Tuple, Optional
import sympy as sp
import networkx as nx

from graph_model import QuiverGraph
from PySide6.QtWidgets import QMessageBox

# ----------------- Monopole-formula primitives (unitary) -----------------

def vecU(m: Tuple[int, ...]) -> sp.Integer:
    """Vector multiplet contribution for U(N):  -∑_{a<b}|m_a - m_b|  (integer)."""
    s = 0
    for i in range(1, len(m)):
        for j in range(i):
            s += abs(m[i] - m[j])
    return sp.Integer(-s)

def hypU(m_left: Tuple[int, ...], m_right: Tuple[int, ...], mult: int = 1) -> sp.Rational:
    """Bifundamental hyper between U(len(left)) and U(len(right)), times multiplicity."""
    s = 0
    for a in m_left:
        for b in m_right:
            s += abs(a - b)
    return sp.Rational(mult * s, 2)

def Flav(m: Tuple[int, ...], F: int) -> sp.Rational:
    """F fundamentals on U(len(m))."""
    return sp.Rational(F, 2) * sum(abs(x) for x in m)

def PgU_for_tuple(m: Tuple[int, ...], t: sp.Symbol) -> sp.Expr:
    """Residual gauge-group dressing P_{U(N)}(t; m)."""
    counts = [len(list(g)) for _, g in itertools.groupby(sorted(m))]
    if not counts:
        return sp.Integer(1)
    prod = sp.Integer(1)
    for c in counts:
        for k in range(1, c + 1):
            prod *= 1 / (1 - t ** (2 * k))
    return sp.simplify(prod)

def monotone_tuples(rank: int, B: int) -> Iterable[Tuple[int, ...]]:
    """
    Generate nonincreasing integer tuples (m1>=...>=mN) with each mi in [-B, B].
    """
    cur = [0] * rank
    def rec(i: int, prev: int):
        if i == rank:
            yield tuple(cur); return
        for v in range(-B, prev + 1):
            cur[i] = v
            yield from rec(i + 1, v)
    yield from rec(0, B)

# ----------------- Convert your QuiverGraph to a compact model -----------------

def _edge_mult_sum(G: nx.MultiGraph, u, v) -> int:
    """Sum 'mult' attributes across parallel edges (default 1 each)."""
    data = G.get_edge_data(u, v, default={})
    total = 0
    for k, attr in data.items():
        total += int(attr.get("mult", 1))
    return total

def sanitize_quiver_graph(G: nx.MultiGraph):
    """
    Build a compact unitary-quiver description from your QuiverGraph.

    Returns:
      gauge_nodes: list of gauge node IDs (order is fixed)
      ranks: dict node -> rank
      Ffund: dict node -> total fundamental count from attached flavor nodes
      bifund: dict frozenset({u,v}) -> multiplicity  (gauge-gauge edges only)
    """
    gauge_nodes = [n for n, a in G.nodes(data=True) if a.get("flav_gauge") == "gauge"]
    flav_nodes  = [n for n, a in G.nodes(data=True) if a.get("flav_gauge") == "flavour"]

    # Validate: only unitary gauge groups for now
    for n in gauge_nodes + flav_nodes:
        gp = G.nodes[n].get("gp_type")
        if gp != "U":
            QMessageBox.warning(None, "Non-unitary", f"Only unitary gauge nodes supported; {gp} node detected.")
            return None, None, None, None
    
    

    ranks = {n: int(G.nodes[n]["gp_rank"]) for n in gauge_nodes}

    # Fundamentals per gauge node from attached flavor nodes (rank * edge multiplicity)
    Ffund = {n: 0 for n in gauge_nodes}
    for f in flav_nodes:
        Frank = int(G.nodes[f].get("gp_rank", 0))
        for nbr in G.neighbors(f):
            if nbr in ranks:
                Ffund[nbr] += Frank * _edge_mult_sum(G, f, nbr)

    # Bifundamental multiplicities between gauge nodes
    bifund = {}
    for i, u in enumerate(gauge_nodes):
        for v in gauge_nodes[i + 1:]:
            m = _edge_mult_sum(G, u, v)
            if m:
                bifund[frozenset((u, v))] = m

    return gauge_nodes, ranks, Ffund, bifund

def gauge_is_tree(gauge_nodes, bifund) -> bool:
    """Check if the gauge-only simple graph is a tree."""
    n = len(gauge_nodes)
    if n == 0:
        return True
    m = len(bifund)
    if m != n - 1:
        return False
    # build adjacency
    adj = {u: set() for u in gauge_nodes}
    for e in bifund:
        u, v = tuple(e)
        adj[u].add(v); adj[v].add(u)
    # connectivity
    seen = set()
    stack = [gauge_nodes[0]]
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x)
        stack.extend(adj[x] - seen)
    return len(seen) == n

# ----------------- Engines (brute & tree-DP) -----------------

def _hs_bruteforce_exact(gauge_nodes, ranks, Ffund, bifund, B: int, t: sp.Symbol) -> sp.Expr:
    Ms = {n: tuple(monotone_tuples(ranks[n], B)) for n in gauge_nodes}
    Pg = {n: {m: PgU_for_tuple(m, t) for m in Ms[n]} for n in gauge_nodes}

    hs = sp.Integer(0)
    for assign in itertools.product(*[Ms[n] for n in gauge_nodes]):
        m = {n: assign[i] for i, n in enumerate(gauge_nodes)}
        dim = sp.Integer(0)
        P   = sp.Integer(1)

        # node contributions
        for n in gauge_nodes:
            dim += vecU(m[n]) + Flav(m[n], Ffund[n])
            P   *= Pg[n][m[n]]

        # edge contributions (bifundamentals)
        for e, mult in bifund.items():
            u, v = tuple(e)
            dim += hypU(m[u], m[v], mult)

        hs += t ** (2 * dim) * P
    return sp.simplify(hs)

def _hs_tree_exact(gauge_nodes, ranks, Ffund, bifund, B: int, t: sp.Symbol) -> sp.Expr:
    if not gauge_nodes:
        return sp.Integer(1)

    # build adjacency with multiplicities
    adj = {u: [] for u in gauge_nodes}
    for e, mult in bifund.items():
        u, v = tuple(e)
        adj[u].append((v, mult))
        adj[v].append((u, mult))

    # Choose root: most central node (highest degree in gauge-only graph)
    degrees = {n: len(adj[n]) for n in gauge_nodes}

    root = max(degrees, key=degrees.get)
    Ms = {n: tuple(monotone_tuples(ranks[n], B)) for n in gauge_nodes}
    Pg = {n: {m: PgU_for_tuple(m, t) for m in Ms[n]} for n in gauge_nodes}

    @functools.lru_cache(None)
    def Z(node, parent, parent_m: Tuple[int, ...]):
        total = sp.Integer(0)
        for m in Ms[node]:
            dim = vecU(m) + Flav(m, Ffund[node])
            if parent is not None:
                mult = {frozenset((node, parent)): mult for (nbr, mult) in adj[node] if nbr == parent}
                # the dict trick above ensures we fetch the correct mult
                mult = next((m for (e, m) in [(frozenset((node, parent)), next((mm for (nb, mm) in adj[node] if nb == parent), 1))]), 1)
            if parent is not None:
                # real multiplicity lookup:
                mlt = next((ml for (nb, ml) in adj[node] if nb == parent), 1)
                dim += hypU(m, parent_m, mlt)

            P = Pg[node][m]
            prod = sp.Integer(1)
            for nb, ml in adj[node]:
                if nb == parent: 
                    continue
                prod *= Z(nb, node, m)
            total += t ** (2 * dim) * P * prod
        return sp.simplify(total)

    hs = sp.Integer(0)
    for m0 in Ms[root]:
        dim0 = vecU(m0) + Flav(m0, Ffund[root])
        P0   = Pg[root][m0]
        prod = sp.Integer(1)
        for nb, ml in adj[root]:
            prod *= Z(nb, root, m0)
        hs += t ** (2 * dim0) * P0 * prod
    return sp.simplify(hs)

# ----------------- Public API -----------------

def hilbert_series_from_quiver_graph(
    G: nx.MultiGraph,
    cutoff: int = 5,
    method: str = "auto",   # 'auto' | 'tree' | 'brute'
    stabilize: bool = True,
    max_cutoff: int = 7,
):
    """
    Unrefined HS(t) from a QuiverGraph built on unitary gauge nodes.

    - Returns a Sympy series in t:  1 + a2 t^2 + ... + O(t^2cutoff)
    - Automatically increases the GNO-charge cutoff until the first 'order'
      coefficients stabilize (unless stabilize=False).

    Notes:
      * Flavor nodes are recognized by node attribute flav_gauge='flav' and only
        contribute fundamentals to adjacent gauge nodes.
      * Edge multiplicities are read from the 'mult' attribute of parallel edges
        (default 1 per edge) and summed.
    """
    t = sp.Symbol('t')

    gauge_nodes, ranks, Ffund, bifund = sanitize_quiver_graph(G)
    if not gauge_nodes:
        return None

    def compute(B: int):
        if method == "brute":
            expr = _hs_bruteforce_exact(gauge_nodes, ranks, Ffund, bifund, B, t)
        else:
            use_tree = (method == "tree") or (method == "auto" and gauge_is_tree(gauge_nodes, bifund))
            if use_tree:
                expr = _hs_tree_exact(gauge_nodes, ranks, Ffund, bifund, B, t)
            else:
                expr = _hs_bruteforce_exact(gauge_nodes, ranks, Ffund, bifund, B, t)
        return sp.series(expr, t, 0, 2*cutoff +1)

    if not stabilize:
        return compute(cutoff)

    prev = None
    B = cutoff
    while B <= max_cutoff:
        cur = compute(B)
        if prev is not None and sp.expand(cur - prev) == 0:
            return sp.series(sp.expand(cur), t, 0, 2*cutoff +1)
        prev = cur
        B += 1
    return sp.series(sp.expand(prev), t, 0, 2*cutoff +1)