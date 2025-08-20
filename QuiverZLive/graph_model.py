import networkx as nx
import numpy as np

from networkx.algorithms.isomorphism import MultiGraphMatcher

from typing import List, Dict, Tuple

class QuiverGraph(nx.MultiGraph):
    """A thin wrapper with helper methods for quiver diagrams."""

    def add_quiver_node(self, node_id, *, gp_type: str, gp_rank: int, node_type: str):
        self.add_node(node_id,
                      gp_type=gp_type,  # U SU SO USp
                      gp_rank=gp_rank,  # positive integer
                      node_type=node_type) # gauge or flav

    def add_quiver_edge(self, u, v, **attrs):
        self.add_edge(u, v, **attrs)




def compute_balance(g: QuiverGraph, node_id: int) -> int | None:
    data = g.nodes[node_id]
    gp_type   = data.get("gp_type")
    gp_rank   = data.get("gp_rank")
    flav_flag = data.get("node_type")

    if flav_flag != "gauge": #or gp_type not in ("U", "SU") or not isinstance(gp_rank, int):
        return None

    nf = 0
    for nbh in g.neighbors(node_id):
        edge_mult = g.number_of_edges(node_id, nbh)
        nbh_type  = g.nodes[nbh].get("gp_type", 0)
        nbh_rank  = g.nodes[nbh].get("gp_rank", 0)
        if nbh_type in ["U","SU","O"]:
            nf += edge_mult * nbh_rank
        elif nbh_type in ["SO", "USp"]:
            nf += edge_mult * nbh_rank/2

    if gp_type == "U":
        return nf - 2 * gp_rank
    elif gp_type == "SU":
        return nf - 2 * gp_rank + 1
    elif gp_type == "USp":
        return nf - gp_rank - 1
    elif gp_type == "SO":
        return nf - gp_rank + 1
    



# for 3d mirror quvier
def mixU_linear(g):
    
    # mixed U
    if not all(g.nodes[n].get("gp_type") in ("U", "SU") for n in g.nodes()):
        return False

    # All edges single
    for u, v in g.edges():
        if g.number_of_edges(u, v) != 1:
            return False
    
    gauge_nodes = [n for n, d in g.nodes.items() if d.get("node_type") == "gauge"]
    flav_nodes = [n for n, d in g.nodes.items() if d.get("node_type") == "flav"]

    # Check gauge nodes linear (2 deg1 nodes and rest deg2)
    subg = g.subgraph(gauge_nodes)
    degrees = [subg.degree(n) for n in subg.nodes]
    if len(gauge_nodes) > 1:
        if not (degrees.count(1) == 2 and degrees.count(2) == len(gauge_nodes) - 2 and nx.is_connected(subg)):
            return False
        
    # Check each gauge connected to one or zero flavours
    for g_node in gauge_nodes:
        flav_edge_count = sum(g.number_of_edges(g_node, flav_node) for flav_node in flav_nodes)
        if flav_edge_count > 1:
            return False

    return True




def is_isomorphic(G: QuiverGraph, H: QuiverGraph) -> bool:
    """
    MultiGraph isomorphism that preserves:
      - node attrs: gp_type, gp_rank, node_type
      - multiplicity (MultiGraph handles parallel edges)
      - (optional) edge attrs if you care: add to edge_match
    """
    def node_match(a, b):
        return (
            a.get("gp_type") == b.get("gp_type") and
            int(a.get("gp_rank", 0)) == int(b.get("gp_rank", 0)) and
            a.get("node_type") == b.get("node_type")
        )

    # If you need to match edge attributes, extend here:
    def edge_match(a, b):
        # example: require same 'attr' if present
        keys = set(a.keys()) | set(b.keys())
        for k in keys:
            if a.get(k) != b.get(k):
                return False
        return True

    # Quick rejects (cheap invariants)
    if G.number_of_nodes() != H.number_of_nodes(): return False
    if G.number_of_edges() != H.number_of_edges(): return False

    # Degree sequence by gp_type/gp_rank as extra cheap screen
    def typed_deg_sig(X):
        sig = []
        for n, d in X.nodes(data=True):
            sig.append((d.get("gp_type"), int(d.get("gp_rank", 0)), X.degree(n)))
        return sorted(sig)
    if typed_deg_sig(G) != typed_deg_sig(H): return False

    matcher = MultiGraphMatcher(G, H, node_match=node_match, edge_match=edge_match)
    return matcher.is_isomorphic()


def QG_to_Mv(g: QuiverGraph):

    # Only unitary gauge nodes, and reject any presence of flav nodes
    if any(d.get("node_type") == "flav" for _, d in g.nodes(data=True)):
        raise ValueError("Quiver contains flavor nodes (framed).")

    gauge_nodes = sorted(
        n for n, d in g.nodes(data=True)
        if d.get("node_type") == "gauge" and d.get("gp_type") == "U"
    )
    idx = {n: i for i, n in enumerate(gauge_nodes)}
    size = len(gauge_nodes)

    M = np.zeros((size, size), dtype=int)
    ranks = np.zeros(size, dtype=int)

    for i, u in enumerate(gauge_nodes):
        ranks[i] = int(g.nodes[u]["gp_rank"])
        # neighbors() includes u itself if there are self-loops
        for v in g.neighbors(u):
            j = idx.get(v)
            if j is not None:
                M[i, j] += g.number_of_edges(u, v)  # counts parallel edges (and self-loops)

    # Make sure it's symmetric; keep the larger count if asymmetries occur
    M = np.maximum(M, M.T)
    return M, ranks




def Mv_to_QG(M: np.ndarray, ranks: np.ndarray) -> QuiverGraph:

    size = len(ranks)

    Q = QuiverGraph()
    # nodes: 0..size-1
    for v, r in enumerate(ranks):
        Q.add_quiver_node(v, gp_type='U', gp_rank=int(r), node_type='gauge')

    # edges use 0-based endpoints
    for i in range(size):
        for j in range(i + 1, size):
            for _ in range(int(M[i, j])):
                Q.add_quiver_edge(i, j)

    return Q


def QG_to_Mv_idx(g: QuiverGraph) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Returns (M, ranks, idx2node) for the induced subgraph of U(gauge) nodes.
    - idx2node[i] is the *graph node id* corresponding to row/col i of M and entry i of ranks.
    - M is made symmetric via max(M, M.T), like your original.
    """
    # reject flavor nodes
    if any(d.get("node_type") == "flav" for _, d in g.nodes(data=True)):
        raise ValueError("Quiver contains flavor nodes (framed).")

    # pick only unitary gauge nodes, keep a deterministic order
    gauge_nodes = sorted(
        n for n, d in g.nodes(data=True)
        if d.get("node_type") == "gauge" and d.get("gp_type") == "U"
    )
    idx2node: List = list(gauge_nodes)         # mapping index -> node id
    node2idx: Dict = {n: i for i, n in enumerate(idx2node)}

    size = len(idx2node)
    M = np.zeros((size, size), dtype=int)
    ranks = np.zeros(size, dtype=int)

    for u in idx2node:
        i = node2idx[u]
        ranks[i] = int(g.nodes[u]["gp_rank"])
        # neighbors() may include u for self-loops; number_of_edges counts parallels
        for v in g.neighbors(u):
            j = node2idx.get(v)
            if j is not None:
                M[i, j] += g.number_of_edges(u, v)
                

    # enforce undirected multiplicities
    M = np.maximum(M, M.T)
    return M, ranks, idx2node


# ---------- (M, v, idx2node) → Graph (preserve node IDs) ----------

def Mv_idx_to_QG(M: np.ndarray, ranks: np.ndarray, idx2node: List) -> QuiverGraph:
    """
    Build a QuiverGraph whose nodes are exactly idx2node[*] (original IDs preserved).
    - Adds U(gauge) nodes with gp_rank=ranks[i].
    - Adds |M[i,j]| undirected edges for i<j (diagonal/self-loops ignored, matching your old code).
    """
    size = len(ranks)
    if M.shape != (size, size):
        raise ValueError("M shape must match len(ranks).")
    if len(idx2node) != size:
        raise ValueError("idx2node length must match len(ranks).")

    Q = QuiverGraph()

    # create nodes using original IDs
    for i, node_id in enumerate(idx2node):
        Q.add_quiver_node(node_id, gp_type="U", gp_rank=int(ranks[i]), node_type="gauge")

    # add undirected edges (ignore diagonal to mirror previous behavior)
    for i in range(size):
        for j in range(i + 1, size):
            m = int(M[i, j])
            for _ in range(m):
                Q.add_quiver_edge(idx2node[i], idx2node[j])

    return Q