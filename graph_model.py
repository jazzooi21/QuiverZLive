import networkx as nx

class QuiverGraph(nx.MultiGraph):
    """A thin wrapper with helper methods for quiver diagrams."""

    def add_quiver_node(self, node_id, *, gp_type: str, gp_rank: int, flav_gauge: str):
        self.add_node(node_id,
                      gp_type=gp_type,
                      gp_rank=gp_rank,
                      flav_gauge=flav_gauge)

    def add_quiver_edge(self, u, v, **attrs):
        self.add_edge(u, v, **attrs)


def compute_balance(g: QuiverGraph, node_id: int) -> int | None:
    data = g.nodes[node_id]
    gp_type   = data.get("gp_type")
    gp_rank   = data.get("gp_rank")
    flav_flag = data.get("flav_gauge")

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