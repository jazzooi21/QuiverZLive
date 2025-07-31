import networkx as nx

class QuiverGraph(nx.MultiGraph):
    """A thin wrapper with helper methods for quiver diagrams."""

    def add_quiver_node(self, node_id, *, gp_type: str, gp_rank: int, flav_gauge: str):
        """
        Insert a node with quiver-specific attributes.

        Parameters
        ----------
        node_id : hashable
            Unique identifier of the node (e.g. a tuple grid coordinate).
        gp_type : str
            'U', 'SU', 'SO', or 'USp'.
        gp_rank : int
            Positive integer (1–12 in current UI).
        flav_gauge : str
            Either 'gauge' or 'flavour'.
        """
        self.add_node(node_id,
                      gp_type=gp_type,
                      gp_rank=gp_rank,
                      flav_gauge=flav_gauge)

    def add_quiver_edge(self, u, v, **attrs):
        """Add an (optionally multi-) edge and carry any extra attrs."""
        self.add_edge(u, v, **attrs)