import networkx as nx

def Dynkin_A(n):
    if n < 1:
        raise ValueError("n must be at least 1 for type A_n")
    return nx.path_graph(n)

def Dynkin_D(n):
    if n < 4:
        raise ValueError("n must be at least 4 for type D_n")
    G = nx.Graph()
    G.add_edges_from((i, i+1) for i in range(1,n-1))  # linear chain
    G.add_edge(n-2, n)
    return G

def Dynkin_E(n):
    if n in (6, 7, 8):
        G = nx.Graph()
        G.add_edges_from((i, i+1) for i in range(1,n-1))  # linear chain: 0-1-2-...-(n-2)
        G.add_edge(n-3, n)
    else:
        raise ValueError("n must be 6, 7, or 8 for type E_n")
    return G
