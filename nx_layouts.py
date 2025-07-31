import math
import networkx as nx

def plot_caterpillar(G, hsep=1.0, vsep=1.0):
    """
    Computes a caterpillar layout so that:
      - the tree's diameter (longest path) is laid out horizontally,
      - side-branches of degree 1 or 2 are placed strictly up/down,
      - side-branches of degree >2 are fanned out evenly around each spine node.

    Returns a pos dict suitable for nx.draw.
    """
    # 1. Find one endpoint of the diameter
    start = next(iter(G))
    far1 = max(
        nx.single_source_shortest_path_length(G, start),
        key=lambda x: nx.shortest_path_length(G, start, x)
    )
    # 2. From that endpoint, find the opposite endpoint & the path
    far2 = max(
        nx.single_source_shortest_path_length(G, far1),
        key=lambda x: nx.shortest_path_length(G, far1, x)
    )
    spine = nx.shortest_path(G, far1, far2)

    # 3. Position the spine on y=0
    pos = {v: (i*hsep, 0) for i, v in enumerate(spine)}

    # 4. Attach side-branches
    for i, v in enumerate(spine):
        x0, y0 = pos[v]
        leaves = [u for u in G.neighbors(v) if u not in spine]
        b = len(leaves)
        if b == 1:
            pos[leaves[0]] = (x0, y0 + vsep)
        elif b == 2:
            pos[leaves[0]] = (x0, y0 + vsep)
            pos[leaves[1]] = (x0, y0 - vsep)
        elif b > 2:
            if b % 2 == 0:
                # even: full‐circle slots excluding 0 and π
                slots = b + 2
                step = 2*math.pi / slots
                angles = [step*j for j in range(1, slots) if j != slots//2]
            else:
                # odd: split into top‐semi and bottom‐semi
                top_n = math.ceil(b/2)
                bot_n = b - top_n
                # angles in (0,π): θ = π*(k+1)/(top_n+1), k=0..top_n-1
                top_angles = [math.pi*(k+1)/(top_n+1) for k in range(top_n)]
                # angles in (π,2π): θ = π + π*(k+1)/(bot_n+1), k=0..bot_n-1
                bot_angles = [math.pi + math.pi*(k+1)/(bot_n+1) for k in range(bot_n)]
                angles = top_angles + bot_angles

            for leaf, θ in zip(leaves, angles):
                pos[leaf] = (x0 + math.cos(θ)*vsep,
                             y0 + math.sin(θ)*vsep)

    return pos

def plot_sunshine(G, radius=1.0, vsep=0.5):
    """
    “Sunshine” layout with fixed 30° spacing for ≤6 tails:
      - Largest cycle laid out as regular n‐gon.
      - First‐level tails:
          * b == 1: on the radial ray.
          * 2 ≤ b ≤ 6: spaced by ±30°*(i - (b-1)/2) around the base angle.
          * b > 6: fanned over ±90° as before.
      - Deeper nodes follow their parent’s ray outward by vsep.
    """
    # 1) Largest cycle
    cycles = nx.cycle_basis(nx.Graph(G))
    cycle = max(cycles, key=len)
    n = len(cycle)

    # 2) Place cycle nodes on regular polygon
    pos = {}
    cycle_angle = {}
    for i, v in enumerate(cycle):
        θ = 2*math.pi * i / n
        cycle_angle[v] = θ
        pos[v] = (math.cos(θ)*radius, math.sin(θ)*radius)

    # 3) First‐level tails
    queue = []
    direction = {}

    for v in cycle:
        base = cycle_angle[v]
        tails = [u for u in G.neighbors(v) if u not in cycle]
        b = len(tails)
        if b == 0:
            continue

        x0, y0 = pos[v]
        if b == 1:
            angles = [base]
        elif b <= 6:
            # center them on base, 30° apart
            # compute offsets: -(b-1)/2 ... +(b-1)/2 times 30°
            offsets = [math.radians(30*(i - (b-1)/2)) for i in range(b)]
            angles  = [base + off for off in offsets]
        else:
            # fan from base-90° to base+90°
            span = math.pi
            step = span / (b-1)
            angles = [base - math.pi/2 + j*step for j in range(b)]

        for u, θ in zip(tails, angles):
            dx, dy = math.cos(θ), math.sin(θ)
            pos[u] = (x0 + dx*vsep, y0 + dy*vsep)
            direction[u] = (dx, dy)
            queue.append(u)

    # 4) BFS for deeper nodes
    visited = set(cycle) | set(direction)
    while queue:
        parent = queue.pop(0)
        dx, dy = direction[parent]
        x0, y0 = pos[parent]
        for w in G.neighbors(parent):
            if w in visited:
                continue
            pos[w] = (x0 + dx*vsep, y0 + dy*vsep)
            direction[w] = (dx, dy)
            visited.add(w)
            queue.append(w)

    return pos