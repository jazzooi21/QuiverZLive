import math
import networkx as nx

def plot_caterpillar(G, hsep=1.0, vsep=1.0):
    """
    Caterpillar layout:
      - Longest path (diameter) is laid horizontally on y=0 (spacing = hsep).
      - Immediate off-spine neighbors are positioned as in your original logic.
      - Any deeper nodes (branches length >= 2) extend straight out from the
        first off-spine node, spaced by vsep per hop, without moving the originals.
    Returns a pos dict suitable for nx.draw.
    """
    if len(G) == 0:
        return {}

    # --- Find tree diameter (spine) ---
    start = next(iter(G))
    far1 = max(
        nx.single_source_shortest_path_length(G, start),
        key=lambda x: nx.shortest_path_length(G, start, x)
    )
    far2 = max(
        nx.single_source_shortest_path_length(G, far1),
        key=lambda x: nx.shortest_path_length(G, far1, x)
    )
    spine = nx.shortest_path(G, far1, far2)

    # --- Place spine on y=0 ---
    pos = {v: (i * hsep, 0.0) for i, v in enumerate(spine)}
    spine_set = set(spine)

    # --- Place immediate side-branches (first-level leaves) ---
    for i, v in enumerate(spine):
        x0, y0 = pos[v]
        leaves = [u for u in G.neighbors(v) if u not in spine_set]
        b = len(leaves)
        if b == 0:
            continue
        if b == 1:
            # single leaf straight up
            pos[leaves[0]] = (x0, y0 + vsep)
        elif b == 2:
            # two leaves up & down
            pos[leaves[0]] = (x0, y0 + vsep)
            pos[leaves[1]] = (x0, y0 - vsep)
        else:
            # fan evenly; avoid 0 and π so we don't overlap the spine
            if b % 2 == 0:
                # even: full‐circle slots excluding 0 and π
                slots = b + 2
                step = 2 * math.pi / slots
                angles = [step * j for j in range(1, slots) if j != slots // 2]
            else:
                # odd: split into top and bottom semis
                top_n = math.ceil(b / 2)
                bot_n = b - top_n
                # angles in (0, π): θ = π*(k+1)/(top_n+1), k=0..top_n-1
                top_angles = [math.pi * (k + 1) / (top_n + 1) for k in range(top_n)]
                # angles in (π, 2π): θ = π + π*(k+1)/(bot_n+1), k=0..bot_n-1
                bot_angles = [math.pi + math.pi * (k + 1) / (bot_n + 1) for k in range(bot_n)]
                angles = top_angles + bot_angles

            # place first-level leaves at radius = vsep
            for leaf, theta in zip(leaves, angles):
                pos[leaf] = (x0 + math.cos(theta) * vsep,
                             y0 + math.sin(theta) * vsep)

    # --- Extend longer branches straight out along the same ray ---
    G_minus_spine = G.subgraph(G.nodes - spine_set)

    for v in spine:
        x0, y0 = pos[v]
        # only roots that already have coordinates from the previous step
        branch_roots = [u for u in G.neighbors(v) if u not in spine_set and u in pos]
        for u in branch_roots:
            xu, yu = pos[u]
            dx, dy = xu - x0, yu - y0
            norm = math.hypot(dx, dy)
            if norm == 0:
                # fallback direction straight up if degenerate
                ux, uy = 0.0, 1.0
            else:
                ux, uy = dx / norm, dy / norm  # unit vector from spine -> root

            # distances within this branch (spine removed)
            lengths = nx.single_source_shortest_path_length(G_minus_spine, u)
            for w, duw in lengths.items():
                if w == u:
                    continue  # keep root's original coordinates
                # place node w further along the same ray, spaced by vsep per hop
                pos[w] = (xu + duw * vsep * ux, yu + duw * vsep * uy)

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
    # Largest cycle
    cycles = nx.cycle_basis(nx.Graph(G))
    cycle = max(cycles, key=len)
    n = len(cycle)

    # Place cycle nodes on regular polygon
    pos = {}
    cycle_angle = {}
    for i, v in enumerate(cycle):
        θ = 2*math.pi * i / n
        cycle_angle[v] = θ
        pos[v] = (math.cos(θ)*radius, math.sin(θ)*radius)

    # First‐level tails
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

    # BFS for deeper nodes
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


def plot_sunshine_multicycles(G, base_radius=1.0, step_r=1.0, vsep=0.5, slot_deg=30):
    """
    Sunshine-like layout for *multiple cycles* (best for cactus graphs).
    Strategy:
      1) Build block–cut tree.
      2) Place a root cycle as a regular polygon.
      3) DFS over the block–cut tree; for each child block sharing an articulation 'a':
         - If cycle: place on circle centered at 'a' with angular slotting (±slot_deg).
         - If bridge/edge: extend outward along the inherited ray from 'a'.
      4) Push deeper tails per parent ray by 'vsep'.
    Params:
      base_radius: radius for the root cycle.
      step_r: radial increment per block level (keeps siblings from colliding).
      vsep: spacing per hop along rays for tails.
      slot_deg: angular spacing between sibling blocks around an articulation.
    Returns:
      pos: dict node -> (x, y)
    """
    def _is_simple_cycle_block(Gsub):
        # simple heuristic good for cactus blocks: |E| == |V| and all deg >= 2
        return Gsub.number_of_edges() == Gsub.number_of_nodes() and all(d >= 2 for _, d in Gsub.degree())
    
    def _regular_ngon(center, radius, nodes, start_angle):
        cx, cy = center
        n = len(nodes)
        return {v: (cx + radius * math.cos(start_angle + 2*math.pi*i/n),cy + radius * math.sin(start_angle + 2*math.pi*i/n),) for i, v in enumerate(nodes)}

    if len(G) == 0:
        return {}

    # --- Block–cut decomposition ---
    blocks = list(nx.biconnected_components(G))           # sets of nodes
    artics = set(nx.articulation_points(G))
    # Map each node to blocks it belongs to
    node2blocks = {v: [] for v in G.nodes}
    for i, B in enumerate(blocks):
        for v in B:
            node2blocks[v].append(i)

    # Build block–cut tree T with nodes: ("B", i) for blocks, ("A", v) for articulation points
    T = nx.Graph()
    for i, B in enumerate(blocks):
        T.add_node(("B", i))
        for v in B:
            if v in artics:
                T.add_node(("A", v))
                T.add_edge(("B", i), ("A", v))

    # Choose root block: prefer a cycle block with max size; else the largest block
    def block_is_cycle(i):
        return _is_simple_cycle_block(G.subgraph(blocks[i]).copy())

    cycle_blocks = [i for i in range(len(blocks)) if block_is_cycle(i)]
    if cycle_blocks:
        root_b = max(cycle_blocks, key=lambda i: len(blocks[i]))
    else:
        root_b = max(range(len(blocks)), key=lambda i: len(blocks[i]))

    # --- Placement state ---
    pos = {}
    # ray_dir remembers an outward unit vector to reuse for bridge blocks & tails
    ray_dir = {}  # node -> (dx, dy)

    # --- Place root block ---
    B0_nodes = list(blocks[root_b])
    G0 = G.subgraph(B0_nodes).copy()
    if block_is_cycle(root_b):
        # get an explicit cycle order (take the longest simple cycle inside the block)
        G0_simple = nx.Graph(G0) if G0.is_multigraph() else G0
        basis = nx.cycle_basis(G0_simple)
        root_cycle = max(basis, key=len)
        # lay the cycle as an n-gon around origin
        pos.update(_regular_ngon((0.0, 0.0), base_radius, root_cycle, start_angle=0.0))
        # If block has extra (rare for cactus), spring them inside (fallback)
        extras = set(B0_nodes) - set(root_cycle)
        if extras:
            # planar/spring fallback; keep cycle fixed
            ptmp = nx.spring_layout(G0, pos={v: pos[v] for v in pos if v in B0_nodes}, fixed=list(set(root_cycle)))
            pos.update({v: ptmp[v] for v in extras})
        # set outward rays per cycle node as its radial from center
        for v in root_cycle:
            x, y = pos[v]
            r = math.hypot(x, y) or 1.0
            ray_dir[v] = (x / r, y / r)
    else:
        # non-cycle root: place roughly on a line and define rays arbitrarily
        ptmp = nx.spring_layout(G0)
        pos.update(ptmp)
        for v in B0_nodes:
            x, y = pos[v]
            r = math.hypot(x, y) or 1.0
            ray_dir[v] = (x / r, y / r)

    # --- Traverse block–cut tree from the root block node ---
    root = ("B", root_b)
    parent = {root: None}
    order = list(nx.bfs_tree(T, root).nodes())

    # angular bookkeeping at each articulation to slot children without overlap
    used_slots = {}  # ("A", v) -> current integer slot index (…,-1,0,+1, …)

    def next_slot(av):
        k = used_slots.get(av, -1) + 1
        used_slots[av] = k
        return k

    for x in order[1:]:
        p = next(n for n in T.neighbors(x) if n == parent[x] or parent.get(n) == x or parent.get(x) == n) \
            if x in parent else parent.setdefault(x, next(iter(T.neighbors(x))))
        parent[x] = p
        # We only place blocks; articulation nodes just route angles
        if x[0] != "B":
            continue

        Bi = x[1]
        Bnodes = list(blocks[Bi])
        GBi = G.subgraph(Bnodes).copy()

        # Find the articulation vertex connecting this block to its parent articulation
        # path in T: ("B", Bi) -- ("A", a) -- ("B", parent_block)
        arts_here = [n for n in T.neighbors(("B", Bi)) if n[0] == "A"]
        # pick the articulation that is already positioned (belongs to parent side)
        a_node = None
        for av in arts_here:
            v = av[1]
            if v in pos:
                a_node = v
                a_tag = ("A", v)
                break
        if a_node is None:
            # if none placed yet (rare at root), just skip; will be placed later pass
            continue

        ax, ay = pos[a_node]
        # base direction at the articulation (reuse if known; else point right)
        base_dxdy = ray_dir.get(a_node, (1.0, 0.0))
        base_ang = math.atan2(base_dxdy[1], base_dxdy[0])

        # allocate a sibling slot around the articulation
        k = next_slot(a_tag)
        θ = base_ang + math.radians(slot_deg) * k

        # distance from articulation to place this block's perimeter
        R = base_radius + step_r * (nx.shortest_path_length(T, source=root, target=x) // 2)
        if _is_simple_cycle_block(GBi):
            # extract that cycle’s order
            GBi_simple = nx.Graph(GBi) if GBi.is_multigraph() else GBi
            basis = nx.cycle_basis(GBi_simple)
            cyc = max(basis, key=len)

            # Put articulation vertex 'a_node' at angle θ; distribute the rest around
            n = len(cyc)
            # Position center for this cycle at distance R along θ
            cx = ax + R * math.cos(θ)
            cy = ay + R * math.sin(θ)
            cy = ay + R * math.sin(θ)

            # Place as n-gon; rotate so 'a_node' lands near the articulation side
            # First, create an initial n-gon order starting with a_node (if present)
            if a_node in cyc:
                start_idx = cyc.index(a_node)
                cyc_order = cyc[start_idx:] + cyc[:start_idx]
            else:
                cyc_order = cyc

            P = _regular_ngon((cx, cy), base_radius, cyc_order, start_angle=θ + math.pi)  # polygon faces back toward a
            # keep existing articulation position (do not move a_node), but define its ray
            if a_node in P:
                P[a_node] = (ax, ay)
            pos.update(P)

            # Set outward rays for the cycle’s nodes (from local center)
            for v in cyc_order:
                xx, yy = pos[v]
                rdx, rdy = xx - cx, yy - cy
                r = math.hypot(rdx, rdy) or 1.0
                ray_dir[v] = (rdx / r, rdy / r)
        else:
            # Treat as a small tree/bridge block: push other nodes outward from 'a_node'
            # Choose a unit ray from articulation
            ux, uy = math.cos(θ), math.sin(θ)
            # BFS inside this block starting at a_node (if present), else pick any and aim outward
            start = a_node if a_node in Bnodes else Bnodes[0]
            dists = nx.single_source_shortest_path_length(GBi, start)
            for v, d in dists.items():
                if v == a_node:
                    continue
                pos[v] = (ax + d * vsep * ux, ay + d * vsep * uy)
                ray_dir[v] = (ux, uy)

        # Propagate parent pointers for BFS over T
        for y in T.neighbors(x):
            if y not in parent:
                parent[y] = x

    # --- Finally, extend any remaining tail nodes (outside all blocks) along their ray ---
    # (This is helpful if the graph had extra tree parts not captured above.)
    visited = set(pos)
    queue = [v for v in pos if v in ray_dir]
    while queue:
        u = queue.pop()
        ux, uy = ray_dir[u]
        for w in G.neighbors(u):
            if w in visited:
                continue
            # first-time placement for tail
            x0, y0 = pos[u]
            pos[w] = (x0 + vsep * ux, y0 + vsep * uy)
            ray_dir[w] = (ux, uy)
            visited.add(w)
            queue.append(w)

    return pos
