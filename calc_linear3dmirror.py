from __future__ import annotations

from typing import List, Sequence, Tuple, Generator

import numpy as np

from graph_model import QuiverGraph

# ────────────────────────── partition helpers ───────────────────────────

def _transpose_partition(part: Sequence[int]) -> List[int]:
    if not part:
        return []
    m = max(part)
    return [sum(1 for x in part if x >= i) for i in range(1, m + 1)]


def _cumul(part: Sequence[int]) -> List[int]:
    return [sum(part[i:]) for i in range(len(part))]


def _collapse_pair(a: int, b: int) -> List[int]:
    if a > b:
        return [a, b]
    tot = a + b
    # use integer halves (Mathematica used Floor)
    return [tot - (tot // 2), (tot // 2)]


def _collapse_once(part: List[int]) -> List[int]:
    unordered = [k for k in range(len(part) - 1) if part[k] < part[k + 1]]
    if not unordered:
        return part[:]
    i = min(unordered)
    return part[:i] + _collapse_pair(part[i], part[i + 1]) + part[i + 2:]


def _collapse(part: List[int]) -> List[int]:
    out = part[:]
    while True:
        nxt = _collapse_once(out)
        if nxt == out:
            return out
        out = nxt


# ───────────────────────────── main class ───────────────────────────────
class MagneticQuiver:
    """Compute magnetic quiver(s) for a given electric quiver.

    Parameters
    ----------
    flavor_nodes : list[int]
        Fundamental hypermultiplet multiplicities attached to each gauge node.
    gauge_ranks  : list[int]
        Ranks of the gauge groups in the electric quiver.
    gauge_type   : Sequence[str]
        Gauge‑group types, e.g. ["u", "s", "u"]. Use "s" for **SU**, any other
        string for **U**. Case‑insensitive.
    """

    # ───── construction ─────
    def __init__(self,
                 flavor_nodes: Sequence[int],
                 gauge_ranks : Sequence[int],
                 gauge_type  : Sequence[str]):

        if len(flavor_nodes) != len(gauge_ranks):
            raise ValueError("flavor_nodes and gauge_ranks must have equal length")
        if len(gauge_type) != len(gauge_ranks):
            raise ValueError("gauge_type must have one entry per gauge node")

        self.flavor_nodes = list(map(int, flavor_nodes))
        self.gauge_ranks  = list(map(int, gauge_ranks))
        # normalise to lowercase strings
        self.gauge_type   = [str(t).lower() for t in gauge_type]

        self.nf = sum(self.flavor_nodes)  # total flavours
        self.n  = len(self.gauge_ranks)   # number of gauge factors

        # ------------------------------------------------------------------
        # Pre‑computations
        # ------------------------------------------------------------------
        gn_ext = [0] + self.gauge_ranks + [0]
        fn_ext = [0] + self.flavor_nodes + [0]

        # Slopes for the (n+1) NS5 intervals
        self._slopes = [
            gn_ext[i] - gn_ext[i + 1] + sum(fn_ext[j] for j in range(i + 1, self.n + 2))
            for i in range(self.n + 1)
        ]

        # Total left branes available per flavour index
        self._total_left = list(reversed(_cumul(_transpose_partition(_cumul(self.flavor_nodes)))))

        # Colour blocks induced by U/SU pattern
        su_pos = [i for i, t in enumerate(self.gauge_type) if t == 's']
        if not su_pos:
            self._colours: List[List[int]] = [list(range(self.n + 1))]
        else:
            blocks = (
                [list(range(0, su_pos[0]))]
                + [list(range(su_pos[i], su_pos[i + 1])) for i in range(len(su_pos) - 1)]
                + [list(range(su_pos[-1], self.n + 1))]
            )
            self._colours = blocks

        # Candidate lockings → flatten to NS5 indices (1‑based as in MMA)
        self._brane_lockings = self._enumerate_lockings()
        # Filter by S‑rule + positivity
        self._brane_lockings = [
            bl for bl in self._brane_lockings
            if self._left_table_max_len(bl) <= self.nf and self._black_nodes_ok(bl)
        ]
        # Keep only dominant lockings
        self._brane_lockings = self._dominant(self._brane_lockings)

    # ───── helper: ordered partitions ─────
    @staticmethod
    def _all_lockings(items: List[int]) -> List[List[List[int]]]:
        if not items:
            return [[]]
        if len(items) == 1:
            return [[[items[0]]]]
        out: List[List[List[int]]] = []
        for k in range(1, len(items) + 1):
            head = [items[:k]]
            for tail in MagneticQuiver._all_lockings(items[k:]):
                out.append(head + tail)
        # normalise (tuples are hashable; inner groups sorted)
        norm = [tuple(tuple(sorted(tuple(g))) for g in part) for part in out]
        uniq = sorted(set(norm), key=lambda t: (len(t), t))
        return [list(map(list, part)) for part in uniq]

    def _enumerate_lockings(self) -> List[List[List[int]]]:
        """
        Return list of *groups* per locking (not flattened).
        Each locking is a list of groups; each group is a list of 1‑based NS5 indices.
        Mirrors the Mathematica construction.
        """
        colour_indices = list(range(len(self._colours)))
        patts = self._all_lockings(colour_indices)
        # For each pattern (list of colour groups), build a list of NS5-index lists
        return [[[1 + idx for idx in self._colours[col]] for col in p] for p in patts]

    # ───── S‑rule tables & tests ─────
    def _brane_srule(self, g) -> List[int]:
        """S‑rule capacity for a *group* of NS5 indices (1‑based)."""
        # Accept either a single int or a list of ints.
        if isinstance(g, int):
            indices = [g]
        else:
            indices = list(g)
        slopes = [self._slopes[j - 1] for j in indices]
        return _cumul(_transpose_partition(_collapse(slopes)))

    def _left_table_max_len(self, bl: List[int]) -> int:
        return max((len(self._brane_srule(g)) for g in bl), default=0)

    def _table_left(self, bl) -> np.ndarray:
        """
        Left-brane table: one row per group (locking block), padded to nf entries.
        """
        rows: List[List[int]] = []
        for group in bl:
            rs = self._brane_srule(group)
            rows.append(list(reversed(np.pad(rs, (0, self.nf - len(rs))))))
        return np.array(rows, dtype=int)


    def _table_right(self, bl) -> np.ndarray:
        """
        Right-brane table: one row per group, n identical columns per row.
        In the Mathematica code each entry is sum_{j in group} slopesBranes[[j]].
        """
        rows = []
        for group in bl:
            # group is a list of 1-based NS5 indices
            if isinstance(group, int):  # defensive fallback
                s = sum(self._slopes[:group])
            else:
                s = sum(self._slopes[j - 1] for j in group)
            rows.append([s] * self.n)
        return np.array(rows, dtype=int)


    def _black_nodes_vec(self, bl: List[int]) -> np.ndarray:
        return np.array(self._total_left) - self._table_left(bl).sum(axis=0)

    def _black_nodes_ok(self, bl: List[int]) -> bool:
        return np.all(self._black_nodes_vec(bl) >= 0)

    # ───── dominance ─────
    @staticmethod
    def _compare_vertical(a: List[List[int]], b: List[List[int]]) -> bool:
        return all(any(set(s1).issubset(s2) for s1 in a) for s2 in b)

    def _compare_horizontal(self, a: List[int], b: List[int]) -> bool:
        ba, bb = self._black_nodes_vec(a), self._black_nodes_vec(b)
        return np.all(ba - bb >= 0) and np.any(ba != bb)

    def _dominant(self, cand: List[List[int]]) -> List[List[int]]:
        return [
            bl for bl in cand
            if not any(
                bl != other and self._compare_horizontal(other, bl) and self._compare_vertical(other, bl)
                for other in cand
            )
        ]

    # ───── build QuiverGraph ─────
    def _build_quiver_graph(self, bl: List[int]) -> QuiverGraph:
        nc = len(bl)
        ftL, ftR = self._table_left(bl), self._table_right(bl)

        # Safe index helper: return 0 when out of bounds
        def _safe(a: np.ndarray, idx: int) -> int:
            return int(a[idx]) if 0 <= idx < len(a) else 0

        def il(i: int, j: int) -> int:
            l1, l2 = ftL[i], ftL[j]
            s = sum(l1[k] * l2[k + 1] + l2[k] * l1[k + 1] - 2 * l1[k] * l2[k] for k in range(self.nf - 1))
            return int(s - l1[self.nf - 1] * l2[self.nf - 1])

        def ir(i: int, j: int) -> int:
            return int(ftR[j, :min(self.n, bl[i])].sum() + ftR[i, :min(self.n, bl[j])].sum())

        black = self._black_nodes_vec(bl)
        idx_pos = [k for k, b in enumerate(black) if b > 0]
        nb = len(idx_pos)
        size = nc + nb

        # Build symmetric intersection matrix M
        M = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(i + 1, size):
                if j < nc:  # colour × colour
                    M[i, j] = il(i, j) + ir(i, j)
                elif i < nc <= j:  # colour × black
                    k = idx_pos[j - nc]
                    M[i, j] = _safe(ftL[i], k + 1) - 2 * _safe(ftL[i], k) + _safe(ftL[i], k - 1)
                else:  # black × black
                    M[i, j] = 1 if abs(i - j) == 1 else 0
        M += M.T

        ranks = np.concatenate([np.ones(nc, dtype=int), black[idx_pos].astype(int)])
        balances = M @ ranks - 2 * ranks

        Q = QuiverGraph()
        for v, r in enumerate(ranks, 1):
            gp_type = 'colour' if v <= nc else 'black'
            flav_gauge = 'balanced' if int(balances[v - 1]) == 0 else 'unbalanced'
            Q.add_quiver_node(v, gp_type=gp_type, gp_rank=int(r), flav_gauge=flav_gauge)

        for i in range(size):
            for j in range(i + 1, size):
                mult = int(M[i, j])
                if mult:
                    Q.add_quiver_edge(i + 1, j + 1, multiplicity=mult)
        return Q

    # ───── public API ─────
    def magnetic_quivers(self) -> Generator[Tuple[List[int], QuiverGraph], None, None]:
        """Yield (brane_locking, QuiverGraph) for each dominant locking."""
        for bl in self._brane_lockings:
            yield bl, self._build_quiver_graph(bl)



mq = MagneticQuiver([2, 0, 2], [1, 2, 1], ["u", "s", "u"])
