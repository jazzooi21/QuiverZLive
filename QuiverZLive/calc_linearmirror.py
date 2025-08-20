# Code for mixed unitary linear 3d mirrors adapted from Mathematica implementation.

# Credit:
# Antoine Bourget, Julius Grimminger, Amihay Hanany, Rudolph Kalveks, Zhenghao Zhong
# #Higgs Branches of U/SU Quivers via Brane Locking [https://arxiv.org/pdf/2111.04745]
# http://www.antoinebourget.org/attachments/files/SUquivers.nb

from __future__ import annotations

from typing import List, Sequence, Tuple, Generator

import numpy as np

from graph_model import QuiverGraph, Mv_to_QG, QG_to_Mv

# ────────────────────────── partition helpers ───────────────────────────

def _transpose_partition(part: Sequence[int]) -> List[int]:
    if not part:
        return []
    m = max(part)
    return [sum(1 for x in part if x >= i) for i in range(1, m + 1)]


def _cumul(part: Sequence[int]) -> List[int]:
    return [sum(part[i:]) for i in range(len(part))]


def _collapse_pair(a: int, b: int) -> List[int]:
    """Integer-preserving HW pair collapse."""
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
    flavor_ranks : list[int]
        Fundamental hypermultiplet multiplicities attached to each gauge node.
    gauge_ranks  : list[int]
        Ranks of the gauge groups in the electric quiver.
    gauge_type   : Sequence[str]
        Gauge‑group types, e.g. ["u", "s", "u"]. Use "s" for **SU**, any other
        string for **U**. Case‑insensitive.
    """

    # ───── construction ─────
    def __init__(self,
                 flavor_ranks: Sequence[int],
                 gauge_ranks : Sequence[int],
                 gauge_type  : Sequence[str]):

        if len(flavor_ranks) != len(gauge_ranks):
            raise ValueError("flavor_ranks and gauge_ranks must have equal length")
        if len(gauge_type) != len(gauge_ranks):
            raise ValueError("gauge_type must have one entry per gauge node")

        self.flavor_ranks = list(map(int, flavor_ranks))
        self.gauge_ranks  = list(map(int, gauge_ranks))
        # normalise to lowercase strings
        self.gauge_type   = [str(t).lower() for t in gauge_type]

        self.nf = sum(self.flavor_ranks)  # total flavours
        self.n  = len(self.gauge_ranks)   # number of gauge factors

        # ------------------------------------------------------------------
        # Pre‑computations
        # ------------------------------------------------------------------
        gn_ext = [0] + self.gauge_ranks + [0]
        fn_ext = [0] + self.flavor_ranks + [0]

        # Slopes for the (n+1) NS5 intervals
        self._slopes = [
            gn_ext[i] - gn_ext[i + 1] + sum(fn_ext[j] for j in range(i + 1, self.n + 2))
            for i in range(self.n + 1)
        ]
        # Store SU boundaries (0-based between NS5 intervals) and prefix sums of slopes
        self._su_boundaries = {i for i, t in enumerate(self.gauge_type) if t == 's'}  # 0-based cuts
        self._prefix_slopes = np.cumsum(self._slopes).astype(int)  # P[g] = sum_{j<=g} slopes[j]

        # Total left branes available per flavour index
        self._total_left = list(reversed(_cumul(_transpose_partition(_cumul(self.flavor_ranks)))))

        # Colour blocks induced by U/SU pattern
        su_pos = [i for i, t in enumerate(self.gauge_type) if t == 's']
        if not su_pos:
            self._colours: List[List[int]] = [list(range(self.n + 1))]
        else:
            # split NS5 indices 0..n at boundaries (su_index + 1)
            cuts = [0] + [p + 1 for p in su_pos] + [self.n + 1]
            blocks = [list(range(cuts[i], cuts[i + 1])) for i in range(len(cuts) - 1)]
            self._colours = [b for b in blocks if b]  # sanity; shouldn't be empty now

            

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
        """All set partitions of `items` (order-insensitive up to canonicalization)."""
        def rec(lst: List[int]) -> List[List[List[int]]]:
            if not lst:
                return [[]]
            x, *rest = lst
            parts = []
            for p in rec(rest):
                # put x into each existing block
                for i in range(len(p)):
                    q = [blk[:] for blk in p]
                    q[i].append(x)
                    parts.append(q)
                # or start a new singleton block
                parts.append([[x]] + [blk[:] for blk in p])
            # canonicalize: sort inside blocks; then sort blocks lexicographically
            canon = []
            for p in parts:
                gp = [tuple(sorted(g)) for g in p]
                canon.append(tuple(sorted(gp)))
            uniq = sorted(set(canon))
            return [ [list(g) for g in part] for part in uniq ]
        return rec(items)

    def _enumerate_lockings(self) -> List[List[List[int]]]:
        colour_indices = list(range(len(self._colours)))
        patts = self._all_lockings(colour_indices)

        # Each pattern p is a list of groups (lists of colour indices).
        # Map each group to the union of its NS5 indices (1-based), and KEEP the grouping.
        lockings: List[List[List[int]]] = []
        for p in patts:
            groups: List[List[int]] = []
            for group in p:
                # union of NS5 indices from all colours in this group, 0-based
                ns5 = sorted({idx for c in group for idx in self._colours[c]})
                groups.append(ns5)
            lockings.append(groups)
        return lockings
    
    # ───── S‑rule tables & tests ─────
    def _brane_srule(self, g: int) -> List[int]:
        # g is 0-based; index slopes directly with g
        return _cumul(_transpose_partition(_collapse([self._slopes[g]])))

    def _brane_srule_group(self, G: List[int]) -> List[int]:
        """Left-table vector for a GROUP: collapse the entire slope vector at once (0-based)."""
        if not G:
            return []
        seq = [self._slopes[g] for g in G]   # g in 0..n
        return _cumul(_transpose_partition(_collapse(seq)))


    def _left_table_max_len(self, bl: List[List[int]]) -> int:
        return max((len(self._brane_srule_group(G)) for G in bl), default=0)

    def _table_left(self, bl: List[List[int]]) -> np.ndarray:
        rows: List[List[int]] = []
        for G in bl:
            rs = self._brane_srule_group(G)
            rows.append(list(reversed(np.pad(rs, (0, self.nf - len(rs))))))
        return np.array(rows, dtype=int)

    def _table_right(self, bl: List[List[int]]) -> np.ndarray:
        # ftRight[group, i] = sum_{j in group, j>i} slopes[j], with i = 0..n-1 and j = 0..n
        ft = np.zeros((len(bl), self.n), dtype=int)
        for g_idx, G in enumerate(bl):
            for i in range(self.n):            # i = 0..n-1
                s = 0
                for j in G:                     # j = 0..n
                    if j > i:
                        s += int(self._slopes[j])
                ft[g_idx, i] = s
        return ft


    def _black_nodes_vec(self, bl: List[List[int]]) -> np.ndarray:
        return np.array(self._total_left) - self._table_left(bl).sum(axis=0)

    def _black_nodes_ok(self, bl: List[List[int]]) -> bool:
        return np.all(self._black_nodes_vec(bl) >= 0)

    # ───── dominance ─────
    @staticmethod
    def _compare_vertical(a: List[List[int]], b: List[List[int]]) -> bool:
        return all(any(set(s1).issubset(s2) for s1 in a) for s2 in b)

    def _compare_horizontal(self, a: List[List[int]], b: List[List[int]]) -> bool:
        ba, bb = self._black_nodes_vec(a), self._black_nodes_vec(b)
        return np.all(ba - bb >= 0) and np.any(ba != bb)

    @staticmethod
    def _refines(a: List[List[int]], b: List[List[int]]) -> bool:
        # every group in a is contained in some group in b
        a_sets = [set(x) for x in a]
        b_sets = [set(y) for y in b]
        return all(any(x.issubset(y) for y in b_sets) for x in a_sets)

    def _dominant(self, cand: List[List[List[int]]]) -> List[List[List[int]]]:
        def dominates(A: List[List[int]], B: List[List[int]]) -> bool:
            # horizontal: black nodes vector (allow equality)
            BA = self._black_nodes_vec(A)
            BB = self._black_nodes_vec(B)
            horiz_ok = np.all(BA - BB >= 0)  # allow equality

            # vertical: A refines B (A is finer), and strictly finer if tie in horizontals
            vert_ok = self._refines(A, B)

            # strict improvement if either strictly better horizontally OR strictly finer at tie
            strict = (np.any(BA != BB)) or (len(A) > len(B))
            return horiz_ok and vert_ok and strict

        return [bl for bl in cand if not any(other is not bl and dominates(other, bl) for other in cand)]
    
    
    # ----- SU helpers ------
    def _right_link_endpoint(self, G: List[int]) -> int:
        # right-link measured at the group's rightmost NS5 (1-based)
        return int(self._prefix_slopes[max(G)]) if G else 0

    def _has_s_wall_between(self, Gi: List[int], Gj: List[int]) -> bool:
        """True if there's at least one SU boundary between colour groups Gi (left) and Gj (right).
        NS5 indices are 1-based; a boundary b sits between NS5 b and b+1."""
        ai, bi = min(Gi), max(Gi)
        aj, bj = min(Gj), max(Gj)
        if aj < ai:  # ensure Gi is left of Gj
            ai, bi, aj, bj = aj, bj, ai, bi
        # SU boundary b splits between b and b+1.
        # Separation holds iff there exists b with bi <= b < aj (note the <= on bi).
        return any(bi <= b < aj for b in self._su_boundaries)
    

    # ───── build QuiverGraph ─────
    def _right_link(self, G: List[int]) -> int:
        # linking number on the right for a group G
        return sum(sum(self._slopes[:g + 1]) for g in G)
        # return sum(sum(self._slopes[j - 1] for j in range(1, g + 1)) for g in G)

    def _build_quiver_graph(self, bl: List[List[int]]) -> QuiverGraph:
        nc = len(bl)
        ftL, ftR = self._table_left(bl), self._table_right(bl)

        def _safe(a: np.ndarray, idx: int) -> int:
            return int(a[idx]) if 0 <= idx < len(a) else 0

        def il(i: int, j: int) -> int:
            l1, l2 = ftL[i], ftL[j]
            s = sum(l1[k]*l2[k+1] + l2[k]*l1[k+1] - 2*l1[k]*l2[k] for k in range(self.nf - 1))
            return int(s - l1[self.nf - 1]*l2[self.nf - 1])

        def ir(i: int, j: int) -> int:
            s = 0
            for ii in bl[i]:
                if ii < self.n:                 # ii in 0..n-1
                    s += int(ftR[j, ii])        # index ftR at ii
            for jj in bl[j]:
                if jj < self.n:
                    s += int(ftR[i, jj])
            return s



        black = self._black_nodes_vec(bl)
        idx_pos = [k for k, b in enumerate(black) if b > 0]
        nb = len(idx_pos)
        size = nc + nb

        M = np.zeros((nc + np.count_nonzero(self._black_nodes_vec(bl) > 0), nc + np.count_nonzero(self._black_nodes_vec(bl) > 0)), dtype=int)
        for i in range(nc + np.count_nonzero(self._black_nodes_vec(bl) > 0)):
            for j in range(i + 1, nc + np.count_nonzero(self._black_nodes_vec(bl) > 0)):
                if j < nc:                       # colour × colour
                    M[i, j] = il(i, j) + ir(i, j)
                elif i < nc <= j:                # colour × black
                    k = idx_pos[j - nc]
                    M[i, j] = _safe(ftL[i], k + 1) - 2*_safe(ftL[i], k) + _safe(ftL[i], k - 1)
                else:                            # black × black
                    M[i, j] = 1 if abs(i - j) == 1 else 0
        M += M.T


        ranks = np.concatenate([np.ones(nc, dtype=int), black[idx_pos].astype(int)])

        Q = Mv_to_QG(M, ranks)

        return Q, M, ranks

    # ───── public API ─────
    def magnetic_quivers(self) -> Generator[Tuple[List[List[int]], QuiverGraph], None, None]:
        """Yield (brane_locking, QuiverGraph) for each dominant locking."""
        for bl in self._brane_lockings:
            Q, _, _ = self._build_quiver_graph(bl)
            yield bl, Q

    def magnetic_quivers_full(self) -> Generator[
        Tuple[List[List[int]], QuiverGraph, np.ndarray, np.ndarray], None, None
    ]:
        """Yield (brane_locking, QuiverGraph, intersection_matrix M, ranks)."""
        for bl in self._brane_lockings:
            Q, M, ranks = self._build_quiver_graph(bl)
            yield bl, Q, M, ranks   