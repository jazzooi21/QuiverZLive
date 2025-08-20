# Code for Hasse diagrams adapted from Mathematica implementation.

# Credit:
# Antoine Bourget, Marcus Sperling, Zhenghao Zhong
# Higgs branch RG-flows via Decay and Fission [https://arxiv.org/abs/2401.08757]
# Decay and Fission of Magnetic Quivers [https://arxiv.org/abs/2312.05304]
# https://www.antoinebourget.org/attachments/files/FissionDecay.nb

from __future__ import annotations

from .graph_model import QuiverGraph, QG_to_Mv, QG_to_Mv_idx

from typing import List, Tuple, Iterable, Sequence, Union, Optional, Dict
from collections import Counter
from fractions import Fraction
from functools import reduce
import operator
import math
import numpy as np

# ───────────────────────────── Types ─────────────────────────────

Vec = np.ndarray              # 1D int array
Mat = np.ndarray              # 2D int array
RankVector = np.ndarray       # alias for Vec
Leaf = List[Tuple[int, ...]]  # a multiset (list) of rank-vectors (tuples)
Quiver = Tuple[Mat, RankVector]


def _as_vec(x: Iterable[int]) -> Vec:
    return np.array(list(x), dtype=int)

def _as_tuple_vec(v: Vec) -> Tuple[int, ...]:
    return tuple(int(x) for x in v.tolist())

def _sum_leaf(leaf: Leaf) -> Vec:
    """Sum of vectors inside a leaf (each is a tuple)."""
    if not leaf:
        return np.zeros(0, dtype=int)
    dim = len(leaf[0])
    s = np.zeros(dim, dtype=int)
    for t in leaf:
        s += np.array(t, dtype=int)
    return s

def _unique_order(seq, key=lambda x: x):
    """Stable unique (first occurrence kept)."""
    seen = set()
    out = []
    for x in seq:
        k = key(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out

def _gcd_all(ints: Iterable[int]) -> int:
    ints = [abs(int(i)) for i in ints]
    return reduce(math.gcd, ints, 0)

def _is_vector(x) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))

# ───────────────────────── Mathematica → Python primitives ─────────────────────────

def balance(M: Mat, ranks: RankVector) -> Vec:
    """balance[quiver_] := quiver[[1]] . quiver[[2]];"""
    return (M @ ranks)

def clean(M: Mat, ranks: RankVector) -> Quiver:
    """
    clean[quiver_] keeps only nodes with strictly positive rank,
    removing corresponding rows/cols.
    """
    idx = [i for i, r in enumerate(ranks) if r > 0]
    if not idx:
        return (np.zeros((0, 0), dtype=int), np.zeros((0,), dtype=int))
    M2 = M[np.ix_(idx, idx)]
    r2 = ranks[idx]
    return (M2, r2)

def divide_diagonal(M: Mat) -> Mat:
    """divideDiagonal[mat_] := mat - 1/2*DiagonalMatrix[Diagonal[mat]];"""
    d = np.diag(M).astype(float) / 2.0
    return M.astype(float) - np.diag(d)

# (Plot omitted per your request.)

# ───────────────────────── lengthNodes / longNodes ─────────────────────────

def length_nodes(M: Mat) -> List[Fraction]:
    """
    lengthNodes[mat_]:
    Solve for ratios x_i/x_j = M[i,j]/M[j,i] when M[i,j] != 0, with x_1 = 1.
    Returns a list of Fractions (componentwise lengths).
    """
    n = M.shape[0]
    # adjacency over nonzero entries
    nbrs = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and M[i, j] != 0:
                nbrs[i].append(j)

    x: List[Optional[Fraction]] = [None] * n

    for start in range(n):
        if x[start] is not None:
            continue
        # new component
        x[start] = Fraction(1, 1)
        stack = [start]
        while stack:
            i = stack.pop()
            for j in nbrs[i]:
                # Need M[j,i] too; if it's zero, skip (or treat as 1 to avoid div-by-zero)
                if M[i, j] == 0 or M[j, i] == 0:
                    continue
                proposed = x[i] * Fraction(M[j, i], M[i, j])  # from M[i,j]/M[j,i] = x_i/x_j
                if x[j] is None:
                    x[j] = proposed
                    stack.append(j)
                else:
                    # Check consistency; if inconsistent, you may choose to average or ignore.
                    # Here we assert consistency in well-formed inputs.
                    if x[j] != proposed:
                        # choose a consistent normalization by scaling component,
                        # but since x[start]=1 is fixed, we just keep existing
                        pass

    # Fill any still-None entries with 1
    return [xx if xx is not None else Fraction(1, 1) for xx in x]

def long_nodes(M: Mat) -> List[int]:
    """Indices (0-based) attaining the maximum 'length' from length_nodes."""
    ln = length_nodes(M)
    mx = max(ln) if ln else Fraction(0, 1)
    return [i for i, v in enumerate(ln) if v == mx]

# ───────────────────────── Sign/zero predicates ─────────────────────────

def positiveQ(x: Union[int, float, Vec, Sequence[int]]) -> bool:
    """Abs[x] == x  (elementwise for vectors)."""
    if _is_vector(x):
        arr = np.array(x, dtype=int)
        return np.all(arr >= 0)
    return abs(x) == x

def zeroQ(x: Union[int, float, Vec, Sequence[int]]) -> bool:
    """Abs[x] == -Abs[x]  (elementwise for vectors ⇒ all zeros)."""
    if _is_vector(x):
        arr = np.array(x)
        return np.all(arr == 0)
    return abs(x) == -abs(x)  # True iff x == 0

def positiveQStrict(x: Union[int, float, Vec, Sequence[int]]) -> bool:
    """(Abs[x] == x) && Not[x == -x] ⇒ nonnegative and not identically zero."""
    if _is_vector(x):
        arr = np.array(x, dtype=int)
        return (np.all(arr >= 0) and np.any(arr > 0))
    return (abs(x) == x) and not (x == -x)

# ───────────────────────── Misc primitives ─────────────────────────

def number_common_elements(list1: Sequence[Tuple[int, ...]],
                           list2: Sequence[Tuple[int, ...]]) -> int:
    """With multiplicity, like Mathematica's approach."""
    c1 = Counter(list1)
    c2 = Counter(list2)
    return sum(min(c1[k], c2.get(k, 0)) for k in c1)

def elementarize(A: np.ndarray) -> np.ndarray:
    """
    elementarize[matrix_] : zero out A[i,j] if there exists k with A[i,k]==1 and A[k,j]==1.
    """
    A = np.array(A, dtype=int)
    n = A.shape[0]
    out = A.copy()
    for i in range(n):
        for j in range(n):
            if any((A[i, k] == 1 and A[k, j] == 1) for k in range(n)):
                out[i, j] = 0
    return out

# ───────────────────────── Instanton moduli test ─────────────────────────

def is_inst_mod_space(M: Mat, ranks: RankVector) -> bool:
    """
    True iff ∃ long-node i with ranks[i] > 0 such that:
      (M @ aux) * aux == 0  and  aux·M·e_i == gcd(aux)
    where aux = ranks - e_i.
    This matches the Mathematica "product equals 0" logic.
    """
    n = len(ranks)
    long_idx = long_nodes(M)

    def unit(i: int) -> Vec:
        e = np.zeros(n, dtype=int)
        e[i] = 1
        return e

    for i in long_idx:
        if ranks[i] <= 0:
            continue
        e = unit(i)
        aux = ranks - e
        gcd = _gcd_all(aux.tolist())
        v = (M @ aux) * aux
        ok_kernel_on_support = zeroQ(v)
        intersection = int(np.dot(aux, M[:, i]))
        ok_intersection = (intersection == gcd)
        if ok_kernel_on_support and ok_intersection:
            # Found at least one passing node ⇒ product would be 0
            return True
    # No passing nodes ⇒ product would be 1
    return False

# ───────────────────────── Hasse auxiliaries ─────────────────────────

def descends(leaf1: Leaf, leaf2: Leaf) -> bool:
    """
    Port of descends[leaf1_, leaf2_].
    Each leaf is a multiset (list) of integer vectors (tuples).
    """
    L1 = len(leaf1)
    L2 = len(leaf2)
    common = number_common_elements(leaf1, leaf2)
    # vector totals
    t1 = _sum_leaf(leaf1)
    t2 = _sum_leaf(leaf2)
    dt = t1 - t2

    cond1 = (L2 == L1) and (common == L1 - 1) and positiveQ(dt)
    cond2 = (L2 == L1 - 1) and (common == L1 - 1) and positiveQ(dt)
    cond3 = (L2 == L1 + 1) and (common == L1 - 1) and positiveQ(dt)
    return cond1 or cond2 or cond3

def multiplicity_elementary_transition(leaf1: Leaf, leaf2: Leaf) -> int:
    """
    Port of multiplicityElementaryTransition[leaf1_, leaf2_]:
    Tally leaf1 and leaf2; find the (elem, count1) in leaf1 whose exact
    pair is not present in leaf2's tally; return count1.
    """
    c1 = Counter(leaf1)
    c2 = Counter(leaf2)
    # emulate Mathematica's "MemberQ[t2, {elem, count1}]"
    for elem, cnt1 in c1.items():
        if c2.get(elem, None) != cnt1:
            return cnt1
    # If nothing differs, multiplicity 0 (or raise).
    return 0

# ───────────────────────── Core search routines ─────────────────────────

def level1_descendents(M: Mat, ranks: RankVector) -> List[Vec]:
    """
    level1Descendents[mat_, ranks_]:
      Start from ranks - e_k for each k with ranks[k]>0.
      While any v has (M @ v) with negative entries, decrease v at those bad nodes.
      Keep only v >= 0 and (M @ v) >= 0.
    """
    n = len(ranks)

    def e(k: int) -> Vec:
        v = np.zeros(n, dtype=int)
        v[k] = 1
        return v

    good: List[Vec] = []
    for k in range(n):
        if ranks[k] <= 0:
            continue
        list_good = [ranks - e(k)]
        # Relax until all satisfy (M @ v) >= 0
        while not all(positiveQ(M @ v) for v in list_good):
            aux1 = [v for v in list_good if positiveQ(M @ v)]
            aux2 = [v for v in list_good if not positiveQ(M @ v)]
            # For each bad v, find coordinates where (M @ v)[bad] < 0 and decrement there
            newly = []
            for v in aux2:
                bad_nodes = [idx for idx, val in enumerate(M @ v) if val < 0]
                for bad in bad_nodes:
                    w = v - e(bad)
                    newly.append(w)
            list_good = _unique_order(aux1 + newly, key=lambda vv: _as_tuple_vec(vv))
        # Keep only v >= 0
        good.extend(v for v in list_good if positiveQ(v))
    # Deduplicate
    good = _unique_order(good, key=lambda vv: _as_tuple_vec(vv))
    return good

def all_good_subquivers(M: Mat, ranks: RankVector) -> List[Vec]:
    """
    allGoodSubquivers[{M, ranks}] via repeated level1_descendents until closure.
    Returns a list of vectors, sorted by total sum.
    """
    def apply_desc(list_of_vs: List[Vec]) -> List[Vec]:
        out: List[Vec] = []
        for v in list_of_vs:
            out.extend(level1_descendents(M, v))
        # dedup
        return _unique_order(out, key=lambda vv: _as_tuple_vec(vv))

    k = 0
    desc: Dict[int, List[Vec]] = {}
    desc[0] = [ranks.copy()]
    while desc[k]:
        k += 1
        desc[k] = apply_desc(desc[k - 1])

    all_vs: List[Vec] = []
    for j in range(0, k + 1):
        all_vs.extend(desc.get(j, []))
    # Unique by tuple, then sort by sum of entries
    all_vs = _unique_order(all_vs, key=lambda vv: _as_tuple_vec(vv))
    all_vs.sort(key=lambda vv: int(np.sum(vv)))
    return all_vs

# ───────────────────────── Fission / Hasse ─────────────────────────

def fission_decay(qg: QuiverGraph):

    M, ranks, idx2node = QG_to_Mv_idx(qg)
    M = M.copy()
    # Set diagonal entries of M to -2
    # no self-loops
    np.fill_diagonal(M, -2)


    """
    Port of fissionDecay[quiv_]. Returns:
      - all_leaves: List[Leaf]  (each leaf is a sorted list of rank-vectors (tuples))
      - hasse_mult: np.ndarray  (adjacency with multiplicities)
    """
    n = len(ranks)
    zero = np.zeros(n, dtype=int)

    sol0 = all_good_subquivers(M, ranks)

    # sol1 = Select[sol0, Not[# . mat . # >= 0 && Total[#] == 1] &];
    def qform_nonneg_and_sum1(v: Vec) -> bool:
        return int(v @ (M @ v)) >= 0 and int(np.sum(v)) == 1
    sol1 = [v for v in sol0 if not qform_nonneg_and_sum1(v)]

    # sol = Select[sol1, Not[isInstModSpace[{mat, #}]] &];
    sol = [v for v in sol1 if not is_inst_mod_space(M, v)]

    # Prepare sets for fast membership
    sol_tuples = [_as_tuple_vec(v) for v in sol]
    sol_set = set(sol_tuples)

    # chunks[0] = { {0*ranks} }
    chunks: Dict[int, List[Leaf]] = {}
    chunks[0] = [[_as_tuple_vec(zero)]]


    # chunks[1] = { {v} | v in Drop[sol, 1] }
    # In Mathematica, SortBy[...] tends to put the zero vector first, so Drop[1] skips it.
    # We'll skip zero explicitly:
    chunks[1] = [[tv] for tv in sol_tuples if any(tv)]  # any(tv) => nonzero

    leaves: Dict[int, List[Leaf]] = {}
    leaves[1] = chunks[0] + chunks[1]

    j = 1
    while len(leaves[j]) > len(leaves.get(j - 1, [])):
        next_chunks: List[Leaf] = []
        for v_leaf in chunks[j]:
            sum_v = _sum_leaf(v_leaf)  # vector
            for s in sol_tuples:
                s_vec = np.array(s, dtype=int)
                w = s_vec - sum_v
                if positiveQStrict(w) and positiveQ(M @ w):
                    w_t = _as_tuple_vec(w)
                    if w_t in sol_set:
                        new_leaf = sorted(v_leaf + [w_t])  # canonical form
                        next_chunks.append(new_leaf)
        # dedup leaves
        next_chunks = _unique_order(next_chunks, key=lambda lf: tuple(lf))
        chunks[j + 1] = next_chunks
        leaves[j + 1] = _unique_order(leaves[j] + next_chunks, key=lambda lf: tuple(lf))
        j += 1

    all_leaves: List[Leaf] = leaves[j]

    # Build raw Hasse adjacency by descends, then elementarize
    L = len(all_leaves)
    hasse = np.zeros((L, L), dtype=int)
    for i in range(L):
        for k in range(L):
            if descends(all_leaves[i], all_leaves[k]):
                hasse[i, k] = 1

    hasse_elem = elementarize(hasse)

    # Multiplicity
    hasse_mult = np.zeros((L, L), dtype=int)
    for i in range(L):
        for k in range(L):
            if hasse_elem[i, k] == 1:
                mult = multiplicity_elementary_transition(all_leaves[i], all_leaves[k])
                hasse_mult[i, k] = mult

    return M, ranks, idx2node, all_leaves, hasse_mult