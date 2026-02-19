import argparse
import math
from typing import Dict, List, Tuple

import numpy as np

# The three verification routines below avoid full big-integer arithmetic during
# the bulk sieve by working modulo two independent primes MOD1, MOD2 < 2^32 and
# packing residues into a single uint64 key via key = r1 | (r2 << 32).  This is
# a standard double-hashing filter: any pair with R_m(x) = R_n(y) will match,
# and a spurious collision requires R_m(x) congruent to R_n(y) (mod MOD1 * MOD2), which
# occurs with probability ~2^{-64} per pair.  Exact arithmetic confirms every
# candidate the sieve admits, allowing for considerably faster computation.

MOD1 = 4294967291
MOD2 = 4294967279


def repunit(a: int, t: int) -> int:
    return (pow(a, t) - 1) // (a - 1)


def build_repunit_tables(bases: np.ndarray, max_t: int, mod: int) -> np.ndarray:
    b = bases.astype(np.uint64)
    m = np.uint64(mod)
    row = np.ones_like(b) % m
    out = np.empty((max_t, b.size), dtype=np.uint64)
    out[0] = row
    for i in range(1, max_t):
        row = (row * b + 1) % m
        out[i] = row
    return out


def build_y_n_map(
    y_min: int, y_max: int, n_min: int, n_max: int
) -> Dict[int, List[Tuple[int, int]]]:
    ys = np.arange(y_min, y_max + 1, dtype=np.uint64)
    t1 = build_repunit_tables(ys, n_max, MOD1)
    t2 = build_repunit_tables(ys, n_max, MOD2)

    result: Dict[int, List[Tuple[int, int]]] = {}
    shift = np.uint64(32)
    for n in range(n_min, n_max + 1):
        keys = t1[n - 1] | (t2[n - 1] << shift)
        for i, y in enumerate(ys):
            result.setdefault(int(keys[i]), []).append((int(y), n))
    return result


def find_nondiv_solutions(
    x_min: int,
    x_max: int,
    m_min: int,
    m_max: int,
    y_n_map: Dict[int, List[Tuple[int, int]]],
) -> List[Tuple[int, int, int, int]]:
    xs = np.arange(x_min, x_max + 1, dtype=np.uint64)
    t1 = build_repunit_tables(xs, m_max, MOD1)
    t2 = build_repunit_tables(xs, m_max, MOD2)

    solutions: List[Tuple[int, int, int, int]] = []
    shift = np.uint64(32)
    for m in range(m_min, m_max + 1):
        keys = t1[m - 1] | (t2[m - 1] << shift)
        for i, x_u in enumerate(xs):
            candidates = y_n_map.get(int(keys[i]))
            if not candidates:
                continue
            x = int(x_u)
            for y, n in candidates:
                k = n - 1
                if not (x < y and n < m):
                    continue
                if (m - 1) % k == 0:
                    continue
                if repunit(x, m) == repunit(y, n):
                    solutions.append((x, y, m, n))
    return solutions


def check_nondiv_regime() -> List[Tuple[int, int, int, int]]:
    y_n_map = build_y_n_map(3, 316, 4, 503)
    return find_nondiv_solutions(2, 315, 5, 4197, y_n_map)


def check_div_finite_cases() -> List[Tuple[int, int, int, int]]:
    cases: List[Tuple[int, int]] = []

    for k in range(13, 31):
        cases.append((k, 3))

    for k, y_max in ((5, 5), (8, 6), (10, 7), (12, 7)):
        for y in range(3, y_max + 1):
            cases.append((k, y))

    for k in (7, 9):
        cases.append((k, 3))

    solutions: List[Tuple[int, int, int, int]] = []
    for k, y in cases:
        n = k + 1
        target = repunit(y, n)
        for x in range(2, y):
            val = 1
            m = 1
            while val < target:
                m += 1
                val = val * x + 1
                if m <= n:
                    continue
                if (m - 1) % k != 0:
                    continue
                if val == target:
                    solutions.append((x, y, m, n))
                    break
                if m > 512:
                    break
    return solutions


def largest_y_with_sqrt_lt_6log2(limit: int = 20000) -> int:
    best = 0
    for y in range(2, limit + 1):
        if math.sqrt(y) < 6.0 * math.log(y, 2):
            best = y
    return best


def check_n7_div_regime() -> List[Tuple[int, int, int, int]]:
    y_max = largest_y_with_sqrt_lt_6log2(20000)
    if y_max != 5575:
        raise RuntimeError(f"unexpected threshold y_max={y_max}")

    n = 7
    m_max = int(math.floor(math.log(7 * (y_max**6), 2))) + 1
    if m_max != 78:
        raise RuntimeError(f"unexpected m_max={m_max}")

    valid_ms = [6 * t + 1 for t in range(2, (m_max - 1) // 6 + 1)]

    by_value: Dict[int, List[Tuple[int, int]]] = {}
    for x in range(2, y_max):
        for m in valid_ms:
            by_value.setdefault(repunit(x, m), []).append((x, m))

    solutions: List[Tuple[int, int, int, int]] = []
    for y in range(3, y_max + 1):
        target = repunit(y, n)
        for x, m in by_value.get(target, []):
            if x < y and n < m and (m - 1) % 6 == 0:
                solutions.append((x, y, m, n))
    return solutions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-nondiv", action="store_true")
    ap.add_argument("--skip-div-finite", action="store_true")
    ap.add_argument("--skip-n7-div", action="store_true")
    args = ap.parse_args()

    if not args.skip_nondiv:
        sols = check_nondiv_regime()
        print(f"nondiv (y<=316, n<=503, m<=4197): {len(sols)}")
        for s in sols:
            print("  ", s)

    if not args.skip_div_finite:
        sols = check_div_finite_cases()
        print(f"div finite (k,y list): {len(sols)}")
        for s in sols:
            print("  ", s)

    if not args.skip_n7_div:
        sols = check_n7_div_regime()
        print(f"n=7, 6|(m-1) (y<=5575, m<=78): {len(sols)}")
        for s in sols:
            print("  ", s)


if __name__ == "__main__":
    main()