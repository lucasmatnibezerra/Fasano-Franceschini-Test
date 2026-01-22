from typing import Sequence
import os
import numpy as np

USE_NUMBA_RANGE = os.environ.get("FF_RANGE_IMPL", "").lower() == "numba"

from rtree import index  # mandatory dependency
from numba import njit  # mandatory dependency

HAVE_RTREE = True


@njit(fastmath=True, cache=True)
def _count_in_range_numba(points, lower, upper, with_lower, with_upper):
    n, d = points.shape
    c = 0
    for i in range(n):
        ok = True
        for j in range(d):
            x = points[i, j]
            lo = lower[j]
            hi = upper[j]
            if with_lower[j] == 0:
                if x <= lo:
                    ok = False
                    break
            else:
                if x < lo:
                    ok = False
                    break
            if with_upper[j] == 0:
                if x >= hi:
                    ok = False
                    break
            else:
                if x > hi:
                    ok = False
                    break
        if ok:
            c += 1
    return c


@njit(fastmath=True, cache=True)
def _points_mask_in_range(points, lower, upper, with_lower, with_upper):
    n, d = points.shape
    mask = np.ones(n, dtype=np.uint8)
    for j in range(d):
        lo = lower[j]
        hi = upper[j]
        wl = with_lower[j]
        wu = with_upper[j]
        for i in range(n):
            x = points[i, j]
            if wl == 0:
                if x <= lo:
                    mask[i] = 0
                    continue
            else:
                if x < lo:
                    mask[i] = 0
                    continue
            if wu == 0:
                if x >= hi:
                    mask[i] = 0
                    continue
            else:
                if x > hi:
                    mask[i] = 0
                    continue
    return mask


class _NumbaRangeTree:
    def __init__(self, points: Sequence[Sequence[float]]):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0:
            raise ValueError("Need at least one point (n,d)")
        self.points = pts
        self.n, self.dimension = pts.shape

    def _prep_bounds(self, lower, upper, with_lower, with_upper):
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        d = self.dimension
        if lower.shape[0] != d or upper.shape[0] != d:
            raise ValueError("Dimensão de lower/upper incorreta")
        if with_lower is None:
            with_lower = np.ones(d, dtype=np.uint8)
        else:
            with_lower = np.asarray(with_lower, dtype=np.uint8)
        if with_upper is None:
            with_upper = np.ones(d, dtype=np.uint8)
        else:
            with_upper = np.asarray(with_upper, dtype=np.uint8)
        return lower, upper, with_lower, with_upper

    def count_in_range(self, lower, upper, with_lower=None, with_upper=None):
        lower, upper, wl, wu = self._prep_bounds(lower, upper, with_lower, with_upper)
        return int(_count_in_range_numba(self.points, lower, upper, wl, wu))

    def points_in_range(self, lower, upper, with_lower=None, with_upper=None):
        lower, upper, wl, wu = self._prep_bounds(lower, upper, with_lower, with_upper)
        mask = _points_mask_in_range(self.points, lower, upper, wl, wu)
        return [tuple(row) for row in self.points[mask == 1]]


class _RTreeRangeTree:
    def __init__(self, points: Sequence[Sequence[float]]):
        pts = [tuple(pt) for pt in points]
        if len(pts) == 0:
            raise ValueError("Need at least one point to build RangeTree")
        self.points = pts
        self.n = len(pts)
        self.dimension = len(pts[0])
        prop = index.Property()
        prop.dimension = self.dimension
        self.idx = index.Index(properties=prop)
        for i, pt in enumerate(self.points):
            bounds = tuple(pt) + tuple(pt)
            self.idx.insert(i, bounds)

    @staticmethod
    def _adjust_bounds(lower, upper, with_lower, with_upper):
        lb = list(lower)
        ub = list(upper)
        if with_lower:
            for i, ok in enumerate(with_lower):
                if not ok:
                    lb[i] += 1e-12
        if with_upper:
            for i, ok in enumerate(with_upper):
                if not ok:
                    ub[i] -= 1e-12
        return lb, ub

    def count_in_range(self, lower, upper, with_lower=None, with_upper=None):
        lb, ub = self._adjust_bounds(lower, upper, with_lower, with_upper)
        bounds = tuple(lb + ub)
        return sum(1 for _ in self.idx.intersection(bounds))

    def points_in_range(self, lower, upper, with_lower=None, with_upper=None):
        lb, ub = self._adjust_bounds(lower, upper, with_lower, with_upper)
        bounds = tuple(lb + ub)
        return [self.points[i] for i in self.idx.intersection(bounds)]


class RangeTree(
    _NumbaRangeTree if (USE_NUMBA_RANGE or not HAVE_RTREE) else _RTreeRangeTree
):
    pass


class NaiveRangeCounter(_NumbaRangeTree):
    pass
