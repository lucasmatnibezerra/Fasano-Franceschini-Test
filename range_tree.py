# range_tree.py

from rtree import index

class RangeTree:
    """
    Orthogonal range query structure using R-tree index.
    Supports counting and reporting points in axis-aligned rectangles.
    """

    def __init__(self, points):
        """
        Initialize the tree.

        Parameters
        ----------
        points : array-like of shape (n, d)
            Lista de n pontos em d dimensões.
        """
        self.points = [tuple(pt) for pt in points]
        self.n = len(self.points)
        if self.n == 0:
            raise ValueError("Need at least one point to build RangeTree")
        self.dimension = len(self.points[0])

        # Configura R-tree para d dimensões
        prop = index.Property()
        prop.dimension = self.dimension
        self.idx = index.Index(properties=prop)

        # Insere cada ponto como caixa degenerada (min=max)
        for i, pt in enumerate(self.points):
            bounds = tuple(pt) + tuple(pt)
            self.idx.insert(i, bounds)

    def count_in_range(self, lower, upper, with_lower=None, with_upper=None):
        """
        Conta quantos pontos estão dentro do hiper-retângulo definido.

        Parameters
        ----------
        lower : sequence of length d
            Limites inferiores.
        upper : sequence of length d
            Limites superiores.
        with_lower : sequence of bool, optional
            Se False em i, trata lower[i] como > (strict). Por padrão, é <= inclusive.
        with_upper : same as with_lower for upper, optional.

        Returns
        -------
        count : int
            Número de pontos no retângulo.
        """
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

        bounds = tuple(lb + ub)
        return sum(1 for _ in self.idx.intersection(bounds))

    def points_in_range(self, lower, upper, with_lower=None, with_upper=None):
        """
        Retorna a lista de pontos dentro do hiper-retângulo.

        Parameters
        ----------
        lower, upper, with_lower, with_upper : same as count_in_range

        Returns
        -------
        pts : list of tuple
            Pontos encontrados.
        """
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

        bounds = tuple(lb + ub)
        return [self.points[i] for i in self.idx.intersection(bounds)]


class NaiveRangeCounter:
    """
    Implementação ingênua (force‐brute) para contagem de pontos em retângulo.
    """

    def __init__(self, points):
        """
        Parameters
        ----------
        points : array-like of shape (n, d)
            Pontos de dados.
        """
        self.points = [tuple(pt) for pt in points]

    def count_in_range(self, lower, upper, with_lower=None, with_upper=None):
        """
        Conta quantos pontos caem no retângulo (mesma interface).
        """
        count = 0
        for pt in self.points:
            ok = True
            for i, coord in enumerate(pt):
                lo, hi = lower[i], upper[i]
                if with_lower and not with_lower[i]:
                    if coord <= lo:
                        ok = False
                        break
                else:
                    if coord < lo:
                        ok = False
                        break
                if with_upper and not with_upper[i]:
                    if coord >= hi:
                        ok = False
                        break
                else:
                    if coord > hi:
                        ok = False
                        break
            if ok:
                count += 1
        return count

    def points_in_range(self, lower, upper, with_lower=None, with_upper=None):
        """
        Retorna os pontos dentro do retângulo (mesma interface).
        """
        result = []
        for pt in self.points:
            ok = True
            for i, coord in enumerate(pt):
                lo, hi = lower[i], upper[i]
                if with_lower and not with_lower[i]:
                    if coord <= lo:
                        ok = False
                        break
                else:
                    if coord < lo:
                        ok = False
                        break
                if with_upper and not with_upper[i]:
                    if coord >= hi:
                        ok = False
                        break
                else:
                    if coord > hi:
                        ok = False
                        break
            if ok:
                result.append(pt)
        return result
