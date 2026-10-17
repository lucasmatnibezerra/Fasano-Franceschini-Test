# -*- coding: utf-8 -*-
"""
Gerador de 5 datasets 8D/40k com drifts de correlação (River-friendly):
1) Copula Gaussiana (5 janelas graduais)
2) Mistura Linear com Rotação (5 abruptos)
3) Grafo de Dependência (Precisão Esparsa) com rewiring (5 abruptos)
4) Correlação dinâmica tipo DCC (5 incrementais)
5) HMM de regimes (5 abruptos com recorrência)
Saídas: 5 CSVs + impressão de ground truth
"""

from __future__ import annotations
import math, numpy as np, pandas as pd
from typing import List, Dict, Iterator, Optional, Tuple, Literal
from dataclasses import dataclass

try:
    # River é usado apenas como base leve para metadata; para benchmarks
    # permitimos fallback local quando o pacote não estiver instalado.
    from river.datasets.base import SyntheticDataset  # type: ignore
except Exception:  # pragma: no cover

    class SyntheticDataset:  # type: ignore
        def __init__(self, task=None, n_features=None, n_samples=None):
            self.task = task
            self.n_features = n_features
            self.n_samples = n_samples


Transition = Literal["abrupt", "gradual", "incremental"]

# ====================== utilidades numéricas ======================


def nearest_pd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    B = (A + A.T) / 2
    vals, vecs = np.linalg.eigh(B)
    vals[vals < eps] = eps
    return vecs @ np.diag(vals) @ vecs.T


def cholesky_factor_from_psd(R: np.ndarray) -> np.ndarray:
    R = nearest_pd((R + R.T) / 2)
    vals, vecs = np.linalg.eigh(R)
    vals[vals < 1e-12] = 1e-12
    return vecs @ np.diag(np.sqrt(vals))


def build_block_corr(
    d: int, blocks: List[List[int]], rho_in: float, rho_out: float = 0.0
) -> np.ndarray:
    C = np.full((d, d), rho_out, dtype=float)
    np.fill_diagonal(C, 1.0)
    for blk in blocks:
        for i in blk:
            for j in blk:
                if i != j:
                    C[i, j] = rho_in
    return nearest_pd(C)


def corr_to_cov(C: np.ndarray, var: np.ndarray | float = 1.0) -> np.ndarray:
    if np.isscalar(var):
        var = np.full(C.shape[0], float(var))
    D = np.diag(np.sqrt(var))
    return D @ C @ D


# Aproximações rápidas para Φ e Φ^{-1} (evitar SciPy)
def phi(x: np.ndarray) -> np.ndarray:
    """CDF da Normal(0,1) sem SciPy, usando aproximação de erf (A&S 7.1.26).

    Implementa uma aproximação vetorizada de erf para evitar dependência de SciPy.
    """
    z = np.asarray(x, dtype=float) / np.sqrt(2.0)
    # Abramowitz and Stegun formula 7.1.26 para erf(z)
    # erf(z) ≈ sign(z) * (1 - t * exp(-z^2 - a1 + a2*t + a3*t^2 + ...))
    sign = np.sign(z)
    a = [
        1.26551223,
        1.00002368,
        0.37409196,
        0.09678418,
        -0.18628806,
        0.27886807,
        -1.13520398,
        1.48851587,
        -0.82215223,
        0.17087277,
    ]
    t = 1.0 / (1.0 + 0.5 * np.abs(z))
    poly = (
        a[1] * t
        + a[2] * t**2
        + a[3] * t**3
        + a[4] * t**4
        + a[5] * t**5
        + a[6] * t**6
        + a[7] * t**7
        + a[8] * t**8
        + a[9] * t**9
    )
    erf_approx = sign * (1.0 - t * np.exp(-(z * z) - a[0] + poly))
    return 0.5 * (1.0 + erf_approx)


def inv_phi(u: np.ndarray) -> np.ndarray:
    # aproximação racional (Acklam)
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    u = np.clip(u, 1e-12, 1 - 1e-12)
    plow, phigh = 0.02425, 1 - 0.02425
    x = np.empty_like(u)
    # baixa cauda
    q = np.sqrt(-2 * np.log(u[u < plow]))
    x[u < plow] = (
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    # faixa central
    mask = (u >= plow) & (u <= phigh)
    q = u[mask] - 0.5
    r = q * q
    x[mask] = (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )
    # alta cauda
    q = np.sqrt(-2 * np.log(1 - u[u > phigh]))
    x[u > phigh] = -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    return x


# ====================== base com ground-truth ======================


class GTStream(SyntheticDataset):
    def __init__(self, d: int, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.d = d
        self.feature_names = [f"x{i + 1}" for i in range(d)]
        self._gt_drift_starts: List[int] = []
        self._gt_transitions: List[Tuple[int, int, str]] = []
        super().__init__(task="Regression", n_features=d, n_samples=None)

    def _push_drift(self, idx: int):
        self._gt_drift_starts.append(idx)

    def _push_span(self, s: int, e: int, typ: str):
        self._gt_transitions.append((s, e, typ))

    def ground_truth(self) -> Dict[str, list]:
        return {
            "drift_start_indices": list(self._gt_drift_starts),
            "transition_spans": [
                {"start": s, "end": e, "type": t} for s, e, t in self._gt_transitions
            ],
        }

    def to_pandas(self, limit: Optional[int] = None) -> pd.DataFrame:
        X, idxs = [], []
        for i, (x, _) in enumerate(self):
            X.append(x)
            idxs.append(i)
            if limit is not None and i + 1 >= limit:
                break
        df = pd.DataFrame(X)
        df.insert(0, "idx", idxs)
        # concept_id e is_transition
        gt = self.ground_truth()
        starts = gt["drift_start_indices"]
        spans = gt["transition_spans"]
        concept_id = np.zeros(len(df), dtype=int)
        k = 0
        nxt = 1
        for i in range(len(df)):
            if nxt < len(starts) and i >= starts[nxt]:
                k += 1
                nxt += 1
            concept_id[i] = k
        df["concept_id"] = concept_id
        is_tr = np.zeros(len(df), dtype=bool)
        for span in spans:
            is_tr[span["start"] : span["end"] + 1] = True
        df["is_transition"] = is_tr
        return df

    def to_csv(self, path: str, limit: Optional[int] = None):
        df = self.to_pandas(limit=limit)
        df.to_csv(path, index=False)
        return path


# ====================== 1) Copula Gaussiana ======================


@dataclass
class CopulaRegime:
    R: np.ndarray  # correlação DP
    n: int  # amostras do regime estável
    transition: Optional[Transition] = None
    window: int = 0  # usado se transition=="gradual"
    name: str = ""


class GaussianCopulaStream(GTStream):
    def __init__(self, regimes: List[CopulaRegime], seed: Optional[int] = None):
        super().__init__(d=regimes[0].R.shape[0], seed=seed)
        self.regimes = regimes

    def __iter__(self) -> Iterator[tuple[Dict[str, float], None]]:
        idx = 0
        for ridx, r in enumerate(self.regimes):
            if ridx == 0:
                self._push_drift(idx)  # início do 1º conceito

            # janela gradual com mistura sigmoide (entre regime anterior e atual)
            if r.transition == "gradual" and r.window > 0 and ridx > 0:
                R_old = self.regimes[ridx - 1].R
                L_old = cholesky_factor_from_psd(R_old)
                L_new = cholesky_factor_from_psd(r.R)
                mid = r.window / 2
                steep = 10 / r.window
                start = idx
                for t in range(r.window):
                    p_new = 1 / (1 + math.exp(-steep * (t - mid)))
                    L = L_new if self.rng.random() < p_new else L_old
                    z = self.rng.standard_normal(self.d)
                    g = L @ z
                    u = phi(g)
                    x = inv_phi(u)  # marginais ~N(0,1)
                    yield ({f"x{i + 1}": float(x[i]) for i in range(self.d)}, None)
                    idx += 1
                self._push_span(start, idx - 1, "gradual")
                self._push_drift(idx)  # novo conceito 100% ativo

            # regime estável
            L = cholesky_factor_from_psd(r.R)
            for _ in range(r.n):
                z = self.rng.standard_normal(self.d)
                g = L @ z
                u = phi(g)
                x = inv_phi(u)
                yield ({f"x{i + 1}": float(x[i]) for i in range(self.d)}, None)
                idx += 1


# ====================== 2) Mistura Linear com Rotação ======================


@dataclass
class MixRegime:
    A: np.ndarray  # mixing, Sigma = A A^T
    n: int
    transition: Optional[Transition] = None
    name: str = ""


class LinearMixStream(GTStream):
    def __init__(self, regimes: List[MixRegime], seed: Optional[int] = None):
        super().__init__(d=regimes[0].A.shape[0], seed=seed)
        self.regimes = regimes

    def __iter__(self) -> Iterator[tuple[Dict[str, float], None]]:
        idx = 0
        for ridx, r in enumerate(self.regimes):
            if ridx == 0:
                self._push_drift(idx)
            else:
                if r.transition == "abrupt":
                    self._push_drift(idx)
            for _ in range(r.n):
                y = self.rng.standard_normal(self.d)
                x = r.A @ y
                yield ({f"x{i + 1}": float(x[i]) for i in range(self.d)}, None)
                idx += 1


def block_rotation(d: int, i: int, j: int, theta: float) -> np.ndarray:
    R = np.eye(d)
    c, s = math.cos(theta), math.sin(theta)
    R[i, i] = R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


# ====================== 3) Grafo de Dependência (Precisão) ======================


@dataclass
class GraphRegime:
    Theta: np.ndarray  # precisão DP
    n: int
    transition: Optional[Transition] = None
    name: str = ""


class GraphPrecisionStream(GTStream):
    def __init__(self, regimes: List[GraphRegime], seed: Optional[int] = None):
        super().__init__(d=regimes[0].Theta.shape[0], seed=seed)
        self.regimes = regimes

    def __iter__(self) -> Iterator[tuple[Dict[str, float], None]]:
        idx = 0
        for ridx, r in enumerate(self.regimes):
            if ridx == 0:
                self._push_drift(idx)
            else:
                if r.transition == "abrupt":
                    self._push_drift(idx)
            Theta = nearest_pd(r.Theta)
            Sigma = nearest_pd(np.linalg.inv(Theta))
            # Sigma is not rescaled to unit variances, so S2's marginal variances
            # (0.38-0.66) change with the regime, by up to 1.47x at a drift: S2
            # is not marginal-preserving, unlike S1 and S4. The manuscript says
            # so (Sec. V-B). Dividing Sigma by outer(sd, sd) would fix it without
            # touching the zero pattern of Theta, but the published S2 results
            # were produced without it.
            L = cholesky_factor_from_psd(Sigma)
            for _ in range(r.n):
                z = self.rng.standard_normal(self.d)
                x = L @ z
                yield ({f"x{i + 1}": float(x[i]) for i in range(self.d)}, None)
                idx += 1


def make_chain_precision(
    d: int, w: float = 0.4, lam: float = 0.1, extra_edges: List[Tuple[int, int]] = None
) -> np.ndarray:
    Theta = np.eye(d)
    for i in range(d - 1):
        Theta[i, i + 1] = Theta[i + 1, i] = -w
    if extra_edges:
        for i, j in extra_edges:
            Theta[i, j] = Theta[j, i] = -w
    # reforçar diagonal para DP (não fixa a variância marginal: ver GraphPrecisionStream)
    for i in range(d):
        Theta[i, i] += lam + (np.abs(Theta[i]).sum() - np.abs(Theta[i, i]))
    return nearest_pd(Theta)


# ====================== 4) Correlação Dinâmica tipo DCC ======================


@dataclass
class DCCRegime:
    S: np.ndarray  # correlação de longo prazo (dxd)
    a: float  # coef. de choque
    b: float  # coef. de persistência
    n: int
    transition: Optional[Transition] = None
    S2: Optional[np.ndarray] = None
    name: str = ""


class DCCStream(GTStream):
    def __init__(self, regimes: List[DCCRegime], seed: Optional[int] = None):
        super().__init__(d=regimes[0].S.shape[0], seed=seed)
        self.regimes = regimes

    def __iter__(self) -> Iterator[tuple[Dict[str, float], None]]:
        idx = 0
        # estado inicial
        R_prev = self.regimes[0].S.copy()
        L_prev = cholesky_factor_from_psd(R_prev)
        for ridx, r in enumerate(self.regimes):
            if ridx == 0:
                self._push_drift(idx)
            # incremental: interpolação de S ao longo de n (a,b fixos)
            if r.transition == "incremental" and r.S2 is not None:
                self._push_drift(idx)  # início da transição
                start = idx
                for t in range(r.n):
                    alpha = (t + 1) / r.n
                    S_t = nearest_pd((1 - alpha) * r.S + alpha * r.S2)
                    z = self.rng.standard_normal(self.d)
                    g = L_prev @ z
                    e = g / np.linalg.norm(g)  # padronização simples (proxy)
                    R_t = nearest_pd(
                        (1 - r.a - r.b) * S_t + r.a * np.outer(e, e) + r.b * R_prev
                    )
                    L_t = cholesky_factor_from_psd(R_t)
                    x = L_t @ self.rng.standard_normal(self.d)
                    yield ({f"x{i + 1}": float(x[i]) for i in range(self.d)}, None)
                    idx += 1
                    R_prev, L_prev = R_t, L_t
                self._push_span(start, idx - 1, "incremental")
            else:
                # regime estável DCC (a,b,S fixos)
                self._push_drift(idx) if ridx > 0 and r.transition == "abrupt" else None
                for _ in range(r.n):
                    z = self.rng.standard_normal(self.d)
                    g = L_prev @ z
                    e = g / np.linalg.norm(g)
                    R_t = nearest_pd(
                        (1 - r.a - r.b) * r.S + r.a * np.outer(e, e) + r.b * R_prev
                    )
                    L_t = cholesky_factor_from_psd(R_t)
                    x = L_t @ self.rng.standard_normal(self.d)
                    yield ({f"x{i + 1}": float(x[i]) for i in range(self.d)}, None)
                    idx += 1
                    R_prev, L_prev = R_t, L_t


# ====================== 5) HMM de Regimes (recorrência) ======================


@dataclass
class HMMRegime:
    Sigma: np.ndarray
    n: int
    transition: Optional[Transition] = None
    name: str = ""


class HMMStream(GTStream):
    def __init__(self, regimes: List[HMMRegime], seed: Optional[int] = None):
        super().__init__(d=regimes[0].Sigma.shape[0], seed=seed)
        self.regimes = regimes

    def __iter__(self) -> Iterator[tuple[Dict[str, float], None]]:
        idx = 0
        for ridx, r in enumerate(self.regimes):
            if ridx == 0:
                self._push_drift(idx)
            else:
                if r.transition == "abrupt":
                    self._push_drift(idx)
            L = cholesky_factor_from_psd(r.Sigma)
            for _ in range(r.n):
                x = L @ self.rng.standard_normal(self.d)
                yield ({f"x{i + 1}": float(x[i]) for i in range(self.d)}, None)
                idx += 1


# ====================== configurações (8D, 40k, 5 drifts) ======================

D = 8
BLOCKS = [[0, 1, 2], [3, 4, 5], [6, 7]]


def make_R(rho_in: float, rho_out: float = 0.0, blocks=BLOCKS) -> np.ndarray:
    return build_block_corr(D, blocks, rho_in=rho_in, rho_out=rho_out)


def make_Sigma_from_R(rho_in: float, rho_out: float = 0.0, blocks=BLOCKS) -> np.ndarray:
    return corr_to_cov(make_R(rho_in, rho_out, blocks), 1.0)


def example_copula_8d_40k(seed=101) -> GaussianCopulaStream:
    # 6 conceitos com 5 janelas graduais de 800 cada => total 40k:
    # [6000, 800, 6000, 800, 6000, 800, 6000, 800, 6000, 800, 6000] = 40k
    rhos = [0.10, 0.35, 0.60, 0.20, 0.70, 0.85]
    regs: List[CopulaRegime] = []
    for i, rho in enumerate(rhos):
        R = make_R(rho)
        if i == 0:
            regs.append(CopulaRegime(R=R, n=6000, transition=None, name=f"A rho={rho}"))
        else:
            regs.append(
                CopulaRegime(
                    R=R,
                    n=6000,
                    transition="gradual",
                    window=800,
                    name=f"{chr(65 + i)} rho={rho}",
                )
            )
    return GaussianCopulaStream(regs, seed=seed)


def example_linear_mix_8d_40k(seed=102) -> LinearMixStream:
    # 6 conceitos estáveis com 5 drifts abruptos: 8000,8000,8000,8000,4000,4000
    gains = np.array([2.0, 1.2, 1.0, 1.8, 1.0, 0.8, 1.0, 1.0])
    A0 = np.diag(gains) @ block_rotation(D, 0, 3, math.radians(0))
    A1 = np.diag(gains) @ block_rotation(D, 0, 3, math.radians(25))
    A2 = np.diag(gains) @ block_rotation(D, 1, 4, math.radians(40))
    A3 = np.diag(gains) @ block_rotation(D, 2, 5, math.radians(60))
    A4 = np.diag(gains) @ block_rotation(D, 0, 1, math.radians(35))
    A5 = np.diag(gains) @ block_rotation(D, 6, 7, math.radians(50))
    segs = [8000, 8000, 8000, 8000, 4000, 4000]
    A_list = [A0, A1, A2, A3, A4, A5]
    regs = []
    for i, (A, n) in enumerate(zip(A_list, segs)):
        regs.append(
            MixRegime(A=A, n=n, transition=None if i == 0 else "abrupt", name=f"R{i}")
        )
    return LinearMixStream(regs, seed=seed)


def example_graph_precision_8d_40k(seed=103) -> GraphPrecisionStream:
    # 6 conceitos com rewiring do grafo e pesos
    Theta0 = make_chain_precision(D, w=0.35, lam=0.3, extra_edges=[])
    Theta1 = make_chain_precision(D, w=0.45, lam=0.3, extra_edges=[(0, 2), (3, 5)])
    Theta2 = make_chain_precision(D, w=0.25, lam=0.3, extra_edges=[(2, 4)])
    Theta3 = make_chain_precision(D, w=0.55, lam=0.3, extra_edges=[(1, 3), (4, 6)])
    Theta4 = make_chain_precision(D, w=0.30, lam=0.3, extra_edges=[(5, 7)])
    Theta5 = make_chain_precision(D, w=0.60, lam=0.3, extra_edges=[(0, 4), (2, 6)])
    segs = [8000, 8000, 8000, 8000, 4000, 4000]
    Th_list = [Theta0, Theta1, Theta2, Theta3, Theta4, Theta5]
    regs = []
    for i, (Th, n) in enumerate(zip(Th_list, segs)):
        regs.append(
            GraphRegime(
                Theta=Th, n=n, transition=None if i == 0 else "abrupt", name=f"G{i}"
            )
        )
    return GraphPrecisionStream(regs, seed=seed)


def example_dcc_8d_40k(seed=104) -> DCCStream:
    # 5 transições incrementais, separadas por estáveis (similar ao seu incremental multi)
    # cronograma: A(4000) -> A→B(3000) -> B(4000) -> B→C(3000) -> C(4000) -> C→D(3000)
    #             -> D(4000) -> D→E(3000) -> E(4000) -> E→F(3000) -> F(5000) = 40k
    rhos = [0.15, 0.35, 0.60, 0.20, 0.70, 0.85]
    Ss = [make_R(rho) for rho in rhos]
    a, b = 0.05, 0.92  # parâmetros DCC leves
    regs: List[DCCRegime] = []
    # A estável
    regs.append(DCCRegime(S=Ss[0], a=a, b=b, n=4000, transition=None, name="A"))
    # 5 incrementais + 4 estáveis + F estável final
    lens = [3000, 4000, 3000, 4000, 3000, 4000, 3000, 4000, 3000, 5000]
    pair_idx = 0
    for i in range(0, len(lens), 2):
        # incremental i: k -> k+1
        k = pair_idx
        regs.append(
            DCCRegime(
                S=Ss[k],
                a=a,
                b=b,
                n=lens[i],
                transition="incremental",
                S2=Ss[k + 1],
                name=f"{chr(65 + k)}→{chr(66 + k)}",
            )
        )
        # estável (k+1)
        regs.append(
            DCCRegime(
                S=Ss[k + 1],
                a=a,
                b=b,
                n=lens[i + 1],
                transition="abrupt",
                name=f"{chr(66 + k)}",
            )
        )
        pair_idx += 1
    return DCCStream(regs, seed=seed)


def example_hmm_8d_40k(seed=105) -> HMMStream:
    # recorrência determinística de estados: A->B->A->C->B->C (6 segmentos, 5 drifts)
    SigA = make_Sigma_from_R(0.20)
    SigB = make_Sigma_from_R(0.70)
    SigC = make_Sigma_from_R(0.40, rho_out=0.10)  # levemente "densa"
    segs = [7000, 7000, 7000, 7000, 7000, 5000]  # soma 40k
    states = [
        ("A", SigA),
        ("B", SigB),
        ("A", SigA),
        ("C", SigC),
        ("B", SigB),
        ("C", SigC),
    ]
    regs = []
    for i, ((name, Sig), n) in enumerate(zip(states, segs)):
        regs.append(
            HMMRegime(
                Sigma=Sig, n=n, transition=None if i == 0 else "abrupt", name=name
            )
        )
    return HMMStream(regs, seed=seed)


# ====================== execução ======================

if __name__ == "__main__":
    streams = [
        ("copula_gauss_8d_40k.csv", example_copula_8d_40k()),
        ("mix_rotation_8d_40k.csv", example_linear_mix_8d_40k()),
        ("graph_precision_8d_40k.csv", example_graph_precision_8d_40k()),
        ("dcc_dynamic_8d_40k.csv", example_dcc_8d_40k()),
        ("hmm_regimes_8d_40k.csv", example_hmm_8d_40k()),
    ]
    for fname, stream in streams:
        path = stream.to_csv(fname)
        gt = stream.ground_truth()
        print(f"\n=== {fname} ===")
        print("drift_start_indices:", gt["drift_start_indices"])
        print("transition_spans:", gt["transition_spans"])
        print("salvo em:", path)
