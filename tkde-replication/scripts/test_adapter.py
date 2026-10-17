"""Functional check of the FFWIN adapter on streams with known ground truth."""
import sys
import numpy as np

sys.path.insert(0, ".")   # ajuste se ff_pkg/detectors estiverem noutro lugar
from detectors.ffwin import FasanoFranceschiniWindow


def make_stream(n, d, rho_schedule, seed):
    """Correlated Gaussian stream; marginals are N(0,1) in every regime."""
    rng = np.random.default_rng(seed)
    out = []
    for start, end, rho in rho_schedule:
        R = np.full((d, d), rho)
        np.fill_diagonal(R, 1.0)
        L = np.linalg.cholesky(R)
        Z = rng.standard_normal((end - start, d))
        out.append(Z @ L.T)
    return np.vstack(out)


def run(data, **kwargs):
    det = FasanoFranceschiniWindow(**kwargs)
    alarms = []
    for i, row in enumerate(data):
        if det.update({f"f{j}": v for j, v in enumerate(row)}):
            alarms.append(i)
    return alarms, det


D, N, CHANGE = 4, 3000, 1500
PARAMS = dict(n_samples=250, step=125, alpha=0.05, n_permutations=50,
              pca_threshold=None, n_jobs=1, seed=7)

print("=" * 62)
print("TEST 1  drift de correlacao puro  rho 0.0 -> 0.85  em t=1500")
drift_data = make_stream(N, D, [(0, CHANGE, 0.0), (CHANGE, N, 0.85)], seed=1)

m0 = drift_data[:CHANGE].mean(axis=0)
m1 = drift_data[CHANGE:].mean(axis=0)
s0 = drift_data[:CHANGE].std(axis=0)
s1 = drift_data[CHANGE:].std(axis=0)
print(f"  marginais: |dif media| max = {np.abs(m0 - m1).max():.4f}, "
      f"|dif desvio| max = {np.abs(s0 - s1).max():.4f}   (devem ser ~0)")
c0 = np.corrcoef(drift_data[:CHANGE].T)[0, 1]
c1 = np.corrcoef(drift_data[CHANGE:].T)[0, 1]
print(f"  correlacao: {c0:.3f} -> {c1:.3f}   (deve mudar muito)")

alarms, det = run(drift_data, **PARAMS)
print(f"  janelas avaliadas : {len(det.history)}")
print(f"  alarmes           : {alarms}")
after = [a for a in alarms if a >= CHANGE]
before = [a for a in alarms if a < CHANGE]
print(f"  antes do drift    : {len(before)}   (falso positivo)")
print(f"  depois do drift   : {len(after)}")
if after:
    print(f"  atraso deteccao   : {min(after) - CHANGE} amostras")
print(f"  RESULTADO: {'PASS' if after and not before else 'CHECAR'}")

print()
print("=" * 62)
print("TEST 2  fluxo estacionario  (taxa de falso alarme)")
flat = make_stream(N, D, [(0, N, 0.3)], seed=2)
alarms_flat, det_flat = run(flat, **PARAMS)
print(f"  janelas avaliadas : {len(det_flat.history)}")
print(f"  alarmes           : {alarms_flat}")
rate = len(alarms_flat) / max(1, len(det_flat.history))
print(f"  taxa              : {rate:.3f}   (alpha = 0.05)")
print(f"  RESULTADO: {'PASS' if rate <= 0.20 else 'CHECAR'}")

print()
print("=" * 62)
print("TEST 3  hard reset  (nao deve realarmar na mesma mudanca)")
a_reset, _ = run(drift_data, **{**PARAMS, "hard_reset": True})
a_noreset, _ = run(drift_data, **{**PARAMS, "hard_reset": False})
print(f"  com hard_reset    : {len(a_reset)} alarmes  {a_reset}")
print(f"  sem hard_reset    : {len(a_noreset)} alarmes  {a_noreset}")
print(f"  RESULTADO: {'PASS' if len(a_reset) <= len(a_noreset) else 'CHECAR'}")

print()
print("=" * 62)
print("TEST 4  regra do p-valor  paper vs randomized")
for rule in ("paper", "randomized"):
    a, d_ = run(drift_data, **{**PARAMS, "p_value_rule": rule})
    ps = [h["p"] for h in d_.history]
    print(f"  {rule:<11s}: {len(a)} alarmes, p min={min(ps):.4f} max={max(ps):.4f}")

print()
print("=" * 62)
print("TEST 5  backend  brute ('b') vs range tree ('r') devem dar o mesmo D")
a_b, d_b = run(drift_data, **{**PARAMS, "method": "b"})
try:
    a_r, d_r = run(drift_data, **{**PARAMS, "method": "r"})
    Db = [h["D"] for h in d_b.history]
    Dr = [h["D"] for h in d_r.history]
    same = Db == Dr
    print(f"  D identicos       : {same}")
    if not same:
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(Db, Dr)) if x != y]
        print(f"  divergencias      : {len(diff)} de {len(Db)}; primeiras {diff[:5]}")
    print(f"  RESULTADO: {'PASS' if same else 'FALHA -- bug no range_tree'}")
except Exception as exc:
    print(f"  range tree levantou: {type(exc).__name__}: {exc}")
    print("  RESULTADO: FALHA")
print("=" * 62)
