import numpy as np
from ffwin.sliding_window import SlidingWindowFasanoFranceschini

T, d = 5000, 8
X = np.random.randn(T, d).astype(np.float32)
X[2500:] += 0.7

ff = SlidingWindowFasanoFranceschini(
    window_size=200,
    step=100,
    alpha=0.05,
    n_permutations=20,
    method="b",
    pairing="lag_step",
)
res = ff.run_parallel(X)
alarms = [r["start_cur"] for r in res if r["drift"]]
print("alarms (first 10):", alarms[:10])
