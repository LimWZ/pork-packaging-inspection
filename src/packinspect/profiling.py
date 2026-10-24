import time

import numpy as np


def time_per_item(fn, items, warmup=5):
    items = list(items)
    if not items:
        raise ValueError("no items to time")
    for item in items[:warmup]:
        fn(item)
    times = []
    for item in items:
        start = time.perf_counter()
        fn(item)
        times.append((time.perf_counter() - start) * 1000)
    return np.array(times)


def summarise_times(times_ms):
    t = np.asarray(times_ms, dtype=float)
    q1, median, q3 = np.percentile(t, [25, 50, 75])
    return {"median_ms": median, "q1_ms": q1, "q3_ms": q3,
            "mean_ms": t.mean(), "images_per_s": 1000 / median}