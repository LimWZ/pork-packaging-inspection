"""Timing helpers for the performance analysis."""

import time

import numpy as np


def time_per_item(fn, items, warmup=5):
    """Time one call of ``fn`` on each item.

    The first ``warmup`` items are run once without timing first. Early calls
    are slower because memory is being allocated, caches are cold and, on a
    GPU, kernels are compiled, and including them would bias the results.

    Parameters
    ----------
    fn : callable
        Function taking one item. It must finish all its work before
        returning; for GPU code, copy the result back to the CPU inside ``fn``.
    items : sequence
        Inputs to time, e.g. images already loaded into memory.
    warmup : int, optional
        Number of untimed calls before timing starts.

    Returns
    -------
    numpy.ndarray
        Milliseconds per call, one value per item.

    Raises
    ------
    ValueError
        If ``items`` is empty.
    """
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
    """Summarise timings with statistics that resist occasional slow calls.

    The median and interquartile range are reported rather than the mean and
    standard deviation, because background processes cause occasional very
    slow calls that would distort the mean.

    Parameters
    ----------
    times_ms : array_like
        Milliseconds per call.

    Returns
    -------
    dict
        ``median_ms``, ``q1_ms``, ``q3_ms`` (25th and 75th percentiles),
        ``mean_ms`` and ``images_per_s`` (1000 / median).
    """
    t = np.asarray(times_ms, dtype=float)
    q1, median, q3 = np.percentile(t, [25, 50, 75])
    return {"median_ms": median, "q1_ms": q1, "q3_ms": q3,
            "mean_ms": t.mean(), "images_per_s": 1000 / median}