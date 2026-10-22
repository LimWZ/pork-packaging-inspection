"""Tray-level pass/fail evaluation: decision thresholds, rates and confidence intervals.

A tray fails inspection when the detector's highest box confidence on it
reaches the decision threshold. Performance is described by two rates:

- **catch rate** (sensitivity): share of defective trays that fail inspection
- **pass rate** (specificity): share of good trays that pass inspection

Overall accuracy is reported too, but it is misleading here because most
trays are defective: failing every tray already scores 85% on the test set.
"""

import numpy as np
import pandas as pd


def wilson_interval(successes, trials, z=1.96):
    """Wilson score confidence interval for a proportion.

    Preferred over the normal approximation ``p ± z*sqrt(p(1-p)/n)``, which
    can give limits outside [0, 1] and has poor coverage when ``n`` is small
    or ``p`` is near 0 or 1. Both apply to the good-tray rates here.

    Parameters
    ----------
    successes : int
        Number of successes, between 0 and ``trials``.
    trials : int
        Number of trials; must be positive.
    z : float, optional
        Standard normal quantile; 1.96 gives a 95% interval.

    Returns
    -------
    tuple of float
        ``(low, high)``, both in [0, 1].

    Raises
    ------
    ValueError
        If ``trials`` is not positive or ``successes`` is outside [0, trials].
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _as_arrays(gt_defective, scores):
    gt = np.asarray(gt_defective, dtype=bool)
    s = np.asarray(scores, dtype=float)
    if gt.shape != s.shape:
        raise ValueError("gt_defective and scores must be the same length")
    return gt, s


def tray_counts(gt_defective, scores, threshold):
    """Count tray-level outcomes at one decision threshold.

    Parameters
    ----------
    gt_defective : array_like of bool
        True for trays with at least one labelled defect.
    scores : array_like of float
        Highest predicted box confidence per tray (0 if no boxes).
    threshold : float
        A tray fails inspection if its score is at least this value.
        Use 0 to fail every tray and ``numpy.inf`` to pass every tray.

    Returns
    -------
    dict
        ``caught`` (defective, failed), ``missed`` (defective, passed),
        ``passed`` (good, passed) and ``false_rejects`` (good, failed).
    """
    gt, s = _as_arrays(gt_defective, scores)
    fail = s >= threshold
    return {"caught": int(np.sum(fail & gt)), "missed": int(np.sum(~fail & gt)),
            "passed": int(np.sum(~fail & ~gt)), "false_rejects": int(np.sum(fail & ~gt))}


def summarise(gt_defective, scores, threshold, z=1.96):
    """Catch rate, pass rate and accuracy with Wilson intervals at one threshold.

    Parameters
    ----------
    gt_defective, scores, threshold
        As for :func:`tray_counts`.
    z : float, optional
        Standard normal quantile for the intervals.

    Returns
    -------
    dict
        The counts from :func:`tray_counts`, plus ``catch_rate``,
        ``pass_rate`` and ``accuracy``, each with ``_low`` and ``_high``
        interval limits.

    Raises
    ------
    ValueError
        If there are no defective trays or no good trays.
    """
    c = tray_counts(gt_defective, scores, threshold)
    n_defective = c["caught"] + c["missed"]
    n_good = c["passed"] + c["false_rejects"]
    if n_defective == 0 or n_good == 0:
        raise ValueError("need at least one defective and one good tray")
    out = {"threshold": threshold, **c}
    for name, k, n in [("catch_rate", c["caught"], n_defective),
                       ("pass_rate", c["passed"], n_good),
                       ("accuracy", c["caught"] + c["passed"], n_defective + n_good)]:
        out[name] = k / n
        out[f"{name}_low"], out[f"{name}_high"] = wilson_interval(k, n, z)
    return out


def threshold_sweep(gt_defective, scores, thresholds):
    """Catch rate and pass rate at each of several thresholds.

    Parameters
    ----------
    gt_defective, scores
        As for :func:`tray_counts`.
    thresholds : array_like of float
        Thresholds to evaluate.

    Returns
    -------
    pandas.DataFrame
        Columns ``threshold``, ``catch_rate`` and ``pass_rate``.
    """
    gt, s = _as_arrays(gt_defective, scores)
    t = np.asarray(thresholds, dtype=float)
    fail = s[None, :] >= t[:, None]   # one row per threshold
    return pd.DataFrame({
        "threshold": t,
        "catch_rate": fail[:, gt].mean(axis=1),
        "pass_rate": (~fail[:, ~gt]).mean(axis=1),
    })


def choose_threshold(gt_defective, scores, min_catch_rate=0.95):
    """Choose a decision threshold that catches a target share of defective trays.

    Missing a defect (e.g. an unsealed pack reaching a customer) costs more
    than a false reject (a good tray sent for manual checking), so the catch
    rate is treated as a constraint. Among thresholds that meet it, the rule
    picks the highest pass rate, then the highest catch rate among ties.

    The rates only change when the threshold crosses a tray's score, so every
    threshold strictly between two neighbouring scores gives the same result.
    The threshold is placed midway between the two scores bounding the chosen
    interval, which leaves the largest margin on both sides.

    Parameters
    ----------
    gt_defective, scores
        As for :func:`tray_counts`. Use validation data, never test data.
    min_catch_rate : float, optional
        Required share of defective trays caught, in (0, 1].

    Returns
    -------
    float
        The chosen threshold.

    Raises
    ------
    ValueError
        If ``min_catch_rate`` is outside (0, 1], or there are no defective
        trays or no good trays.
    """
    if not 0 < min_catch_rate <= 1:
        raise ValueError("min_catch_rate must be in (0, 1]")
    gt, s = _as_arrays(gt_defective, scores)
    if not gt.any() or gt.all():
        raise ValueError("need at least one defective and one good tray")
    levels = np.unique(s)                          # ascending candidate thresholds
    fail = s[None, :] >= levels[:, None]           # one row per candidate
    catch = fail[:, gt].mean(axis=1)               # falls as the threshold rises
    passed = (~fail[:, ~gt]).mean(axis=1)          # rises as the threshold rises
    ok = catch >= min_catch_rate                   # always true for the lowest level
    best = ok & (passed == passed[ok].max())
    i = np.flatnonzero(best)[np.argmax(catch[best])]
    lower = levels[i - 1] if i > 0 else 0.0
    return float((lower + levels[i]) / 2)

def roc_auc(gt_defective, scores):
    """Area under the ROC curve, as a threshold-free summary of separation.

    Equals the probability that a randomly chosen defective tray scores
    higher than a randomly chosen good tray, with ties counting half
    (the Mann-Whitney statistic). 1 is perfect separation; 0.5 is chance.
    Compares every defective/good pair, which costs O(P*N) time and memory;
    fine for a few hundred trays.

    Parameters
    ----------
    gt_defective, scores
        As for :func:`tray_counts`.

    Returns
    -------
    float
        AUC in [0, 1].

    Raises
    ------
    ValueError
        If there are no defective trays or no good trays.
    """
    gt, s = _as_arrays(gt_defective, scores)
    pos, neg = s[gt], s[~gt]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("need at least one defective and one good tray")
    diff = pos[:, None] - neg[None, :]
    return float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))