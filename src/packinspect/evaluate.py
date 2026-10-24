import numpy as np
import pandas as pd


def wilson_interval(successes, trials, z=1.96):
    # Wilson score confidence interval for a binomial proportion
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
    gt, s = _as_arrays(gt_defective, scores)
    fail = s >= threshold
    return {"caught": int(np.sum(fail & gt)), "missed": int(np.sum(~fail & gt)),
            "passed": int(np.sum(~fail & ~gt)), "false_rejects": int(np.sum(fail & ~gt))}


def summarise(gt_defective, scores, threshold, z=1.96):
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
    gt, s = _as_arrays(gt_defective, scores)
    t = np.asarray(thresholds, dtype=float)
    fail = s[None, :] >= t[:, None]   # one row per threshold
    return pd.DataFrame({
        "threshold": t,
        "catch_rate": fail[:, gt].mean(axis=1),
        "pass_rate": (~fail[:, ~gt]).mean(axis=1),
    })


def choose_threshold(gt_defective, scores, min_catch_rate=0.95):
    # Highest pass rate subject to catch rate >= min_catch_rate, placed midway between score levels
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
    gt, s = _as_arrays(gt_defective, scores)
    pos, neg = s[gt], s[~gt]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("need at least one defective and one good tray")
    diff = pos[:, None] - neg[None, :]
    return float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))