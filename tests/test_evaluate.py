import numpy as np
import pytest

from packinspect.evaluate import (choose_threshold, roc_auc, summarise,
                                  threshold_sweep, tray_counts, wilson_interval)

# Four defective trays and two good trays
GT = [1, 1, 1, 1, 0, 0]
SCORES = [0.9, 0.8, 0.3, 0.6, 0.1, 0.5]


def test_wilson_interval_zero_successes():
    # Known value: 0/10 at 95% gives (0, 0.2775)
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(0.2775, abs=1e-3)


@pytest.mark.parametrize("k, n", [(1, 12), (9, 12), (62, 68), (68, 68)])
def test_wilson_interval_contains_estimate(k, n):
    low, high = wilson_interval(k, n)
    assert 0 <= low <= k / n <= high <= 1


@pytest.mark.parametrize("k, n", [(1, 0), (-1, 5), (6, 5)])
def test_wilson_interval_rejects_bad_input(k, n):
    with pytest.raises(ValueError):
        wilson_interval(k, n)


def test_tray_counts():
    assert tray_counts(GT, SCORES, 0.5) == {
        "caught": 3, "missed": 1, "passed": 1, "false_rejects": 1}


def test_tray_counts_extreme_thresholds():
    assert tray_counts(GT, SCORES, 0.0)["passed"] == 0         # fail every tray
    assert tray_counts(GT, SCORES, np.inf)["caught"] == 0      # pass every tray


def test_tray_counts_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        tray_counts([1, 0], [0.5], 0.5)


def test_summarise_rates():
    s = summarise(GT, SCORES, 0.5)
    assert s["catch_rate"] == pytest.approx(0.75)
    assert s["pass_rate"] == pytest.approx(0.5)
    assert s["accuracy"] == pytest.approx(4 / 6)


def test_summarise_needs_both_classes():
    with pytest.raises(ValueError):
        summarise([1, 1], [0.5, 0.6], 0.5)


def test_threshold_sweep_matches_tray_counts():
    sweep = threshold_sweep(GT, SCORES, [0.0, 0.5, 1.0])
    assert sweep["catch_rate"].tolist() == [1.0, 0.75, 0.0]
    assert sweep["pass_rate"].tolist() == [0.0, 0.5, 1.0]


def test_choose_threshold():
    # 75% target: 0.55 catches 3/4 defective and passes 2/2 good trays
    assert choose_threshold(GT, SCORES, min_catch_rate=0.75) == pytest.approx(0.55)
    # 100% target: 0.2 catches 4/4 and passes 1/2
    assert choose_threshold(GT, SCORES, min_catch_rate=1.0) == pytest.approx(0.2)


def test_choose_threshold_prefers_more_catches_when_pass_rate_ties():
    # Thresholds 0.2 and 0.4 both pass the good tray, but 0.2 also catches
    # the defective tray scoring 0.3, so the rule must not pick 0.4.
    t = choose_threshold([1, 1, 1, 0], [0.9, 0.5, 0.3, 0.1], min_catch_rate=0.6)
    assert t == pytest.approx(0.2)
    assert tray_counts([1, 1, 1, 0], [0.9, 0.5, 0.3, 0.1], t)["caught"] == 3


def test_choose_threshold_needs_good_trays():
    with pytest.raises(ValueError):
        choose_threshold([1, 1], [0.5, 0.6])


@pytest.mark.parametrize("rate", [0.0, 1.5])
def test_choose_threshold_rejects_bad_rate(rate):
    with pytest.raises(ValueError):
        choose_threshold(GT, SCORES, min_catch_rate=rate)


@pytest.mark.parametrize("scores, expected", [
    ([0.9, 0.8, 0.2, 0.1], 1.0),    # every defective tray scores higher
    ([0.1, 0.2, 0.8, 0.9], 0.0),    # every good tray scores higher
    ([0.5, 0.5, 0.5, 0.5], 0.5),    # all tied
])
def test_roc_auc(scores, expected):
    assert roc_auc([1, 1, 0, 0], scores) == pytest.approx(expected)