import pytest

from packinspect.profiling import summarise_times, time_per_item


def test_time_per_item_runs_warmup_then_times_each_item():
    calls = []
    times = time_per_item(calls.append, [1, 2, 3, 4], warmup=2)
    assert calls == [1, 2, 1, 2, 3, 4]      # two warm-up calls, then one per item
    assert len(times) == 4
    assert (times >= 0).all()


def test_time_per_item_rejects_empty_input():
    with pytest.raises(ValueError):
        time_per_item(lambda x: x, [])


def test_summarise_times_uses_median():
    s = summarise_times([1, 2, 3, 4, 100])   # one slow outlier
    assert s["median_ms"] == pytest.approx(3)
    assert s["mean_ms"] == pytest.approx(22)
    assert s["images_per_s"] == pytest.approx(1000 / 3)