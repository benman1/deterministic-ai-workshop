import numpy as np

import meridian as m


def test_split_sizes_and_no_scenario_straddles_splits():
    rows = m.load()
    assert {s: sum(r["split"] == s for r in rows) for s in ("train", "calibration", "test")} == {
        "train": 800,
        "calibration": 200,
        "test": 200,
    }
    split_of = {}
    for r in rows:
        key = (r["label"], r["scenario"])
        assert split_of.setdefault(key, r["split"]) == r["split"]


def test_every_route_present_in_each_split():
    for s in ("train", "calibration", "test"):
        assert {r["label"] for r in m.load(s)} == set(m.ROUTES)


def test_wilson_matches_the_slide_example():
    # 28 of 30: Wilson 0.787 to 0.982 (the Wald interval runs past 1.0)
    lo, hi = m.wilson(28, 30)
    assert round(lo, 3) == 0.787 and round(hi, 3) == 0.982


def test_cluster_bootstrap_is_wider_than_naive_when_clusters_are_identical():
    correct = np.array([1] * 25 + [0] * 25)
    clusters = np.repeat(np.arange(10), 5)  # 10 clusters; each cluster is all-right or all-wrong
    correct = np.repeat(np.array([1] * 5 + [0] * 5), 5)
    lo, hi = m.cluster_bootstrap_ci(correct, clusters, n_boot=2000)
    nlo, nhi = m.wilson(int(correct.sum()), len(correct))
    assert (hi - lo) > (nhi - nlo)
