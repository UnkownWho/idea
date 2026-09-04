import numpy as np
from types import SimpleNamespace

from experiments.run_protocol_baselines import (
    choose_bsv,
    official_concat_allowed,
    official_concat_features,
    oracle_pvp_features,
    single_view_inputs,
)


def synthetic_dataset(aligned_prop=1.0, complete_prop=1.0):
    mask = np.ones((4, 2), dtype=np.float32)
    views = [np.arange(4, dtype=np.float32).reshape(-1, 1),
             (10 + np.arange(4, dtype=np.float32)).reshape(-1, 1)]
    sample_ids = np.array([[0, 2], [1, 0], [2, 3], [3, 1]], dtype=np.int64)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    return SimpleNamespace(
        views=views, labels=labels, mask_matrix=mask, view_sample_ids=sample_ids,
        global_ids=np.arange(4), num_views=2, view_dims=[1, 1], n_samples=4,
        num_clusters=2, aligned_prop=aligned_prop, complete_prop=complete_prop,
        is_pvp=aligned_prop < 1, is_psp=complete_prop < 1,
        paired_mask_matrix=np.array([True, False, True, False]),
        dataset_name="synthetic",
    )


def test_pvp_single_view_uses_source_ids_for_labels():
    dataset = synthetic_dataset(aligned_prop=0.5)
    _, labels, _, source_ids = single_view_inputs(dataset, 1)
    np.testing.assert_array_equal(source_ids, [2, 0, 3, 1])
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])


def test_pvp_rowwise_concat_is_not_official():
    assert not official_concat_allowed(synthetic_dataset(aligned_prop=0.5))
    assert official_concat_allowed(synthetic_dataset())


def test_bsv_reuses_all_metrics_from_selected_view():
    selected, scores = choose_bsv({
        "View0-KMeans": {"acc": 0.9, "nmi": 0.2, "ari": 0.3},
        "View1-KMeans": {"acc": 0.8, "nmi": 0.9, "ari": 0.9},
    })
    assert selected == "View0-KMeans"
    assert scores == {"acc": 0.9, "nmi": 0.2, "ari": 0.3}


def test_psp_zero_fill_concat_keeps_all_samples():
    dataset = synthetic_dataset(complete_prop=0.5)
    dataset.mask_matrix[1, 1] = 0
    features, labels = official_concat_features(dataset, add_mask=True)
    assert features.shape == (4, 4)
    assert len(labels) == 4
    np.testing.assert_array_equal(features[1], [1, 0, 1, 0])


def test_oracle_pvp_restores_true_global_pairs():
    dataset = synthetic_dataset(aligned_prop=0.5)
    oracle, labels = oracle_pvp_features(dataset, add_mask=False)
    expected = np.concatenate(dataset.views, axis=1)
    np.testing.assert_array_equal(oracle, expected)
    np.testing.assert_array_equal(labels, dataset.labels)


def test_oracle_matches_aligned_concat_when_protocol_is_aligned():
    dataset = synthetic_dataset(aligned_prop=1.0, complete_prop=1.0)
    dataset.view_sample_ids[:] = np.arange(4)[:, None]
    ordinary, ordinary_labels = official_concat_features(dataset, add_mask=True)
    oracle, oracle_labels = oracle_pvp_features(dataset, add_mask=True)
    np.testing.assert_array_equal(oracle, ordinary)
    np.testing.assert_array_equal(oracle_labels, ordinary_labels)
