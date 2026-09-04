"""Protocol-calibration baselines for the clean SURE two-view protocol.

This script is deliberately independent from the SURE/PBGraph training code.
Labels are used only by the final metric calculation; all row restoration and
fusion decisions use the dataset's explicit sample-ID metadata.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

# Allow direct execution from the repository root on Windows.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Clustering import clustering_metric
from anchor_data.clean_sure_dataset import DATASET_NAMES, CleanSUREDataset


def resolve_dataset_name(data):
    if isinstance(data, int) or str(data).isdigit():
        try:
            return DATASET_NAMES[int(data)]
        except KeyError as exc:
            raise ValueError(f"Unsupported data id: {data}") from exc
    return str(data)


def _metric_row(features, labels, num_clusters, seed):
    if len(features) < num_clusters:
        raise ValueError("Number of samples must be at least num_clusters.")
    assignments = KMeans(
        n_clusters=num_clusters, n_init=20, random_state=int(seed)
    ).fit_predict(np.asarray(features, dtype=np.float32))
    scores, _ = clustering_metric(labels, assignments, num_clusters)
    return {"acc": float(scores["accuracy"]), "nmi": float(scores["NMI"]), "ari": float(scores["ARI"])}


def single_view_inputs(dataset, view_idx):
    """Return visible source rows and their labels in protocol-row order."""
    valid = dataset.mask_matrix[:, view_idx] > 0
    source_ids = dataset.view_sample_ids[valid, view_idx].astype(np.int64)
    return dataset.views[view_idx][source_ids], dataset.labels[source_ids], valid, source_ids


def paired_mask(dataset):
    mask = getattr(dataset, "paired_mask_matrix", None)
    if mask is None:
        mask = np.zeros(len(dataset), dtype=bool)
        for row in getattr(dataset, "paired_indices", []):
            mask[int(row)] = True
        mask &= np.asarray(dataset.mask_matrix).all(axis=1)
    return np.asarray(mask, dtype=bool)


def official_concat_allowed(dataset):
    return not bool(dataset.is_pvp)


def _rowwise_features(dataset, add_mask):
    blocks = []
    n_samples = getattr(dataset, "n_samples", None)
    if n_samples is None:
        n_samples = len(dataset)
    for row in range(n_samples):
        row_blocks = []
        for view_idx in range(dataset.num_views):
            if dataset.mask_matrix[row, view_idx] > 0:
                source_id = dataset.view_sample_ids[row, view_idx]
                row_blocks.append(dataset.views[view_idx][source_id])
            else:
                row_blocks.append(np.zeros(dataset.view_dims[view_idx], dtype=np.float32))
        if add_mask:
            row_blocks.append(dataset.mask_matrix[row].astype(np.float32))
        blocks.append(np.concatenate(row_blocks))
    return np.asarray(blocks, dtype=np.float32), dataset.labels.copy()


def official_concat_features(dataset, add_mask=True):
    if not official_concat_allowed(dataset):
        raise ValueError("Row-wise concat is not an official baseline when PVP is active.")
    return _rowwise_features(dataset, add_mask)


def aligned_subset_features(dataset):
    rows = paired_mask(dataset)
    blocks = []
    for view_idx in range(dataset.num_views):
        source_ids = dataset.view_sample_ids[rows, view_idx]
        blocks.append(dataset.views[view_idx][source_ids])
    return np.concatenate(blocks, axis=1), dataset.labels[rows], int(rows.sum())


def oracle_pvp_features(dataset, add_mask=True):
    """Build true global-ID pairs from the raw source-order view arrays.

    CleanSUREDataset keeps ``views[v]`` in original source-ID order. Its
    ``view_sample_ids[row, v]`` describes which source row is placed at a
    protocol row, so using that value as the destination would reconstruct the
    permuted row-wise pairing rather than the oracle pairing.
    """
    n_samples = getattr(dataset, "n_samples", None)
    if n_samples is None:
        n_samples = len(dataset)
    n_samples = int(n_samples)
    restored = [
        np.asarray(dataset.views[view_idx][:n_samples], dtype=np.float32).copy()
        for view_idx in range(dataset.num_views)
    ]
    restored_mask = np.zeros((n_samples, dataset.num_views), dtype=np.float32)
    for row, global_id in enumerate(np.asarray(dataset.global_ids, dtype=np.int64)):
        for view_idx in range(dataset.num_views):
            if dataset.mask_matrix[row, view_idx] > 0:
                restored_mask[global_id, view_idx] = 1.0
    # The source arrays are already indexed by the true global/sample ID.
    blocks = restored + ([restored_mask] if add_mask else [])
    return np.concatenate(blocks, axis=1), dataset.labels[:n_samples].copy()


def choose_bsv(view_results):
    """Select one complete view and reuse it for all three BSV metrics."""
    selected = max(view_results.items(), key=lambda item: (item[1]["acc"], item[0]))
    return selected[0], dict(selected[1])


def _result(dataset, baseline, features, labels, num_clusters, seed, official=True, note=""):
    scores = _metric_row(features, labels, num_clusters, seed)
    return {
        "dataset": dataset.dataset_name,
        "aligned_prop": dataset.aligned_prop,
        "complete_prop": dataset.complete_prop,
        "seed": seed,
        "baseline": baseline,
        "samples": len(labels),
        "acc": scores["acc"],
        "nmi": scores["nmi"],
        "ari": scores["ari"],
        "official": bool(official),
        "note": note,
    }


def run_seed(args, seed, print_warning=True):
    dataset_name = resolve_dataset_name(args.data)
    dataset = CleanSUREDataset(
        dataset_name, args.data_root, args.aligned_prop, args.complete_prop, seed
    )
    results = []
    view_results = {}
    for view_idx in range(dataset.num_views):
        features, labels, _, _ = single_view_inputs(dataset, view_idx)
        view_results[f"View{view_idx}-KMeans"] = _metric_row(
            features, labels, args.num_clusters or dataset.num_clusters, seed
        )
        results.append(_result(dataset, f"View{view_idx}-KMeans", features, labels,
                               args.num_clusters or dataset.num_clusters, seed))

    bsv_name, bsv_scores = choose_bsv(view_results)
    results.append({**results[0], "baseline": "BSV", "acc": bsv_scores["acc"],
                    "nmi": bsv_scores["nmi"], "ari": bsv_scores["ari"],
                    "note": f"selected={bsv_name}"})

    clusters = args.num_clusters or dataset.num_clusters
    if official_concat_allowed(dataset):
        features, labels = official_concat_features(dataset, args.concat_add_mask)
        name = "Concat" if not dataset.is_psp else "Concat-ZeroFill+Mask"
        results.append(_result(dataset, name, features, labels, clusters, seed))
    elif print_warning:
        print("PVP detected: row-wise concat is disabled because rows are not guaranteed to be paired. "
              "Use Concat-AlignedSubset for diagnostic or --oracle-pvp for a non-official upper bound.")

    if dataset.is_pvp or dataset.is_psp:
        features, labels, count = aligned_subset_features(dataset)
        if count >= clusters:
            results.append(_result(dataset, "Concat-AlignedSubset (diagnostic)", features, labels,
                                   clusters, seed, official=False,
                                   note=f"paired_samples={count}"))

    if args.oracle_pvp:
        if dataset.complete_prop >= 1.0:
            oracle_features, oracle_labels = oracle_pvp_features(dataset, args.concat_add_mask)
            print(f"oracle_num_samples={len(oracle_labels)}")
            print(f"oracle_first_10_global_ids={dataset.global_ids[:10].tolist()}")
            print(f"oracle_x0_shape={tuple(dataset.views[0].shape)}")
            print(f"oracle_x1_shape={tuple(dataset.views[1].shape)}")
            print(f"oracle_label_first_10={oracle_labels[:10].tolist()}")
            features, labels = oracle_features, oracle_labels
        else:
            features, labels = oracle_pvp_features(dataset, args.concat_add_mask)
        results.append(_result(dataset, "Oracle-PVP-Concat (non-official)", features, labels,
                               clusters, seed, official=False,
                               note="global-id restored upper bound"))
    return results


def _print_results(rows):
    for row in rows:
        print(f"{row['baseline']}: samples={row['samples']} acc={row['acc']:.4f} "
              f"nmi={row['nmi']:.4f} ari={row['ari']:.4f} official={row['official']}")


def _aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["baseline"]].append(row)
    print("\nAcross-seed mean +/- std:")
    aggregate = []
    for baseline, values in grouped.items():
        means = {key: float(np.mean([item[key] for item in values])) for key in ("acc", "nmi", "ari")}
        stds = {key: float(np.std([item[key] for item in values])) for key in ("acc", "nmi", "ari")}
        official = values[0]["official"]
        print(f"{baseline}: acc={means['acc']:.4f} +/- {stds['acc']:.4f}, "
              f"nmi={means['nmi']:.4f} +/- {stds['nmi']:.4f}, "
              f"ari={means['ari']:.4f} +/- {stds['ari']:.4f} official={official}")
        aggregate.append({**values[0], "seed": "mean", "acc": means["acc"],
                          "nmi": means["nmi"], "ari": means["ari"],
                          "note": "std=" + repr(stds)})
    return aggregate


def build_parser():
    parser = argparse.ArgumentParser(description="Protocol-calibration baselines")
    parser.add_argument("--data", default=0)
    parser.add_argument("--data-root", default="./datasets")
    parser.add_argument("--aligned-prop", type=float, default=1.0)
    parser.add_argument("--complete-prop", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--num-clusters", type=int, default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--oracle-pvp", action="store_true", default=False)
    parser.add_argument("--concat-add-mask", action="store_true", default=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    seeds = args.seeds if args.seeds else [args.seed]
    print(f"Dataset={resolve_dataset_name(args.data)}, aligned_prop={args.aligned_prop}, "
          f"complete_prop={args.complete_prop}, seeds={seeds}")
    rows = []
    for seed in seeds:
        print(f"\nSeed={seed}")
        seed_rows = run_seed(args, seed)
        _print_results(seed_rows)
        rows.extend(seed_rows)
    aggregate = _aggregate(rows) if len(seeds) > 1 else []
    if args.output_csv:
        output_dir = os.path.dirname(os.path.abspath(args.output_csv))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
            fields = ["dataset", "aligned_prop", "complete_prop", "seed", "baseline",
                      "samples", "acc", "nmi", "ari", "official", "note"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows + aggregate)
        print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
