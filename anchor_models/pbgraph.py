import numpy as np
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def align_pseudo_labels(old_labels, new_labels, num_clusters):
    """Align new KMeans ids to the previous ids without using ground truth."""
    old_labels = np.asarray(old_labels, dtype=np.int64).reshape(-1)
    new_labels = np.asarray(new_labels, dtype=np.int64).reshape(-1)
    if old_labels.shape != new_labels.shape:
        raise ValueError(f"Pseudo-label shapes differ: {old_labels.shape} vs {new_labels.shape}")

    valid = (
        (old_labels >= 0) & (old_labels < num_clusters)
        & (new_labels >= 0) & (new_labels < num_clusters)
    )
    contingency = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    np.add.at(contingency, (old_labels[valid], new_labels[valid]), 1)
    row_ind = col_ind = None
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-contingency)
    except ImportError:
        # Greedy fallback: each old/new cluster can be used at most once.
        mapping = {}
        unused_old = set(range(num_clusters))
        unused_new = set(range(num_clusters))
        for flat_index in np.argsort(contingency.ravel())[::-1]:
            old_id, new_id = np.unravel_index(flat_index, contingency.shape)
            if old_id in unused_old and new_id in unused_new:
                mapping[int(new_id)] = int(old_id)
                unused_old.remove(old_id)
                unused_new.remove(new_id)
            if not unused_old or not unused_new:
                break
        for new_id, old_id in zip(sorted(unused_new), sorted(unused_old)):
            mapping[int(new_id)] = int(old_id)
        aligned = np.array([mapping.get(int(label), int(label)) if label >= 0 else -1 for label in new_labels])
        trace_after = sum(contingency[mapping[new_id], new_id] for new_id in mapping)
        return aligned, mapping, {
            "applied": True,
            "agreement_before": float(np.mean(old_labels[valid] == new_labels[valid])) if valid.any() else 0.0,
            "agreement_after": float(np.mean(old_labels[valid] == aligned[valid])) if valid.any() else 0.0,
            "trace_before": int(np.trace(contingency)),
            "trace_after": int(trace_after),
        }

    mapping = {int(new_id): int(old_id) for old_id, new_id in zip(row_ind, col_ind)}
    aligned = np.array([mapping.get(int(label), int(label)) if label >= 0 else -1 for label in new_labels])
    trace_after = sum(contingency[old_id, new_id] for old_id, new_id in zip(row_ind, col_ind))
    return aligned, mapping, {
        "applied": True,
        "agreement_before": float(np.mean(old_labels[valid] == new_labels[valid])) if valid.any() else 0.0,
        "agreement_after": float(np.mean(old_labels[valid] == aligned[valid])) if valid.any() else 0.0,
        "trace_before": int(np.trace(contingency)),
        "trace_after": int(trace_after),
    }


def apply_label_mapping(labels, mapping):
    labels = np.asarray(labels, dtype=np.int64)
    return np.array([mapping.get(int(label), int(label)) if label >= 0 else -1 for label in labels])


def row_normalize(matrix, eps=1e-8):
    return matrix / matrix.sum(dim=1, keepdim=True).clamp_min(eps)


def balanced_anchor_cluster_graph(counts, num_iters=20, eps=1e-8):
    """Approximate row-stochastic, column-balanced anchor-to-cluster graph."""
    graph = counts.clamp_min(eps)
    for _ in range(max(int(num_iters), 1)):
        graph = row_normalize(graph, eps)
        target = graph.size(0) / graph.size(1)
        graph = graph * (target / graph.sum(dim=0, keepdim=True).clamp_min(eps))
    return row_normalize(graph, eps)


def graph_from_assignments(assignments, pseudo_labels, visible, num_clusters, sinkhorn_iters):
    assignments = assignments[visible]
    pseudo_labels = pseudo_labels[visible]
    if assignments.numel() == 0:
        return torch.full(
            (assignments.shape[1], num_clusters),
            1.0 / num_clusters,
            device=assignments.device,
            dtype=assignments.dtype,
        )
    one_hot = F.one_hot(pseudo_labels.long(), num_classes=num_clusters).to(assignments.dtype)
    counts = assignments.t().matmul(one_hot).add(1e-8)
    return balanced_anchor_cluster_graph(counts, sinkhorn_iters)


def pseudo_labels_from_z(z_by_view, mask, num_clusters, source="fusion_z", seed=0):
    visible = mask.astype(bool)
    if source == "view_z":
        labels = []
        for view_idx, z in enumerate(z_by_view):
            valid = visible[:, view_idx]
            pred = np.full(len(mask), -1, dtype=np.int64)
            pred[valid] = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed).fit_predict(z[valid])
            labels.append(pred)
        return labels

    z_sum = np.zeros_like(z_by_view[0], dtype=np.float32)
    count = np.zeros((len(mask), 1), dtype=np.float32)
    for view_idx, z in enumerate(z_by_view):
        valid = visible[:, view_idx]
        z_sum[valid] += z[valid]
        count[valid] += 1.0
    z_fusion = z_sum / np.maximum(count, 1.0)
    return KMeans(n_clusters=num_clusters, n_init=10, random_state=seed).fit_predict(z_fusion)


def pseudo_labels_from_features(features, valid, num_clusters, seed=0):
    labels = np.full(len(features), -1, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    if valid.any():
        labels[valid] = KMeans(
            n_clusters=num_clusters,
            n_init=10,
            random_state=seed,
        ).fit_predict(features[valid])
    return labels


def pair_aware_fusion_features(z_by_view, view_sample_ids, mask, paired_indices):
    """Fuse only known pairs; use one visible source for unpaired rows."""
    n_samples = len(mask)
    latent_dim = z_by_view[0].shape[1]
    paired = np.zeros(n_samples, dtype=bool)
    paired[np.asarray(paired_indices, dtype=np.int64)] = True
    features = np.zeros((n_samples, latent_dim), dtype=np.float32)
    valid = np.zeros(n_samples, dtype=bool)
    aligned_count = 0
    single_view_count = 0

    for row in range(n_samples):
        visible_views = np.flatnonzero(mask[row] > 0)
        if len(visible_views) == 0:
            continue
        if paired[row] and len(visible_views) >= 2:
            # Known aligned rows use the same global source id in both views.
            features[row] = 0.5 * (z_by_view[0][row] + z_by_view[1][row])
            aligned_count += 1
        else:
            view_idx = int(visible_views[0])
            source_id = int(view_sample_ids[row, view_idx])
            features[row] = z_by_view[view_idx][source_id]
            single_view_count += 1
        valid[row] = True

    return features, valid, {
        "num_aligned": aligned_count,
        "num_single_view": single_view_count,
        "num_invalid": int((~valid).sum()),
    }


def q_from_graph(assignments, graph):
    q = assignments.matmul(graph).clamp_min(1e-8)
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)


def graph_pair_loss(graphs):
    if len(graphs) < 2:
        return torch.zeros((), device=graphs[0].device)
    losses = []
    for left in range(len(graphs)):
        for right in range(left + 1, len(graphs)):
            losses.append((graphs[left] - graphs[right]).pow(2).mean())
    return torch.stack(losses).mean()
