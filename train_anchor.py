import argparse
import os
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Clustering import Clustering, clustering_metric
from anchor_data.clean_sure_dataset import DATASET_NAMES, CleanSUREDataset, seed_worker
from anchor_models.shared_anchor import SharedAnchorModel
from anchor_models.pbgraph import (
    align_pseudo_labels,
    apply_label_mapping,
    graph_from_assignments,
    graph_pair_loss,
    pseudo_labels_from_z,
    q_from_graph,
)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_experiment_logging(dataset_name):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    log_dir = os.path.join("./log", dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{dataset_name}_anchor_time={timestamp}.txt")
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    return log_path


def parse_args():
    parser = argparse.ArgumentParser(description="Clean shared-anchor experiment for SURE-format two-view datasets")
    parser.add_argument("--data", default="0", type=str, help="SURE dataset number (0-6) or dataset name.")
    parser.add_argument("--data-root", default="./datasets", type=str)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--aligned-prop", default=1.0, type=float)
    parser.add_argument("--complete-prop", default=1.0, type=float)
    parser.add_argument("--latent-dim", default=10, type=int)
    parser.add_argument("--hidden-dim", default=512, type=int)
    parser.add_argument("--num-anchors", default=64, type=int)
    parser.add_argument("--temperature", default=0.5, type=float)
    parser.add_argument("--lambda-rec", default=1.0, type=float)
    parser.add_argument("--lambda-self", default=1.0, type=float)
    parser.add_argument("--lambda-entropy", default=0.1, type=float)
    parser.add_argument("--lambda-balance", default=1.0, type=float)
    parser.add_argument("--lambda-pair-q", default=0.0, type=float)
    parser.add_argument("--cluster-head", default="param", choices=("param", "pbgraph"), type=str)
    parser.add_argument("--pbgraph-start-epoch", default=10, type=int)
    parser.add_argument("--pbgraph-update-interval", default=5, type=int)
    parser.add_argument("--pbgraph-ema", default=0.9, type=float)
    parser.add_argument("--pbgraph-sinkhorn-iters", default=20, type=int)
    parser.add_argument("--lambda-pseudo-q", default=0.0, type=float)
    parser.add_argument("--lambda-graph-pair", default=0.0, type=float)
    parser.add_argument("--pbgraph-pseudo-source", default="fusion_z", choices=("fusion_z", "view_z"), type=str)
    parser.add_argument("--eval-interval", default=5, type=int)
    parser.add_argument("--oracle-fusion", action="store_true", help="Report row-wise fusion in PVP/Both as oracle only.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def masked_mean(values, mask, eps=1e-8):
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def pair_valid_mask(mask):
    return (mask[:, 0] > 0) & (mask[:, 1] > 0)


def symmetric_kl(q0, q1):
    q0 = q0.clamp_min(1e-8)
    q1 = q1.clamp_min(1e-8)
    return 0.5 * ((q0 * (q0.log() - q1.log())).sum(1) + (q1 * (q1.log() - q0.log())).sum(1))


def compute_losses(batch, outputs, args, pbgraph_state=None):
    mask = batch["mask"].to(outputs["z"][0].device)
    rec_loss = torch.zeros((), device=mask.device)
    self_loss = torch.zeros((), device=mask.device)
    entropy_loss = torch.zeros((), device=mask.device)
    pair_q_loss = torch.zeros((), device=mask.device)
    pseudo_q_loss = torch.zeros((), device=mask.device)
    q_values = []

    for view_idx, x in enumerate(batch["views"]):
        x = x.to(mask.device)
        visible = mask[:, view_idx]
        rec_per_sample = F.mse_loss(outputs["x_hat"][view_idx], x, reduction="none").mean(dim=1)
        self_per_sample = F.mse_loss(outputs["z_hat"][view_idx], outputs["z"][view_idx], reduction="none").mean(dim=1)
        q = outputs["q"][view_idx]
        entropy_per_sample = -(q * torch.log(q.clamp_min(1e-8))).sum(dim=1)

        rec_loss = rec_loss + masked_mean(rec_per_sample, visible)
        self_loss = self_loss + masked_mean(self_per_sample, visible)
        entropy_loss = entropy_loss + masked_mean(entropy_per_sample, visible)
        q_values.append(q[visible > 0])

    if args.lambda_pair_q > 0 and mask.shape[1] >= 2:
        valid = pair_valid_mask(mask)
        if valid.any():
            pair_q_loss = symmetric_kl(outputs["q"][0][valid], outputs["q"][1][valid]).mean()

    if (
        args.cluster_head == "pbgraph"
        and args.lambda_pseudo_q > 0
        and pbgraph_state is not None
        and pbgraph_state.get("pseudo_onehot") is not None
    ):
        global_ids = batch["global_id"].to(mask.device)
        pseudo_onehot = pbgraph_state["pseudo_onehot"].to(mask.device)[global_ids]
        pseudo_terms = []
        for view_idx, q in enumerate(outputs["q"]):
            visible = mask[:, view_idx] > 0
            if visible.any():
                pseudo_terms.append(-(pseudo_onehot[visible] * q[visible].clamp_min(1e-8).log()).sum(1).mean())
        if pseudo_terms:
            pseudo_q_loss = torch.stack(pseudo_terms).mean()

    q_all = torch.cat(q_values, dim=0)
    mean_q = q_all.mean(dim=0)
    num_clusters = mean_q.numel()
    balance_loss = (mean_q * torch.log(mean_q.clamp_min(1e-8) * num_clusters)).sum()

    total = (
        args.lambda_rec * rec_loss
        + args.lambda_self * self_loss
        + args.lambda_entropy * entropy_loss
        + args.lambda_balance * balance_loss
        + args.lambda_pair_q * pair_q_loss
        + args.lambda_pseudo_q * pseudo_q_loss
    )
    return total, {
        "rec": rec_loss.item(),
        "self": self_loss.item(),
        "entropy": entropy_loss.item(),
        "balance": balance_loss.item(),
        "pair_q": pair_q_loss.item(),
        "pseudo_q": pseudo_q_loss.item(),
        "graph_pair": 0.0,
        "total": total.item(),
    }


def train_one_epoch(model, loader, optimizer, device, args, pbgraph_state=None):
    model.train()
    loss_sum = {"rec": 0.0, "self": 0.0, "entropy": 0.0, "balance": 0.0, "pair_q": 0.0, "pseudo_q": 0.0, "graph_pair": 0.0, "total": 0.0}
    n_batches = 0
    for batch in loader:
        views = [x.to(device) for x in batch["views"]]
        mask = batch["mask"].to(device)
        outputs = model(views, mask)
        if args.cluster_head == "pbgraph" and pbgraph_state and pbgraph_state.get("B_list") is not None:
            outputs["q"] = [q_from_graph(outputs["S"][v], pbgraph_state["B_list"][v].to(device)) for v in range(2)]
        loss, loss_dict = compute_losses(batch, outputs, args, pbgraph_state)

        graph_pair = torch.zeros((), device=device)
        if args.cluster_head == "pbgraph" and pbgraph_state and pbgraph_state.get("B_list") is not None:
            graph_pair = graph_pair_loss(pbgraph_state["B_list"])
            loss = loss + args.lambda_graph_pair * graph_pair
        loss_dict["graph_pair"] = graph_pair.item()
        loss_dict["total"] = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for key in loss_sum:
            loss_sum[key] += loss_dict[key]
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in loss_sum.items()}


def collect_outputs(model, loader, dataset, device, B_list=None):
    n_samples = len(dataset)
    latent_dim = model.latent_dim
    num_clusters = dataset.num_clusters
    num_anchors = model.num_anchors
    z_by_view = [np.zeros((n_samples, latent_dim), dtype=np.float32) for _ in range(2)]
    s_by_view = [np.zeros((n_samples, num_anchors), dtype=np.float32) for _ in range(2)]
    q_by_view = [np.zeros((n_samples, num_clusters), dtype=np.float32) for _ in range(2)]
    seen_by_view = [np.zeros(n_samples, dtype=bool) for _ in range(2)]
    z_fused_sum = np.zeros((n_samples, latent_dim), dtype=np.float32)
    q_fused_sum = np.zeros((n_samples, num_clusters), dtype=np.float32)
    s_fused_sum = np.zeros((n_samples, num_anchors), dtype=np.float32)
    fused_count = np.zeros(n_samples, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            views = [x.to(device) for x in batch["views"]]
            outputs = model(views, batch["mask"].to(device))
            if B_list is not None:
                q_list = [q_from_graph(outputs["S"][v], B_list[v]) for v in range(2)]
            else:
                q_list = outputs["q"]
            global_ids = batch["global_id"].numpy()
            view_sample_ids = batch["view_sample_ids"].numpy()
            mask_np = batch["mask"].numpy()
            for view_idx in range(2):
                z_np = outputs["z"][view_idx].cpu().numpy()
                s_np = outputs["S"][view_idx].cpu().numpy()
                q_np = q_list[view_idx].cpu().numpy()
                visible_rows = np.where(mask_np[:, view_idx] > 0)[0]
                source_ids = view_sample_ids[visible_rows, view_idx]
                z_by_view[view_idx][source_ids] = z_np[visible_rows]
                s_by_view[view_idx][source_ids] = s_np[visible_rows]
                q_by_view[view_idx][source_ids] = q_np[visible_rows]
                seen_by_view[view_idx][source_ids] = True
                z_fused_sum[global_ids[visible_rows]] += z_np[visible_rows]
                q_fused_sum[global_ids[visible_rows]] += q_np[visible_rows]
                s_fused_sum[global_ids[visible_rows]] += s_np[visible_rows]
                fused_count[global_ids[visible_rows]] += 1.0
    return {
        "z_by_view": z_by_view,
        "s_by_view": s_by_view,
        "q_by_view": q_by_view,
        "seen_by_view": seen_by_view,
        "z_fused": z_fused_sum / np.maximum(fused_count[:, None], 1.0),
        "q_fused": q_fused_sum / np.maximum(fused_count[:, None], 1.0),
        "s_fused": s_fused_sum / np.maximum(fused_count[:, None], 1.0),
        "fused_seen": fused_count > 0,
    }


def update_pbgraph(model, loader, dataset, device, args, state):
    collected = collect_outputs(model, loader, dataset, device)
    pseudo = pseudo_labels_from_z(
        [collected["z_by_view"][0], collected["z_by_view"][1]],
        dataset.mask_matrix,
        dataset.num_clusters,
        args.pbgraph_pseudo_source,
        args.seed,
    )
    if isinstance(pseudo, list):
        raw_pseudo_list = pseudo
        raw_pseudo_for_onehot = pseudo[0].copy()
        missing = raw_pseudo_for_onehot < 0
        raw_pseudo_for_onehot[missing] = pseudo[1][missing]
    else:
        raw_pseudo_list = [pseudo, pseudo]
        raw_pseudo_for_onehot = pseudo

    previous = state.get("prev_pseudo_labels")
    if previous is None:
        pseudo_for_onehot = raw_pseudo_for_onehot.copy()
        pseudo_list = [labels.copy() for labels in raw_pseudo_list]
        alignment_diag = {
            "applied": False,
            "mapping": {int(label): int(label) for label in range(dataset.num_clusters)},
            "agreement_before": 0.0,
            "agreement_after": 0.0,
            "trace_before": 0,
            "trace_after": 0,
        }
    else:
        pseudo_for_onehot, mapping, alignment_diag = align_pseudo_labels(
            previous,
            raw_pseudo_for_onehot,
            dataset.num_clusters,
        )
        pseudo_list = [apply_label_mapping(labels, mapping) for labels in raw_pseudo_list]
        alignment_diag["mapping"] = mapping

    pseudo_for_onehot = np.maximum(pseudo_for_onehot, 0)
    onehot = F.one_hot(torch.from_numpy(pseudo_for_onehot).long(), dataset.num_clusters).float()

    new_graphs = []
    for view_idx in range(2):
        valid = dataset.mask_matrix[:, view_idx] > 0
        labels_for_view = torch.from_numpy(pseudo_list[view_idx]).long()
        assignments = torch.from_numpy(collected["s_by_view"][view_idx]).float()
        new_graphs.append(
            graph_from_assignments(
                assignments,
                labels_for_view,
                torch.from_numpy(valid),
                dataset.num_clusters,
                args.pbgraph_sinkhorn_iters,
            )
        )
    if state["B_list"] is None:
        state["B_list"] = new_graphs
    else:
        state["B_list"] = [
            args.pbgraph_ema * old + (1.0 - args.pbgraph_ema) * new
            for old, new in zip(state["B_list"], new_graphs)
        ]
        state["B_list"] = [graph / graph.sum(1, keepdim=True).clamp_min(1e-8) for graph in state["B_list"]]
    state["prev_pseudo_labels"] = pseudo_for_onehot.copy()
    state["pseudo_labels"] = torch.from_numpy(pseudo_for_onehot).long()
    state["pseudo_onehot"] = onehot
    state["active"] = True
    counts = np.bincount(pseudo_for_onehot, minlength=dataset.num_clusters)
    ratios = counts / max(len(pseudo_for_onehot), 1)
    print(
        f"PBGraph update: pseudo_label_counts={counts.tolist()}, "
        f"pseudo_label_ratios={_array_string(ratios)}, "
        f"pseudo_label_max_ratio={ratios.max():.4f}, "
        f"pseudo_label_used_clusters={int((counts > 0).sum())}"
    )
    print(
        f"PBGraph label_alignment_applied={alignment_diag['applied']}, "
        f"label_alignment_mapping={alignment_diag['mapping']}, "
        f"label_agreement_before={alignment_diag['agreement_before']:.6f}, "
        f"label_agreement_after={alignment_diag['agreement_after']:.6f}, "
        f"contingency_trace_before={alignment_diag['trace_before']}, "
        f"contingency_trace_after={alignment_diag['trace_after']}"
    )
    for view_idx, graph in enumerate(state["B_list"]):
        col_sums = graph.sum(0).numpy()
        row_entropy = _entropy(graph.numpy()).mean()
        print(
            f"PBGraph view{view_idx}: B_row_entropy_mean={row_entropy:.4f}, "
            f"B_col_sums={_array_string(col_sums)}, B_col_min={col_sums.min():.4f}, "
            f"B_col_max={col_sums.max():.4f}, B_col_std={col_sums.std():.4f}"
        )
    print(f"PBGraph B_pair_mse={graph_pair_loss(state['B_list']).item():.8f}")


def _kmeans_scores(features, labels):
    _, ret = Clustering([features], labels)
    return ret["kmeans"]


def _q_scores(q, labels, n_clusters):
    pred = np.argmax(q, axis=1)
    scores, _ = clustering_metric(labels, pred, n_clusters)
    return scores


def _format_scores(prefix, scores):
    return (
        f"{prefix}: acc={scores['accuracy']:.4f}, "
        f"nmi={scores['NMI']:.4f}, ari={scores['ARI']:.4f}"
    )


def _entropy(probs, axis=1):
    probs = np.clip(probs, 1e-8, 1.0)
    return -(probs * np.log(probs)).sum(axis=axis)


def _q_diagnostics(q):
    num_clusters = q.shape[1]
    q_mean = q.mean(axis=0)
    pred = np.argmax(q, axis=1)
    counts = np.bincount(pred, minlength=num_clusters)
    ratios = counts / max(len(pred), 1)
    sample_entropy = _entropy(q).mean()
    q_mean_entropy = _entropy(q_mean[None, :], axis=1)[0]
    return {
        "q_mean": q_mean,
        "q_argmax_counts": counts,
        "q_argmax_ratios": ratios,
        "unique_predicted_clusters": np.unique(pred),
        "max_cluster_ratio": ratios.max() if len(ratios) else 0.0,
        "mean_sample_entropy": sample_entropy,
        "normalized_mean_entropy": sample_entropy / np.log(num_clusters),
        "q_mean_entropy": q_mean_entropy,
        "effective_clusters": np.exp(q_mean_entropy),
    }


def _s_diagnostics(s):
    num_anchors = s.shape[1]
    mean_s = s.mean(axis=0)
    pred = np.argmax(s, axis=1)
    counts = np.bincount(pred, minlength=num_anchors)
    top_indices = np.argsort(counts)[::-1][:10]
    top_counts = [(int(idx), int(counts[idx])) for idx in top_indices if counts[idx] > 0]
    sample_entropy = _entropy(s).mean()
    mean_s_entropy = _entropy(mean_s[None, :], axis=1)[0]
    return {
        "mean_sample_entropy": sample_entropy,
        "normalized_entropy": sample_entropy / np.log(num_anchors),
        "mean_max_probability": np.max(s, axis=1).mean(),
        "s_argmax_anchor_counts_top10": top_counts,
        "effective_anchors": np.exp(mean_s_entropy),
    }


def _array_string(values):
    return np.array2string(values, precision=4, suppress_small=True, max_line_width=160)


def _format_q_diagnostics(name, diag):
    return [
        f"{name} q diagnostics:",
        f"  q_mean={_array_string(diag['q_mean'])}",
        f"  q_argmax_counts={diag['q_argmax_counts'].tolist()}",
        f"  q_argmax_ratios={_array_string(diag['q_argmax_ratios'])}",
        f"  unique_predicted_clusters={diag['unique_predicted_clusters'].tolist()}, "
        f"max_cluster_ratio={diag['max_cluster_ratio']:.4f}",
        f"  mean_H(q_i)={diag['mean_sample_entropy']:.4f}, "
        f"norm_mean_H={diag['normalized_mean_entropy']:.4f}, "
        f"H(q_mean)={diag['q_mean_entropy']:.4f}, "
        f"effective_clusters={diag['effective_clusters']:.4f}",
    ]


def _format_s_diagnostics(name, diag):
    return [
        f"{name} S diagnostics:",
        f"  mean_H(S_i)={diag['mean_sample_entropy']:.4f}, "
        f"norm_H={diag['normalized_entropy']:.4f}, "
        f"mean_max_prob={diag['mean_max_probability']:.4f}",
        f"  S_argmax_anchor_counts_top10={diag['s_argmax_anchor_counts_top10']}, "
        f"effective_anchors={diag['effective_anchors']:.4f}",
    ]


def evaluate(model, loader, dataset, device, oracle_fusion=False, B_list=None):
    model.eval()
    n_samples = len(dataset)
    latent_dim = model.latent_dim
    num_clusters = dataset.num_clusters
    num_anchors = model.num_anchors

    z_by_view = [np.zeros((n_samples, latent_dim), dtype=np.float32) for _ in range(2)]
    q_by_view = [np.zeros((n_samples, num_clusters), dtype=np.float32) for _ in range(2)]
    s_by_view = [np.zeros((n_samples, num_anchors), dtype=np.float32) for _ in range(2)]
    seen_by_view = [np.zeros(n_samples, dtype=bool) for _ in range(2)]
    z_fused_sum = np.zeros((n_samples, latent_dim), dtype=np.float32)
    q_fused_sum = np.zeros((n_samples, num_clusters), dtype=np.float32)
    s_fused_sum = np.zeros((n_samples, num_anchors), dtype=np.float32)
    fused_count = np.zeros(n_samples, dtype=np.float32)

    with torch.no_grad():
        for batch in loader:
            views = [x.to(device) for x in batch["views"]]
            mask = batch["mask"].to(device)
            outputs = model(views, mask)
            if B_list is not None:
                outputs["q"] = [q_from_graph(outputs["S"][v], B_list[v].to(device)) for v in range(2)]
            global_ids = batch["global_id"].numpy()
            view_sample_ids = batch["view_sample_ids"].numpy()
            mask_np = batch["mask"].numpy()

            for view_idx in range(2):
                z_np = outputs["z"][view_idx].cpu().numpy()
                q_np = outputs["q"][view_idx].cpu().numpy()
                s_np = outputs["S"][view_idx].cpu().numpy()
                visible_rows = np.where(mask_np[:, view_idx] > 0)[0]
                if len(visible_rows) == 0:
                    continue

                source_ids = view_sample_ids[visible_rows, view_idx]
                z_by_view[view_idx][source_ids] = z_np[visible_rows]
                q_by_view[view_idx][source_ids] = q_np[visible_rows]
                s_by_view[view_idx][source_ids] = s_np[visible_rows]
                seen_by_view[view_idx][source_ids] = True

                z_fused_sum[global_ids[visible_rows]] += z_np[visible_rows]
                q_fused_sum[global_ids[visible_rows]] += q_np[visible_rows]
                s_fused_sum[global_ids[visible_rows]] += s_np[visible_rows]
                fused_count[global_ids[visible_rows]] += 1.0

    labels = dataset.labels
    results = {}
    diagnostics = {}
    for view_idx in range(2):
        seen = seen_by_view[view_idx]
        if seen.sum() == 0:
            continue
        results[f"view{view_idx}_z_kmeans"] = _kmeans_scores(z_by_view[view_idx][seen], labels[seen])
        results[f"view{view_idx}_q_argmax"] = _q_scores(q_by_view[view_idx][seen], labels[seen], num_clusters)
        diagnostics[f"view{view_idx}_q"] = _q_diagnostics(q_by_view[view_idx][seen])
        diagnostics[f"view{view_idx}_S"] = _s_diagnostics(s_by_view[view_idx][seen])

    can_officially_fuse = not dataset.is_pvp
    if can_officially_fuse or oracle_fusion:
        seen = fused_count > 0
        z_fused = z_fused_sum[seen] / fused_count[seen, None]
        q_fused = q_fused_sum[seen] / fused_count[seen, None]
        s_fused = s_fused_sum[seen] / fused_count[seen, None]
        prefix = "fusion" if can_officially_fuse else "oracle_fusion"
        results[f"{prefix}_z_kmeans"] = _kmeans_scores(z_fused, labels[seen])
        results[f"{prefix}_q_argmax"] = _q_scores(q_fused, labels[seen], num_clusters)
        diagnostics[f"{prefix}_q"] = _q_diagnostics(q_fused)
        diagnostics[f"{prefix}_S"] = _s_diagnostics(s_fused)

    return results, diagnostics


def main():
    args = parse_args()
    try:
        data_id = int(args.data)
    except ValueError:
        data_id = None
    if data_id is not None:
        if data_id not in DATASET_NAMES:
            raise ValueError("--data must be a SURE dataset number from 0 to 6 or a dataset name.")
        dataset_name = DATASET_NAMES[data_id]
    else:
        dataset_name = args.data
        if dataset_name not in DATASET_NAMES.values():
            raise ValueError("--data must be a SURE dataset number from 0 to 6 or a dataset name.")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    dataset = CleanSUREDataset(
        dataset_name=dataset_name,
        data_root=args.data_root,
        aligned_prop=args.aligned_prop,
        complete_prop=args.complete_prop,
        seed=args.seed,
    )
    log_path = setup_experiment_logging(dataset_name)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = SharedAnchorModel(
        view_dims=dataset.view_dims,
        num_clusters=dataset.num_clusters,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_anchors=args.num_anchors,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("==========")
    print(f"Args: {args}")
    print(f"Log file: {log_path}")
    print(f"Device: {device}")
    print(f"Dataset={dataset_name}, samples={len(dataset)}, view_dims={dataset.view_dims}, clusters={dataset.num_clusters}")
    print(f"PVP={dataset.is_pvp}, PSP={dataset.is_psp}")
    if args.cluster_head == "pbgraph" and args.num_anchors < dataset.num_clusters:
        print("Warning: num_anchors < num_clusters; PBGraph may be under-capacity.")
    pbgraph_state = {
        "active": False,
        "pseudo_labels": None,
        "prev_pseudo_labels": None,
        "pseudo_onehot": None,
        "B_list": None,
    }
    print("==========")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        loss_dict = train_one_epoch(model, train_loader, optimizer, device, args, pbgraph_state)
        if (
            args.cluster_head == "pbgraph"
            and epoch >= args.pbgraph_start_epoch
            and (epoch - args.pbgraph_start_epoch) % max(args.pbgraph_update_interval, 1) == 0
        ):
            update_pbgraph(model, eval_loader, dataset, device, args, pbgraph_state)
        print(
            f"Epoch {epoch:03d}: total={loss_dict['total']:.4f}, rec={loss_dict['rec']:.4f}, "
            f"self={loss_dict['self']:.4f}, entropy={loss_dict['entropy']:.4f}, "
            f"balance={loss_dict['balance']:.4f}, pair_q={loss_dict['pair_q']:.4f}, "
            f"pseudo_q={loss_dict['pseudo_q']:.4f}, graph_pair={loss_dict['graph_pair']:.8f}, "
            f"pbgraph_active={pbgraph_state['active']}"
        )

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            results, diagnostics = evaluate(
                model, eval_loader, dataset, device,
                oracle_fusion=args.oracle_fusion,
                B_list=pbgraph_state["B_list"] if pbgraph_state["active"] else None,
            )
            for name, scores in results.items():
                print(_format_scores(name, scores))
            for name, diag in diagnostics.items():
                if name.endswith("_q"):
                    for line in _format_q_diagnostics(name, diag):
                        print(line)
                elif name.endswith("_S"):
                    for line in _format_s_diagnostics(name, diag):
                        print(line)
            if dataset.is_pvp and not args.oracle_fusion:
                print("PVP/Both: row-wise fusion is disabled. Use --oracle-fusion to report it as non-official.")

    print(f"Finished in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
