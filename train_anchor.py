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
from anchor_data.clean_sure_dataset import CleanSUREScene15Dataset, seed_worker
from anchor_models.shared_anchor import SharedAnchorModel


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
    parser = argparse.ArgumentParser(description="Clean shared-anchor experiment for Scene15")
    parser.add_argument("--data", default="0", type=str, help="Only Scene15/0 is supported in the first version.")
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


def compute_losses(batch, outputs, args):
    mask = batch["mask"].to(outputs["z"][0].device)
    rec_loss = torch.zeros((), device=mask.device)
    self_loss = torch.zeros((), device=mask.device)
    entropy_loss = torch.zeros((), device=mask.device)
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

    q_all = torch.cat(q_values, dim=0)
    mean_q = q_all.mean(dim=0)
    num_clusters = mean_q.numel()
    balance_loss = (mean_q * torch.log(mean_q.clamp_min(1e-8) * num_clusters)).sum()

    total = (
        args.lambda_rec * rec_loss
        + args.lambda_self * self_loss
        + args.lambda_entropy * entropy_loss
        + args.lambda_balance * balance_loss
    )
    return total, {
        "rec": rec_loss.item(),
        "self": self_loss.item(),
        "entropy": entropy_loss.item(),
        "balance": balance_loss.item(),
        "total": total.item(),
    }


def train_one_epoch(model, loader, optimizer, device, args):
    model.train()
    loss_sum = {"rec": 0.0, "self": 0.0, "entropy": 0.0, "balance": 0.0, "total": 0.0}
    n_batches = 0
    for batch in loader:
        views = [x.to(device) for x in batch["views"]]
        mask = batch["mask"].to(device)
        outputs = model(views, mask)
        loss, loss_dict = compute_losses(batch, outputs, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for key in loss_sum:
            loss_sum[key] += loss_dict[key]
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in loss_sum.items()}


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


def evaluate(model, loader, dataset, device, oracle_fusion=False):
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
    if args.data not in ("0", "Scene15"):
        raise ValueError("First clean anchor version only supports Scene15: use --data 0 or --data Scene15.")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    dataset = CleanSUREScene15Dataset(
        data_root=args.data_root,
        aligned_prop=args.aligned_prop,
        complete_prop=args.complete_prop,
        seed=args.seed,
    )
    log_path = setup_experiment_logging("Scene15")
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
    print(f"Scene15 samples={len(dataset)}, view_dims={dataset.view_dims}, clusters={dataset.num_clusters}")
    print(f"PVP={dataset.is_pvp}, PSP={dataset.is_psp}")
    print("==========")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        loss_dict = train_one_epoch(model, train_loader, optimizer, device, args)
        print(
            f"Epoch {epoch:03d}: total={loss_dict['total']:.4f}, rec={loss_dict['rec']:.4f}, "
            f"self={loss_dict['self']:.4f}, entropy={loss_dict['entropy']:.4f}, "
            f"balance={loss_dict['balance']:.4f}"
        )

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            results, diagnostics = evaluate(model, eval_loader, dataset, device, oracle_fusion=args.oracle_fusion)
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
