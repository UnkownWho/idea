import os
import random

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def _load_scene15(data_root):
    mat_path = os.path.join(data_root, "Scene15.mat")
    mat = sio.loadmat(mat_path)
    views = [mat["X"][0][0].astype(np.float32), mat["X"][0][1].astype(np.float32)]
    labels = np.squeeze(mat["Y"]).astype(np.int64)
    labels = labels - labels.min()
    return views, labels


def _build_pvp_permutation(n_samples, aligned_prop, rng):
    global_ids = np.arange(n_samples, dtype=np.int64)
    n_aligned = int(np.ceil(aligned_prop * n_samples))
    shuffled_rows = rng.permutation(global_ids)
    aligned_indices = np.sort(shuffled_rows[:n_aligned])
    unaligned_indices = np.sort(shuffled_rows[n_aligned:])

    permutation = np.stack([global_ids.copy(), global_ids.copy()], axis=0)
    shuffle_idx = np.arange(len(unaligned_indices), dtype=np.int64)
    if len(unaligned_indices) > 1:
        shuffle_idx = rng.permutation(len(unaligned_indices))
        permutation[1, unaligned_indices] = unaligned_indices[shuffle_idx]

    inverse_permutation = np.empty_like(permutation)
    for view_idx in range(2):
        inverse_permutation[view_idx, permutation[view_idx]] = global_ids

    return permutation, inverse_permutation, aligned_indices, unaligned_indices, shuffle_idx


def _build_psp_mask(n_samples, complete_prop, rng):
    mask = np.ones((n_samples, 2), dtype=np.float32)
    if complete_prop >= 1.0:
        return mask

    n_complete = int(np.ceil(complete_prop * n_samples))
    all_indices = np.arange(n_samples, dtype=np.int64)
    complete_indices = set(rng.choice(all_indices, size=n_complete, replace=False).tolist())
    for idx in all_indices:
        if idx in complete_indices:
            continue
        missing_view = int(rng.integers(0, 2))
        mask[idx, missing_view] = 0.0
    return mask


class CleanSUREScene15Dataset(Dataset):
    """Scene15 two-view dataset with explicit PVP/PSP metadata.

    Each row is anchored by global_id/view0. In PVP settings, view1 at the same
    row may come from a different original sample, recorded in view_sample_ids.
    """

    def __init__(self, data_root="./datasets", aligned_prop=1.0, complete_prop=1.0, seed=0):
        if not 0.0 < aligned_prop <= 1.0:
            raise ValueError("aligned_prop must be in (0, 1].")
        if not 0.0 < complete_prop <= 1.0:
            raise ValueError("complete_prop must be in (0, 1].")

        self.data_root = data_root
        self.aligned_prop = float(aligned_prop)
        self.complete_prop = float(complete_prop)
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        self.views, self.labels = _load_scene15(self.data_root)
        self.num_views = 2
        self.view_dims = [view.shape[1] for view in self.views]
        self.n_samples = len(self.labels)
        self.global_ids = np.arange(self.n_samples, dtype=np.int64)
        self.num_clusters = int(np.unique(self.labels).size)

        (
            self.permutation,
            self.inverse_permutation,
            self.aligned_indices,
            self.unaligned_indices,
            self.shuffle_idx,
        ) = _build_pvp_permutation(self.n_samples, self.aligned_prop, rng)
        self.view_sample_ids = self.permutation.T.copy()

        self.mask_matrix = _build_psp_mask(self.n_samples, self.complete_prop, rng)
        self.valid_indices_per_view = [
            np.where(self.mask_matrix[:, view_idx] > 0)[0].astype(np.int64) for view_idx in range(self.num_views)
        ]
        self.paired_indices = self.aligned_indices.copy()

        self.is_pvp = self.aligned_prop < 1.0
        self.is_psp = self.complete_prop < 1.0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        index = int(index)
        mask = self.mask_matrix[index]
        sample_views = []
        for view_idx in range(self.num_views):
            source_id = int(self.view_sample_ids[index, view_idx])
            if mask[view_idx] > 0:
                value = self.views[view_idx][source_id]
            else:
                value = np.zeros(self.view_dims[view_idx], dtype=np.float32)
            sample_views.append(torch.from_numpy(value.astype(np.float32)))

        return {
            "views": sample_views,
            "mask": torch.from_numpy(mask.astype(np.float32)),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "global_id": torch.tensor(self.global_ids[index], dtype=torch.long),
            "view_sample_ids": torch.from_numpy(self.view_sample_ids[index].astype(np.int64)),
        }


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
