import os
import random

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


DATASET_NAMES = {
    0: "Scene15",
    1: "Caltech101",
    2: "Reuters_dim10",
    3: "NoisyMNIST-30000",
    4: "2view-caltech101-8677sample",
    5: "AWA-7view-10158sample",
    6: "MNIST-USPS",
}


def _normalize_like_sure(x):
    x = np.asarray(x, dtype=np.float32)
    minimum = np.min(x, axis=0, keepdims=True)
    denominator = np.max(x, axis=0, keepdims=True) - minimum
    denominator[denominator == 0] = 1.0
    return (x - minimum) / denominator


def _load_sure_dataset(dataset_name, data_root):
    mat_path = os.path.join(data_root, f"{dataset_name}.mat")
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Dataset file not found: {mat_path}")
    mat = sio.loadmat(mat_path)

    if dataset_name == "Scene15":
        views = [mat["X"][0][0], mat["X"][0][1]]
        labels = mat["Y"]
    elif dataset_name == "Caltech101":
        views = [mat["X"][0][3], mat["X"][0][4]]
        labels = mat["Y"]
    elif dataset_name == "Reuters_dim10":
        views = [
            np.vstack((mat["x_train"][0], mat["x_test"][0])),
            np.vstack((mat["x_train"][1], mat["x_test"][1])),
        ]
        views = [_normalize_like_sure(view) for view in views]
        labels = np.hstack((mat["y_train"], mat["y_test"]))
    elif dataset_name == "NoisyMNIST-30000":
        views = [mat["X1"], mat["X2"]]
        labels = mat["Y"]
    elif dataset_name == "2view-caltech101-8677sample":
        views = [mat["X"][0][0].T, mat["X"][0][1].T]
        labels = mat["gt"]
    elif dataset_name == "MNIST-USPS":
        views = [mat["X1"], _normalize_like_sure(mat["X2"])]
        labels = mat["Y"]
    elif dataset_name == "AWA-7view-10158sample":
        # Match original SURE: use its sixth and seventh views in this two-view model.
        views = [mat["X"][0][5].T, mat["X"][0][6].T]
        labels = mat["gt"]
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    views = [np.asarray(view, dtype=np.float32) for view in views]
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    labels = labels - labels.min()
    if len(views) != 2:
        raise ValueError(f"PBGraph-v1 expects two selected views, got {len(views)}")
    if any(view.ndim != 2 or view.shape[0] != len(labels) for view in views):
        raise ValueError(
            f"Invalid view/label shapes for {dataset_name}: "
            f"views={[view.shape for view in views]}, labels={labels.shape}"
        )
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
    complete_indices = set(rng.choice(n_samples, size=n_complete, replace=False).tolist())
    for index in range(n_samples):
        if index not in complete_indices:
            mask[index, int(rng.integers(0, 2))] = 0.0
    return mask


class CleanSUREDataset(Dataset):
    """Two-view clean protocol with explicit global ids and PVP/PSP metadata."""

    def __init__(self, dataset_name="Scene15", data_root="./datasets", aligned_prop=1.0, complete_prop=1.0, seed=0):
        if dataset_name not in DATASET_NAMES.values():
            raise ValueError(f"Unsupported dataset name: {dataset_name}")
        if not 0.0 < aligned_prop <= 1.0 or not 0.0 < complete_prop <= 1.0:
            raise ValueError("aligned_prop and complete_prop must be in (0, 1].")

        self.dataset_name = dataset_name
        self.data_root = data_root
        self.aligned_prop = float(aligned_prop)
        self.complete_prop = float(complete_prop)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.views, self.labels = _load_sure_dataset(dataset_name, data_root)
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
            np.where(self.mask_matrix[:, view_idx] > 0)[0].astype(np.int64)
            for view_idx in range(self.num_views)
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


class CleanSUREScene15Dataset(CleanSUREDataset):
    """Backward-compatible Scene15 alias."""

    def __init__(self, data_root="./datasets", aligned_prop=1.0, complete_prop=1.0, seed=0):
        super().__init__("Scene15", data_root, aligned_prop, complete_prop, seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
