import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(in_dim, hidden_dim, out_dim, dropout=0.0):
    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.extend([
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
    ])
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class SharedAnchorModel(nn.Module):
    def __init__(
        self,
        view_dims,
        num_clusters,
        latent_dim=10,
        hidden_dim=512,
        num_anchors=64,
        temperature=0.5,
        dropout=0.0,
    ):
        super().__init__()
        self.view_dims = list(view_dims)
        self.num_views = len(self.view_dims)
        self.num_clusters = int(num_clusters)
        self.latent_dim = int(latent_dim)
        self.num_anchors = int(num_anchors)
        self.temperature = float(temperature)

        self.encoders = nn.ModuleList([
            _make_mlp(view_dim, hidden_dim, latent_dim, dropout=dropout) for view_dim in self.view_dims
        ])
        self.decoders = nn.ModuleList([
            _make_mlp(latent_dim, hidden_dim, view_dim, dropout=dropout) for view_dim in self.view_dims
        ])

        self.anchors = nn.Parameter(torch.randn(self.num_anchors, self.latent_dim) * 0.02)
        self.cluster_matrix = nn.Parameter(torch.randn(self.num_clusters, self.num_anchors) * 0.02)

    def forward(self, views, mask=None):
        z_list = []
        x_hat_list = []
        s_list = []
        z_hat_list = []
        q_list = []

        anchors = F.normalize(self.anchors, dim=1)
        for view_idx, x in enumerate(views):
            z = self.encoders[view_idx](x)
            x_hat = self.decoders[view_idx](z)

            z_norm = F.normalize(z, dim=1)
            logits = torch.matmul(z_norm, anchors.t()) / self.temperature
            s = torch.softmax(logits, dim=1)
            z_hat = torch.matmul(s, self.anchors)
            q = torch.softmax(torch.matmul(s, self.cluster_matrix.t()), dim=1)

            z_list.append(z)
            x_hat_list.append(x_hat)
            s_list.append(s)
            z_hat_list.append(z_hat)
            q_list.append(q)

        return {
            "z": z_list,
            "x_hat": x_hat_list,
            "S": s_list,
            "z_hat": z_hat_list,
            "q": q_list,
            "anchors": self.anchors,
            "cluster_matrix": self.cluster_matrix,
        }
