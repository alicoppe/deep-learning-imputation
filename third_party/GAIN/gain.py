"""GAIN — Generative Adversarial Imputation Nets (Yoon, Jordon, van der Schaar, ICML 2018).

PyTorch reimplementation with the same public signature as the authors' reference
TensorFlow code (jsyoon0823/GAIN): ``gain(data_x, gain_parameters) -> imputed_data``.

The original repo targets TensorFlow 1.x, which has no build for modern Python;
this port reproduces the same architecture and training objective in torch so it
runs on the project's existing stack. Behaviour matches the paper:

  - min-max normalisation per column (fit on observed entries)
  - generator / discriminator: [2*dim -> dim -> dim -> dim], ReLU + sigmoid out
  - hint mechanism with rate ``hint_rate``
  - loss: D = -mean(M·logD + (1-M)·log(1-D));  G = G_adv + alpha·MSE_obs
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _normalization(data: np.ndarray):
    """Min-max normalise each column using observed (non-NaN) values."""
    dim = data.shape[1]
    norm = data.copy().astype(float)
    mins = np.zeros(dim)
    maxs = np.zeros(dim)
    for j in range(dim):
        col = norm[:, j]
        mn = np.nanmin(col)
        mx = np.nanmax(col)
        mins[j] = mn
        maxs[j] = mx
        norm[:, j] = (col - mn) / (mx - mn + 1e-6)
    return norm, {"min": mins, "max": maxs}


def _renormalization(norm_data: np.ndarray, params: dict) -> np.ndarray:
    out = norm_data.copy()
    for j in range(norm_data.shape[1]):
        out[:, j] = out[:, j] * (params["max"][j] - params["min"][j] + 1e-6) + params["min"][j]
    return out


class _Net(nn.Module):
    """Shared MLP shape for both generator and discriminator: 2*dim -> dim -> dim -> dim."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, aux], dim=1))


def gain(data_x: np.ndarray, gain_parameters: dict) -> np.ndarray:
    """Impute missing values (np.nan) in data_x with GAIN. Returns a filled copy."""
    batch_size = int(gain_parameters.get("batch_size", 128))
    hint_rate = float(gain_parameters.get("hint_rate", 0.9))
    alpha = float(gain_parameters.get("alpha", 100.0))
    iterations = int(gain_parameters.get("iterations", 10000))
    seed = gain_parameters.get("seed", None)

    if seed is not None:
        torch.manual_seed(int(seed))
        rng = np.random.default_rng(int(seed))
    else:
        rng = np.random.default_rng()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    data_x = np.asarray(data_x, dtype=float)
    no, dim = data_x.shape
    mask = 1.0 - np.isnan(data_x)  # 1 = observed, 0 = missing

    norm_data, norm_params = _normalization(data_x)
    norm_data_filled = np.nan_to_num(norm_data, nan=0.0)

    G = _Net(dim).to(device)
    D = _Net(dim).to(device)
    optG = torch.optim.Adam(G.parameters())
    optD = torch.optim.Adam(D.parameters())
    bce = nn.BCELoss()

    X_all = torch.tensor(norm_data_filled, dtype=torch.float32, device=device)
    M_all = torch.tensor(mask, dtype=torch.float32, device=device)
    eff_batch = min(batch_size, no)

    for _ in range(iterations):
        idx = rng.choice(no, eff_batch, replace=False)
        X = X_all[idx]
        M = M_all[idx]
        Z = torch.tensor(rng.uniform(0.0, 0.01, size=(eff_batch, dim)),
                         dtype=torch.float32, device=device)
        H_temp = torch.tensor(rng.binomial(1, hint_rate, size=(eff_batch, dim)),
                              dtype=torch.float32, device=device)
        H = M * H_temp

        X_in = M * X + (1 - M) * Z  # noise into missing slots

        # --- Discriminator step ---
        G_sample = G(X_in, M)
        X_hat = X_in * M + G_sample.detach() * (1 - M)
        D_prob = D(X_hat, H)
        D_loss = bce(D_prob, M)
        optD.zero_grad(); D_loss.backward(); optD.step()

        # --- Generator step ---
        G_sample = G(X_in, M)
        X_hat = X_in * M + G_sample * (1 - M)
        D_prob = D(X_hat, H)
        G_adv = -torch.mean((1 - M) * torch.log(D_prob + 1e-8))
        G_mse = torch.mean((M * X - M * G_sample) ** 2) / (torch.mean(M) + 1e-8)
        G_loss = G_adv + alpha * G_mse
        optG.zero_grad(); G_loss.backward(); optG.step()

    # --- Final imputation over the full matrix ---
    with torch.no_grad():
        Z = torch.tensor(rng.uniform(0.0, 0.01, size=(no, dim)),
                         dtype=torch.float32, device=device)
        X_in = M_all * X_all + (1 - M_all) * Z
        G_sample = G(X_in, M_all)
        imputed = M_all * X_all + (1 - M_all) * G_sample
        imputed = imputed.cpu().numpy()

    imputed = _renormalization(imputed, norm_params)
    # keep observed values exactly; fill only the originally-missing entries
    out = data_x.copy()
    miss = np.isnan(data_x)
    out[miss] = imputed[miss]
    return out
