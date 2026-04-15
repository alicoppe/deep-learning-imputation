"""MLP model wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.training.base_model import BaseModel, TrainResult


class _EDDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPModel(BaseModel):
    needs_imputed = True

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        epochs: int = 60,
        patience: int = 10,
        batch_size: int = 512,
        lr: float = 1e-3,
        device: str | None = None,
        **kwargs,  # absorb unrelated params from shared sweep config
    ):
        self.hidden_dims = hidden_dims or [512, 256, 128]
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: _MLP | None = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainResult:
        loader = DataLoader(_EDDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        model = _MLP(X_train.shape[1], list(self.hidden_dims), self.dropout).to(self.device)
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5, min_lr=1e-5
        )

        best_val, best_state, no_improve = float("inf"), None, 0
        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(self.epochs):
            model.train()
            running = 0.0
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(Xb).squeeze(), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * len(Xb)

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t).squeeze(), y_val_t).item()

            epoch_train = running / len(loader.dataset)
            train_losses.append(epoch_train)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            print(
                f"  Epoch {epoch+1:3d}/{self.epochs}  train={epoch_train:.3f}  val={val_loss:.3f}"
                f"  lr={optimizer.param_groups[0]['lr']:.1e}" + (" *" if improved else "")
            )

            if no_improve >= self.patience:
                print(f"  Early stopping — no val improvement for {self.patience} consecutive epochs.")
                break

        model.load_state_dict(best_state)
        print(f"  Best val MAE: {best_val:.3f} h\n")
        self._model = model
        return TrainResult(train_losses=train_losses, val_losses=val_losses)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            return (
                self._model(torch.tensor(X, dtype=torch.float32).to(self.device))
                .squeeze()
                .cpu()
                .numpy()
            )
