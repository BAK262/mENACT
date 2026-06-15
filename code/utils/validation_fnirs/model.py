"""Attention-MIL model and training for validation_fnirs_decoding."""

from __future__ import annotations

import copy
import logging
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("validation_fnirs_decoding.model")


def resolve_device():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Attention-MIL training requires CUDA. Install GPU-enabled PyTorch and run on a CUDA-capable device."
        )
    device = torch.device("cuda")
    LOGGER.info("MIL device: %s (%s)", device, torch.cuda.get_device_name(device))
    return device


class AttentionMILModel:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        attention_dim: int,
        n_classes: int,
        dropout: float,
        device,
    ) -> None:
        import torch.nn as nn

        p = float(dropout)
        if p < 0.0 or p >= 1.0:
            raise ValueError(f"Invalid dropout={p}; expected in [0, 1).")
        do = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.net = nn.ModuleDict(
            {
                "encoder": nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    do,
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    do,
                ),
                "att_v": nn.Linear(hidden_dim, attention_dim),
                "att_u": nn.Linear(hidden_dim, attention_dim),
                "att_w": nn.Linear(attention_dim, 1),
                "head": nn.Linear(hidden_dim, n_classes),
            }
        ).to(device)
        self.device = device

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state):
        self.net.load_state_dict(state)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def __call__(self, x, mask):
        import torch

        h = self.net["encoder"](x)
        a_v = torch.tanh(self.net["att_v"](h))
        a_u = torch.sigmoid(self.net["att_u"](h))
        a = self.net["att_w"](a_v * a_u).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))
        w = torch.softmax(a, dim=1)
        w = w * mask.float()
        w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1e-8)
        bag = (w.unsqueeze(-1) * h).sum(dim=1)
        logits = self.net["head"](bag)
        return logits


def make_torch_loader(
    bag_features: Sequence[np.ndarray],
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    *,
    max_windows: int,
    sample_windows: bool,
    shuffle_windows: bool,
):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _BagDataset(Dataset):
        def __init__(self, x_list: Sequence[np.ndarray], y: np.ndarray) -> None:
            self.x_list = list(x_list)
            self.y = np.asarray(y, dtype=np.int64)

        def __len__(self) -> int:
            return len(self.x_list)

        def __getitem__(self, index: int):
            return self.x_list[index], int(self.y[index])

    def _collate(batch):
        xs_raw = [b[0] for b in batch]
        ys = [b[1] for b in batch]

        xs: List[np.ndarray] = []
        for x in xs_raw:
            if (not sample_windows) and (not shuffle_windows):
                xs.append(x)
                continue

            n = int(x.shape[0])
            if n <= 0:
                xs.append(x)
                continue

            idx = np.arange(n, dtype=np.int64)
            if sample_windows and int(max_windows) > 0 and n > int(max_windows):
                idx = np.random.choice(idx, size=int(max_windows), replace=False)
            if shuffle_windows:
                np.random.shuffle(idx)
            xs.append(x[idx, :])

        max_len = max(int(x.shape[0]) for x in xs)
        feat_dim = int(xs[0].shape[1])
        x_pad = np.zeros((len(xs), max_len, feat_dim), dtype=np.float32)
        m_pad = np.zeros((len(xs), max_len), dtype=bool)
        for i, x in enumerate(xs):
            n = int(x.shape[0])
            x_pad[i, :n, :] = x
            m_pad[i, :n] = True
        np.nan_to_num(x_pad, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            torch.from_numpy(x_pad),
            torch.from_numpy(m_pad),
            torch.from_numpy(np.asarray(ys, dtype=np.int64)),
        )

    ds = _BagDataset(bag_features, labels)
    return DataLoader(ds, batch_size=max(1, int(batch_size)), shuffle=shuffle, collate_fn=_collate)


def _eval_loss_weighted_mean(model: AttentionMILModel, loader, device, loss_fn) -> float:
    import torch

    model.eval()
    total_w = 0.0
    total = 0.0
    with torch.no_grad():
        for xb, mb, yb in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            yb = yb.to(device)
            logits = model(xb, mb)
            loss = loss_fn(logits, yb)
            n = float(yb.shape[0])
            total += float(loss.detach().cpu()) * n
            total_w += n
    if total_w <= 0:
        return float("nan")
    return float(total / total_w)


def evaluate_loader_predict(model: AttentionMILModel, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    with torch.no_grad():
        for xb, mb, yb in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            yb = yb.to(device)
            logits = model(xb, mb)
            pred = logits.argmax(dim=1)
            y_true_all.append(yb.detach().cpu().numpy())
            y_pred_all.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.zeros(0, dtype=np.int64)
    y_pred = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.zeros(0, dtype=np.int64)
    return y_true, y_pred


def fit_loso_fold(
    train_bags: Sequence[np.ndarray],
    train_labels: np.ndarray,
    held_out_bags: Sequence[np.ndarray],
    held_out_labels: np.ndarray,
    input_dim: int,
    n_classes: int,
    args,
    device,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object], bool]:
    import torch

    if np.unique(train_labels).size < 2:
        return np.zeros(0, dtype=np.int64), {}, False

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader = make_torch_loader(
        train_bags,
        train_labels,
        batch_size=args.batch_size,
        shuffle=True,
        max_windows=int(args.bag_max_windows),
        sample_windows=bool(int(args.bag_max_windows) > 0),
        shuffle_windows=bool(args.bag_shuffle_windows),
    )
    held_loader = make_torch_loader(
        held_out_bags,
        held_out_labels,
        batch_size=args.batch_size,
        shuffle=False,
        max_windows=0,
        sample_windows=False,
        shuffle_windows=False,
    )

    counts = np.bincount(train_labels.astype(np.int64), minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    class_w = torch.tensor(inv_freq / inv_freq.sum() * n_classes, dtype=torch.float32, device=device)

    model = AttentionMILModel(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim),
        attention_dim=int(args.attention_dim),
        n_classes=n_classes,
        dropout=float(args.dropout),
        device=device,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stopped_epoch = 0
    bad_epochs = 0
    val_loss_history: List[float] = []
    t_train = time.perf_counter()
    for epoch_i in range(int(args.max_epochs)):
        model.train()
        for xb, mb, yb in train_loader:
            xb = xb.to(device)
            mb = mb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb, mb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        val_loss = _eval_loss_weighted_mean(model, held_loader, device=device, loss_fn=loss_fn)
        val_loss_history.append(float(val_loss))
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = int(epoch_i + 1)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                stopped_epoch = int(epoch_i + 1)
                break

    if stopped_epoch <= 0:
        stopped_epoch = int(args.max_epochs)

    train_s = time.perf_counter() - t_train
    model.load_state_dict(best_state)
    _y_true_h, y_pred_h = evaluate_loader_predict(model, held_loader, device=device)
    metrics: Dict[str, object] = {
        "es_val_loss": float(best_val),
        "best_epoch": float(best_epoch) if best_epoch > 0 else float("nan"),
        "stopped_epoch": float(stopped_epoch) if stopped_epoch > 0 else float("nan"),
        "train_s": float(train_s),
        "val_loss_history": val_loss_history,
    }
    return y_pred_h, metrics, True
