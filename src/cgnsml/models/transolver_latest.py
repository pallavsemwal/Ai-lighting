"""Transolver (Wu et al., ICML 2024) with per-case physics attention."""
from __future__ import annotations

import contextlib
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import register


def no_autocast(t):
    return torch.autocast(device_type=t.device.type, enabled=False)


class PhysicsAttention(nn.Module):
    def __init__(self, dim, heads, slices, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "hidden must be divisible by heads"
        self.heads, self.dh = heads, dim // heads
        self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        self.proj_x = nn.Linear(dim, dim)
        self.proj_fx = nn.Linear(dim, dim)
        self.proj_slice = nn.Linear(self.dh, slices)
        self.qkv = nn.Linear(self.dh, 3 * self.dh, bias=False)
        self.drop = dropout
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def split(self, t):
        b, n, _ = t.shape
        return t.view(b, n, self.heads, self.dh).transpose(1, 2)

    def weights(self, z):
        with no_autocast(z):
            logits = self.proj_slice(self.split(self.proj_x(z.float())))
            return torch.softmax(logits / self.temperature.clamp_min(1e-2), dim=-1)

    def gather(self, z, w):
        with no_autocast(z):
            fx = self.split(self.proj_fx(z.float()))
            return torch.einsum("bhnd,bhnm->bhmd", fx, w), w.sum(2)

    def attend(self, num, den):
        tok = num / (den.unsqueeze(-1) + 1e-5)
        q, k, v = self.qkv(tok).chunk(3, dim=-1)
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop if self.training else 0.0)

    def scatter(self, tok, w):
        with no_autocast(w):
            out = torch.einsum("bhmd,bhnm->bhnd", tok.float(), w)
        b, h, n, d = out.shape
        return self.out(out.transpose(1, 2).reshape(b, n, h * d))

    def forward(self, z):
        w = self.weights(z)
        return self.scatter(self.attend(*self.gather(z, w)), w)


class Block(nn.Module):
    def __init__(self, dim, heads, slices, mlp_ratio, dropout):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = PhysicsAttention(dim, heads, slices, dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, h):
        h = h + self.attn(self.ln1(h))
        return h + self.mlp(self.ln2(h))


class TransolverNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, layers=8, heads=8, slices=32,
                 mlp_ratio=1.0, dropout=0.0, grad_ckpt=False):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.pre = nn.Sequential(nn.Linear(in_dim, 2 * hidden), nn.GELU(),
                                 nn.Linear(2 * hidden, hidden))
        self.placeholder = nn.Parameter(torch.rand(hidden) / hidden)
        self.blocks = nn.ModuleList(Block(hidden, heads, slices, mlp_ratio, dropout)
                                    for _ in range(layers))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for b in self.blocks:
            nn.init.orthogonal_(b.attn.proj_slice.weight)

    def forward(self, x):
        h = self.pre(x) + self.placeholder
        for b in self.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                h = checkpoint(b, h, use_reentrant=False)
            else:
                h = b(h)
        return self.head(h)

    @torch.no_grad()
    def forward_chunked(self, x, chunk):
        # Two passes per layer: accumulate slice tokens over all chunks, then update points.
        spans = [slice(s, s + chunk) for s in range(0, len(x), chunk)]
        h = torch.cat([(self.pre(x[s]) + self.placeholder).float() for s in spans])
        for blk in self.blocks:
            att = blk.attn
            num = den = 0
            for s in spans:
                z = blk.ln1(h[s][None])
                n_, d_ = att.gather(z, att.weights(z))
                num, den = num + n_, den + d_
            tok = att.attend(num, den)
            for s in spans:
                hs = h[s][None]
                z = blk.ln1(hs)
                hs = hs + att.scatter(tok, att.weights(z))
                h[s] = (hs + blk.mlp(blk.ln2(hs)))[0].float()
        return torch.cat([self.head(h[s]) for s in spans])


@register("transolver_lt")
class Transolver:
    def __init__(self, hidden=256, layers=8, heads=8, slices=32, mlp_ratio=1.0,
                 dropout=0.0, epochs=500, lr=1e-3, weight_decay=1e-5,
                 sched="onecycle", clip=1.0, loss="mse", amp=True,
                 points_per_case=0, coord_norm="none", xyz_cols=(0, 1, 2),
                 val_every=1, patience=0, predict_chunk=100_000, seed=0,
                 device=None, log_every=10, grad_ckpt=False, cover_all=True):
        if sched not in ("onecycle", "cosine"):
            raise ValueError(f"unknown sched: {sched}")
        if loss not in ("mse", "huber"):
            raise ValueError(f"unknown loss: {loss}")
        if coord_norm not in ("none", "per_case"):
            raise ValueError(f"unknown coord_norm: {coord_norm}")

        self.params = dict(
            hidden=hidden, layers=layers, heads=heads, slices=slices,
            mlp_ratio=mlp_ratio, dropout=dropout, epochs=epochs, lr=lr,
            weight_decay=weight_decay, sched=sched, clip=clip, loss=loss, amp=amp,
            points_per_case=points_per_case, coord_norm=coord_norm,
            xyz_cols=list(xyz_cols), val_every=val_every, patience=patience,
            predict_chunk=predict_chunk, seed=seed, device=device,
            log_every=log_every, grad_ckpt=grad_ckpt, cover_all=cover_all,
        )
        self.grad_ckpt, self.cover_all = bool(grad_ckpt), bool(cover_all)
        self.epochs, self.lr, self.wd = int(epochs), float(lr), float(weight_decay)
        self.sched, self.clip, self.loss_type = sched, float(clip or 0), loss
        self.amp = bool(amp)
        self.points_per_case = int(points_per_case or 0)
        self.coord_norm, self.xyz_cols = coord_norm, list(xyz_cols)
        self.val_every, self.patience = max(1, int(val_every)), int(patience or 0)
        self.predict_chunk, self.seed, self.log_every = int(predict_chunk), int(seed), int(log_every)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.net = None
        self.dims = None
        self.history = []

    def _build(self, in_dim, out_dim):
        p = self.params
        self.dims = (in_dim, out_dim)
        return TransolverNet(in_dim, out_dim, int(p["hidden"]), int(p["layers"]),
                             int(p["heads"]), int(p["slices"]), float(p["mlp_ratio"]),
                             float(p["dropout"]),
                             bool(p.get("grad_ckpt", False))).to(self.device)

    @staticmethod
    def _case_rows(n, groups):
        if groups is None:
            return [np.arange(n)]
        _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        order = np.argsort(inv, kind="stable")
        return np.split(order, np.cumsum(np.bincount(inv))[:-1])

    def _norm_coords(self, x):
        if self.coord_norm == "none":
            return x
        c = self.xyz_cols
        xyz = x[:, c] - x[:, c].mean(0, keepdim=True)
        x = x.clone()
        x[:, c] = xyz / xyz.pow(2).sum(1).mean().sqrt().clamp_min(1e-8)
        return x

    def _cases(self, X, y=None, w=None, groups=None):
        X = np.nan_to_num(np.asarray(X, np.float32))
        if y is not None:
            y = np.asarray(y, np.float32).reshape(len(X), -1)
        if w is not None:
            w = np.asarray(w, np.float32)

        def t(a, idx):
            return None if a is None else torch.from_numpy(a[idx]).to(self.device)

        return [(idx, self._norm_coords(t(X, idx)), t(y, idx), t(w, idx))
                for idx in self._case_rows(len(X), groups)]

    def _loss(self, pred, y, w=None):
        if self.loss_type == "huber":
            err = F.smooth_l1_loss(pred, y, reduction="none").mean(-1)
        else:
            err = (pred - y).pow(2).mean(-1)
        if w is None:
            return err.mean()
        return (err * w).sum() / w.sum().clamp_min(1e-12)

    def _parts(self, n, gen):
        """Point subsets used as training steps for one case.

        points_per_case=0 or >= n: the whole case in one step.
        cover_all=True: a random partition into disjoint chunks of ~k points, so
        every point is seen once per epoch. cover_all=False: one random k-subset.
        """
        k = self.points_per_case
        if not 0 < k < n:
            return [None]
        if not self.cover_all:
            return [None if gen is None else torch.randperm(n, generator=gen)[:k]]
        m = max(1, round(n / k))            # equal-sized chunks, no tiny leftover
        if gen is None:
            return [None] * m
        return list(torch.randperm(n, generator=gen).chunk(m))

    @torch.no_grad()
    def _infer(self, x):
        if len(x) <= self.predict_chunk:
            return self.net(x[None])[0]
        return self.net.forward_chunked(x, self.predict_chunk)

    def _val_loss(self, cases):
        self.net.eval()
        return float(np.mean([self._loss(self._infer(x).float(), y).item()
                              for _, x, y, _ in cases]))

    def fit(self, X, y, sample_weight=None, eval_set=None, groups=None, eval_groups=None):
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)

        if groups is None:
            print("[transolver] warning: no case ids, treating all rows as one simulation")
        train = self._cases(X, y, sample_weight, groups)
        val = self._cases(*eval_set[:2], groups=eval_groups) if eval_set is not None else None
        self.net = self._build(train[0][1].shape[1], train[0][2].shape[1])
        steps_per_epoch = sum(len(self._parts(len(c[1]), None)) for c in train)

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        if self.sched == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.lr, total_steps=self.epochs * steps_per_epoch)
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        use_amp = self.amp and self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
        autocast = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if use_amp else contextlib.nullcontext

        n_params = sum(p.numel() for p in self.net.parameters()) / 1e6
        print(f"[transolver] train={len(train)} cases ({sum(len(c[0]) for c in train):,} rows), "
              f"val={len(val) if val else 0} cases, {n_params:.2f}M params, "
              f"device={self.device}, amp={amp_dtype if use_amp else 'off'}")

        best, best_state, bad, val_loss = float("inf"), None, 0, float("nan")
        self.history = []
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            running = 0.0
            n_steps = 0
            for ci in torch.randperm(len(train), generator=gen).tolist():
                _, xc, yc, wc = train[ci]
                for sel in self._parts(len(xc), gen):
                    if sel is None:
                        xb, yb, wb = xc, yc, wc
                    else:
                        sel = sel.to(self.device)
                        xb, yb = xc[sel], yc[sel]
                        wb = None if wc is None else wc[sel]

                    opt.zero_grad(set_to_none=True)
                    with autocast():
                        pred = self.net(xb[None])[0]
                    loss = self._loss(pred.float(), yb, wb)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss at epoch {epoch}")

                    scaler.scale(loss).backward()
                    if self.clip:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                    scaler.step(opt)
                    scaler.update()
                    if self.sched == "onecycle":
                        sched.step()
                    running += loss.item()
                    n_steps += 1

            if self.sched == "cosine":
                sched.step()
            train_loss = running / max(n_steps, 1)

            stop = False
            if val and (epoch % self.val_every == 0 or epoch == self.epochs):
                val_loss = self._val_loss(val)
                if val_loss < best:
                    best, bad = val_loss, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    bad += 1
                    stop = self.patience > 0 and bad >= self.patience

            lr_now = opt.param_groups[0]["lr"]
            self.history.append({"epoch": epoch, "train": train_loss, "val": val_loss, "lr": lr_now})
            if self.log_every and (epoch == 1 or epoch % self.log_every == 0 or stop):
                print(f"  epoch {epoch:4d}  train={train_loss:.4e}  val={val_loss:.4e}  "
                      f"lr={lr_now:.2e}  {time.time() - t0:.0f}s")
            if stop:
                print(f"  early stop at epoch {epoch}")
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
            print(f"  restored best val={best:.4e}")
        return self

    def predict(self, X, groups=None):
        if self.net is None:
            raise RuntimeError("model is not fitted")
        self.net.eval()
        out = np.empty((len(X), self.dims[1]), np.float32)
        for idx, x, _, _ in self._cases(X, groups=groups):
            out[idx] = self._infer(x).float().cpu().numpy()
        return out

    def save(self, path):
        torch.save({"params": self.params, "dims": self.dims,
                    "state": self.net.state_dict(), "history": self.history}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(**ckpt["params"])
        obj.net = obj._build(*ckpt["dims"])
        obj.net.load_state_dict(ckpt["state"])
        obj.history = ckpt.get("history", [])
        return obj


