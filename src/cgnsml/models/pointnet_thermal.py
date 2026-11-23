from __future__ import annotations


import contextlib
import copy
import time


import numpy as np
import torch
import torch.nn as nn


from . import register
# --------------------------------------------------------------------------- #
# Network - same layers as the notebook
# --------------------------------------------------------------------------- #
class PointNetThermalNet(nn.Module):
    def __init__(self, point_dim=9, global_dim=256, out_dim=1):
        super().__init__()
        # Point feature encoder
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, 64), nn.SiLU(),
            nn.Linear(64, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
        )
        # Global feature extraction (applied after the max pool)
        self.global_encoder = nn.Sequential(
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, global_dim), nn.SiLU(),
        )
        # Temperature predictor
        self.predictor = nn.Sequential(
            nn.Linear(256 + global_dim, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, out_dim),
        )
        self.initialize()


    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


    def decode(self, point_feat, global_feat):
        """point_feat (B, N, 256), global_feat (B, global_dim) -> (B, N, out)."""
        n = point_feat.shape[1]
        g = global_feat.unsqueeze(1).expand(-1, n, -1)
        return self.predictor(torch.cat([point_feat, g], dim=-1))


    def forward(self, x):
        """x: (B, N, F) with B simulations and N points each."""
        point_feat = self.point_encoder(x)                 # (B, N, 256)
        global_feat, _ = torch.max(point_feat, dim=1)      # (B, 256)
        global_feat = self.global_encoder(global_feat)     # (B, global_dim)
        return self.decode(point_feat, global_feat)        # (B, N, out)




# --------------------------------------------------------------------------- #
# cgnsml model wrapper
# --------------------------------------------------------------------------- #
@register("pointnet_thermal")
class PointNetThermal:
    """Per-simulation PointNet.

    Params (all optional; defaults reproduce the notebook):
      global_dim=256, epochs=1500, lr=2e-3, weight_decay=1e-5, loss="mse"|"huber"
      amp=True               mixed precision when running on CUDA
      points_per_case=0      0 = use every row the pipeline sampled for a case;
                             k > 0 = fresh random k rows per case every epoch
      coord_norm="none"      "none" keeps the pipeline's global scaling;
                             "per_case" centres xyz per simulation, one scale
      xyz_cols=[0, 1, 2]     coordinate columns (names like [x, y, z] also work)
      val_every=1, patience=0 (0 = no early stopping, keep best epoch)
      nan_fill=0.0, predict_chunk=200000, seed=0, device=None, log_every=10
    """


    def __init__(self, global_dim=256, epochs=1500, lr=2e-3, weight_decay=1e-5,
                 loss="mse", amp=True, points_per_case=10, coord_norm="none",
                 xyz_cols=(0, 1, 2), val_every=1, patience=0, nan_fill=0.0,
                 predict_chunk=200_000, seed=0, device=None, log_every=10):
        if loss not in ("mse", "huber"):
            raise ValueError("loss must be 'mse' or 'huber'")
        if coord_norm not in ("none", "per_case"):
            raise ValueError("coord_norm must be 'none' or 'per_case'")
        self.cfg = dict(global_dim=int(global_dim), epochs=int(epochs), lr=float(lr),
                        weight_decay=float(weight_decay), loss=loss, amp=bool(amp),
                        points_per_case=int(points_per_case or 0),
                        coord_norm=coord_norm,
                        xyz_cols=[int(c) for c in xyz_cols],
                        val_every=max(1, int(val_every)), patience=int(patience or 0),
                        nan_fill=nan_fill, predict_chunk=int(predict_chunk),
                        seed=int(seed), device=device, log_every=int(log_every))
        self.seed = int(seed)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = None
        self.in_dim = self.out_dim = None
        self.history = []


    # ---------------- helpers ---------------------------------------------- #
    def _build(self, in_dim, out_dim):
        self.in_dim, self.out_dim = in_dim, out_dim
        return PointNetThermalNet(in_dim, self.cfg["global_dim"], out_dim).to(self.device)


    def _prep(self, X):
        X = np.asarray(X, np.float32)
        nf = self.cfg["nan_fill"]
        if nf is not None:
            X = np.nan_to_num(X, nan=float(nf), posinf=float(nf), neginf=float(nf))
        return X


    @staticmethod
    def _case_index(n, groups):
        """List of row-index arrays, one per simulation."""
        if groups is None:
            return [np.arange(n)]
        g = np.asarray(groups).astype(str)
        if len(g) != n:
            raise ValueError(f"groups has {len(g)} entries for {n} rows")
        _, inv = np.unique(g, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        return np.split(order, np.cumsum(np.bincount(inv))[:-1])


    def _coord(self, x):
        if self.cfg["coord_norm"] != "per_case":
            return x
        c = self.cfg["xyz_cols"]
        xyz = x[:, c]
        xyz = xyz - xyz.mean(dim=0, keepdim=True)
        scale = xyz.pow(2).sum(dim=1).mean().sqrt().clamp_min(1e-8)
        x = x.clone()
        x[:, c] = xyz / scale
        return x


    def _to_cases(self, X, y=None, w=None, groups=None):
        cases = []
        for idx in self._case_index(len(X), groups):
            xt = self._coord(torch.from_numpy(X[idx]).to(self.device))
            yt = None if y is None else torch.from_numpy(y[idx]).to(self.device)
            wt = None if w is None else torch.from_numpy(w[idx]).to(self.device)
            cases.append((idx, xt, yt, wt))
        # print('len_cases::::', len(cases))-> this return 11 length vector ( here 11 is the number of test simulations )
        return cases


    def _loss(self, pred, y, w):
        if self.cfg["loss"] == "huber":
            err = nn.functional.smooth_l1_loss(pred, y, reduction="none")
        else:
            err = (pred - y) ** 2
        err = err.mean(dim=-1)
        if w is None:
            return err.mean()
        return (err * w).sum() / w.sum().clamp_min(1e-12)


    def _autocast(self, on):
        if on:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()


    @torch.no_grad()
    def _predict_case(self, xt):
        """Full-precision prediction for one simulation, chunked so large
        cases fit in memory. Same maths as forward()."""
        net, chunk = self.net, self.cfg["predict_chunk"]
        if len(xt) <= chunk:
            return net(xt.unsqueeze(0)).squeeze(0)
        pooled = None
        for s in range(0, len(xt), chunk):
            m = net.point_encoder(xt[s:s + chunk]).max(dim=0).values
            pooled = m if pooled is None else torch.maximum(pooled, m)
        g = net.global_encoder(pooled.unsqueeze(0))
        return torch.cat([net.decode(net.point_encoder(xt[s:s + chunk]).unsqueeze(0),
                                     g).squeeze(0)
                          for s in range(0, len(xt), chunk)])


    def _val_loss(self, cases):
        self.net.eval()
        losses = [self._loss(self._predict_case(xt), yt, None).item()
                  for _, xt, yt, _ in cases]
        return float(np.mean(losses))


    
    # ---------------- cgnsml contract -------------------------------------- #
    def fit(self, X, y, sample_weight=None, eval_set=None, groups=None,
            eval_groups=None):
        print('len X', len(X))
        torch.manual_seed(self.seed)
        X = self._prep(X)
        y = np.asarray(y, np.float32).reshape(len(X), -1)
        w = None if sample_weight is None else np.asarray(sample_weight, np.float32)
        if groups is None:
            print("  ! pointnet_thermal got no case ids - all rows are pooled as "
                  "ONE simulation. Load several files so the pipeline has a "
                  "`case` column.")
        self.net = self._build(X.shape[1], y.shape[1])
        print('groups', groups)
        print('len_groups', len(groups))
        train = self._to_cases(X, y, w, groups)
        print('train', train)


        val = None
        if eval_set is not None:
            Xe = self._prep(eval_set[0])
            ye = np.asarray(eval_set[1], np.float32).reshape(len(Xe), -1)
            val = self._to_cases(Xe, ye, None, eval_groups)


        c = self.cfg
        opt = torch.optim.AdamW(self.net.parameters(), lr=c["lr"],
                                weight_decay=c["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["epochs"])
        use_amp = c["amp"] and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        gen = torch.Generator().manual_seed(self.seed)


        print(f"[pointnet_thermal] {len(train)} training simulation(s), "
              f"{sum(len(i) for i, *_ in train):,} rows, "
              f"{0 if val is None else len(val)} validation simulation(s), "
              f"device={self.device}, amp={use_amp}")


        best, best_state, bad, last_val = float("inf"), None, 0, float("nan")
        self.history = []
        t0 = time.time()
        for epoch in range(c["epochs"]):
            self.net.train()
            total = 0.0
            # i=0
            for ci in torch.randperm(len(train), generator=gen).tolist():
                # print('ci' , ci , 'i', i)
                # i=i+1
                _, xt, yt, wt = train[ci]
                print('len xt', len(xt) )
                print('len yt', len(yt))
                k = c["points_per_case"]
                if k and k < len(xt):
                    sel = torch.randperm(len(xt), generator=gen)[:k].to(self.device)
                    xt, yt = xt[sel], yt[sel]
                    wt = None if wt is None else wt[sel]
                opt.zero_grad(set_to_none=True)
                with self._autocast(use_amp):
                    pred = self.net(xt.unsqueeze(0)).squeeze(0)
                loss = self._loss(pred.float(), yt, wt)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item()
            sched.step()
            train_loss = total / len(train)


            stop = False
            if val is not None and ((epoch + 1) % c["val_every"] == 0
                                    or epoch + 1 == c["epochs"]):
                last_val = self._val_loss(val)
                if last_val < best:
                    best, bad = last_val, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    bad += 1
                    stop = bool(c["patience"]) and bad >= c["patience"]
            self.history.append({"epoch": epoch + 1, "train": train_loss,
                                 "val": last_val})
            if c["log_every"] and ((epoch + 1) % c["log_every"] == 0 or epoch == 0
                                   or stop):
                print(f"  epoch {epoch + 1:4d}  train={train_loss:.4e}  "
                      f"val={last_val:.4e}  {time.time() - t0:.0f}s")
            if stop:
                print(f"  early stop at epoch {epoch + 1}")
                break


        if best_state is not None:
            self.net.load_state_dict(best_state)
            print(f"  kept best epoch: val={best:.4e}")
        return self


    def predict(self, X, groups=None):
        if self.net is None:
            raise RuntimeError("model is not fitted")
        X = self._prep(X)
        self.net.eval()
        out = np.empty((len(X), self.out_dim), np.float32)
        for idx, xt, _, _ in self._to_cases(X, groups=groups):
            out[idx] = self._predict_case(xt).float().cpu().numpy()
        return out


    def save(self, path):
        torch.save({"cfg": self.cfg, "dims": (self.in_dim, self.out_dim),
                    "state": self.net.state_dict(), "history": self.history}, path)


    @classmethod
    def load(cls, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(**blob["cfg"])
        obj.net = obj._build(*blob["dims"])
        obj.net.load_state_dict(blob["state"])
        obj.history = blob.get("history", [])
        return obj


