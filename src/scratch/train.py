"""Training and evaluation of one from-scratch run.

Loss = lambda_sup * [SupCon(f, type) + 0.5 * SupCon(f, type+colour)] + lambda_itc * InfoNCE(z_img, z_txt)
    SupCon  supervised contrastive loss (Khosla et al., 2020) on the retrieval embedding f
    InfoNCE symmetric image-text contrastive loss (as in CLIP, Radford et al., 2021), aligns the
            modalities so text-only / image-only queries work

Test metrics (on products never seen in training), NDCG@10:
    i2i         "more like this": relevant = same article type
    i2i_strict  relevant = same article type AND colour
    text2item   shopper text query ("navy blue shirts for men") -> products; relevant = colour+type+gender
    image2item  photo-only query -> products (catalog items use image + title); relevant = same type
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..evaluate import build_text_queries, ranking_metrics, topk_from_scores
from .data import ScratchData
from .models import MultimodalRec, count_params

OPTIM_DEFAULTS = {  # weight decay per optimizer (Adam: none; AdamW: decoupled)
    "SGD": {"momentum": 0.9, "nesterov": True, "weight_decay": 5e-4},
    "Adam": {"weight_decay": 0.0},
    "AdamW": {"weight_decay": 0.05},
}


@dataclass
class RunConfig:
    method: str = "gated"
    optimizer: str = "AdamW"
    lr: float = 1e-3
    seed: int = 0
    epochs: int = 30
    batch: int = 256
    lambda_sup: float = 1.0
    lambda_itc: float = 1.0
    modality_dropout: float = 0.3
    temperature: float = 0.1
    warmup_epochs: int = 1
    name: str = ""

    def run_name(self):
        return self.name or f"{self.method}_{self.optimizer}_lr{self.lr:g}_s{self.seed}"


# ------------------------------------------------------------------ losses
def supcon(z, y, t):
    sim = z @ z.T / t
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e4)
    pos = (y[:, None] == y[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    has = pos.any(1)
    if not has.any():
        return z.sum() * 0
    return (-(logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()


def info_nce(zi, zt, logit_scale):
    logits = logit_scale.exp().clamp(max=100) * zi @ zt.T
    target = torch.arange(len(zi), device=zi.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


# ------------------------------------------------------------------ batches
MEAN, STD = 0.5, 0.5


def to_input(images_u8: torch.Tensor, train: bool) -> torch.Tensor:
    """uint8 (B,H,W,3) -> normalised float (B,3,H,W), with flip + shift augmentation in training."""
    x = images_u8.permute(0, 3, 1, 2).float().div_(255).sub_(MEAN).div_(STD)
    if train:
        flip = torch.rand(len(x), device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        pad = 4
        xp = F.pad(x, [pad] * 4, value=(1 - MEAN) / STD)   # pad with white, the photo background
        i, j = np.random.randint(0, 2 * pad + 1, 2)
        x = xp[:, :, i:i + x.shape[2], j:j + x.shape[3]]
    return x


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def embed(model, data_t, idx, images=True, texts=True, bs=1024):
    model.eval()
    out = []
    for b in range(0, len(idx), bs):
        j = idx[b:b + bs]
        x = to_input(data_t["images"][j], False) if images else None
        t = data_t["tokens"][j] if texts else None
        out.append(model.encode(x, t)["f"].float())
    return torch.cat(out)


@torch.no_grad()
def evaluate(model, data: ScratchData, data_t, split="test", full=True) -> dict:
    """Metrics on one split; items of that split form the catalog. Returns means + per-query NDCG."""
    idx = data.split[split]
    F_items = embed(model, data_t, idx)
    y_t, y_f = data.y_type[idx], data.y_fine[idx]
    res, perq = {}, {}

    def ndcg_from(scores, labels_q, labels_items, n_rel, exclude_self):
        S = scores.cpu().numpy()
        top = topk_from_scores(S, exclude_self=exclude_self)
        rel = (labels_items[top] == labels_q[:, None]).astype(float)
        m = ranking_metrics(rel, n_rel)
        return m["NDCG@10"], m["_ndcg"]

    S = F_items @ F_items.T
    n_type = pd.Series(y_t).map(pd.Series(y_t).value_counts()).values - 1
    n_fine = pd.Series(y_f).map(pd.Series(y_f).value_counts()).values - 1
    res["i2i"], perq["i2i"] = ndcg_from(S, y_t, y_t, n_type, True)
    if not full:
        return res
    res["i2i_strict"], perq["i2i_strict"] = ndcg_from(S, y_f, y_f, n_fine, True)

    if model.use_txt:  # text query -> products
        df = data.meta.iloc[idx].reset_index(drop=True)
        queries, item_keys = build_text_queries(df, min_items=2)
        qtok = torch.tensor(data.tokens_for([q for q, _ in queries]), device=F_items.device).long()
        Q = model.encode(None, qtok)["f"].float() if len(queries) else None
        if Q is not None:
            qkeys = np.array([k for _, k in queries])
            n_rel = pd.Series(item_keys).value_counts().reindex(qkeys).values
            res["text2item"], perq["text2item"] = ndcg_from(Q @ F_items.T, qkeys, item_keys, n_rel, False)
    if model.use_img:  # image-only query -> products (catalog items use both modalities)
        Qi = embed(model, data_t, idx, texts=False)
        res["image2item"], perq["image2item"] = ndcg_from(Qi @ F_items.T, y_t, y_t, n_type, True)
    res["_per_query"] = perq
    return res


# ------------------------------------------------------------------ training
def make_optimizer(name, params, lr):
    kw = OPTIM_DEFAULTS[name]
    return {"SGD": torch.optim.SGD, "Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[name](params, lr=lr, **kw)


def train_run(cfg: RunConfig, data: ScratchData, device="cpu", data_t=None, verbose=True, test=True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if data_t is None:
        data_t = {"images": torch.tensor(data.images, device=device),
                  "tokens": torch.tensor(data.tokens, device=device).long()}
    y_type = torch.tensor(data.y_type, device=device)
    y_fine = torch.tensor(data.y_fine, device=device)
    tr = data.split["train"]

    model = MultimodalRec(cfg.method, len(data.vocab), data.tokens.shape[1]).to(device)
    opt = make_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
    steps_per_epoch = len(tr) // cfg.batch
    total, warm = cfg.epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best, best_state, t0 = [], -1.0, None, time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.from_numpy(np.random.permutation(tr)).to(device)
        losses = []
        for b in range(steps_per_epoch):
            j = perm[b * cfg.batch:(b + 1) * cfg.batch]
            x = to_input(data_t["images"][j], True) if model.use_img else None
            t = data_t["tokens"][j] if model.use_txt else None
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model.encode(x, t, drop_p=cfg.modality_dropout)
            f = out["f"].float()
            loss = cfg.lambda_sup * (supcon(f, y_type[j], cfg.temperature)
                                     + 0.5 * supcon(f, y_fine[j], cfg.temperature))
            if cfg.lambda_itc > 0 and out["z_img"] is not None and out["z_txt"] is not None:
                loss = loss + cfg.lambda_itc * info_nce(out["z_img"].float(), out["z_txt"].float(), model.logit_scale)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        val = evaluate(model, data, data_t, "val", full=False)["i2i"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_i2i": float(val),
                        "lr": opt.param_groups[0]["lr"], "time_s": time.perf_counter() - t0})
        if val > best:  # early stopping: keep the best epoch on validation
            best, best_state = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  [{cfg.run_name()}] epoch {epoch:3d}/{cfg.epochs} loss={history[-1]['train_loss']:.3f} "
                  f"val_i2i={val:.4f} ({history[-1]['time_s']:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    result = {"config": asdict(cfg), "run": cfg.run_name(), "params": count_params(model),
              "train_time_s": time.perf_counter() - t0, "best_val_i2i": best,
              "best_epoch": int(np.argmax([h["val_i2i"] for h in history]) + 1), "history": history}
    if test:
        result["test"] = evaluate(model, data, data_t, "test")
        val_full = evaluate(model, data, data_t, "val")
        result["val"] = {k: v for k, v in val_full.items() if not k.startswith("_")}
    return model, result
