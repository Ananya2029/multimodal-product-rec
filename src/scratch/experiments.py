"""Full from-scratch study (run on a GPU, e.g. Google Colab). Resumable: finished runs are skipped.

Stages
  1 tune        learning-rate search for each optimizer (SGD / Adam / AdamW) on the VALIDATION split,
                using the proposed gated-fusion model
  2 optimizers  proposed model x 3 optimizers (tuned lr) x seeds          -> optimizer comparison
  3 methods     6 methods x best optimizer (chosen on validation) x seeds  -> method comparison
  4 ablations   proposed model without ITC loss / SupCon / consistency loss / modality dropout
  5 report      tables (mean +- std over seeds), paired bootstrap significance tests, figures,
                and the export used by the Streamlit app

Extension (modality balancing, run after the main study):
  balance         gated model + OGM gradient modulation (alpha tuned on validation) / uni-modal supervision /
                  noise-aware training / noise-aware + uni-modal, x seeds, with the main study's optimizer
  report_balance  table + significance vs the plain gated model (works locally from results/scratch/runs)

    python -m src.scratch.experiments                   # everything (defaults below)
    python -m src.scratch.experiments --smoke           # tiny end-to-end check (CPU, minutes)
    python -m src.scratch.experiments --out /content/drive/MyDrive/mmrec   # save to Google Drive
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import ROOT
from ..metrics import bootstrap_ci, paired_bootstrap_p
from .data import SCRATCH_DIR, ScratchData
from .models import METHOD_NAMES, METHODS
from .train import RunConfig, embed, train_run

PROPOSED = "gated"
OPTIMIZERS = ["SGD", "Adam", "AdamW"]
LR_GRID = {"SGD": [0.03, 0.1, 0.3], "Adam": [3e-4, 1e-3, 3e-3], "AdamW": [3e-4, 1e-3, 3e-3]}
METRICS = {"i2i": "More like this (type)", "i2i_strict": "More like this (type + colour)",
           "text2item": "Text query -> product", "image2item": "Photo query -> product"}
ABLATIONS = {"no_itc": {"lambda_itc": 0.0}, "no_supcon": {"lambda_sup": 0.0},
             "no_consistency": {"lambda_cons": 0.0}, "no_modality_dropout": {"modality_dropout": 0.0}}
ABLATION_NAMES = {"full": "Gated fusion (full model)", "no_itc": "- image-text contrastive loss",
                  "no_supcon": "- supervised contrastive loss", "no_consistency": "- missing-modality consistency",
                  "no_modality_dropout": "- modality dropout"}


class Study:
    def __init__(self, out: Path, epochs: int, tune_epochs: int, seeds: list[int], device: str, smoke=False,
                 batch: int = 256):
        self.out, self.epochs, self.tune_epochs, self.seeds, self.device = out, epochs, tune_epochs, seeds, device
        self.batch = batch
        self.runs_dir = out / "results" / "scratch" / "runs"
        self.res_dir = out / "results" / "scratch"
        self.ckpt_dir = out / "models" / "scratch"
        for d in (self.runs_dir, self.ckpt_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.data = ScratchData(SCRATCH_DIR)
        self.data_t = {"images": torch.tensor(self.data.images, device=device),
                       "tokens": torch.tensor(self.data.tokens, device=device).long()}
        self.smoke = smoke
        print(f"device={device} | {len(self.data.meta):,} products | train {len(self.data.split['train']):,} "
              f"val {len(self.data.split['val']):,} test {len(self.data.split['test']):,}", flush=True)

    # ---- one run (skipped if already done)
    def run(self, cfg: RunConfig, save_ckpt=False, test=True) -> dict:
        cfg = replace(cfg, batch=self.batch)
        name = cfg.run_name()
        f = self.runs_dir / f"{name}.json"
        if f.exists():
            return json.loads(f.read_text())
        model, res = train_run(cfg, self.data, self.device, self.data_t, test=test)
        if test:
            perq = res["test"].pop("_per_query")
            np.savez_compressed(self.runs_dir / f"{name}_perq.npz", **perq)
        f.write_text(json.dumps(res, default=float))
        if save_ckpt:
            torch.save(model.state_dict(), self.ckpt_dir / f"{name}.pt")
        del model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return res

    # ---- stages
    def tune(self) -> dict:
        best = {}
        for opt in OPTIMIZERS:
            scores = {}
            for lr in LR_GRID[opt]:
                cfg = RunConfig(PROPOSED, opt, lr, seed=0, epochs=self.tune_epochs, name=f"tune_{opt}_lr{lr:g}")
                scores[lr] = self.run(cfg, test=False)["best_val_i2i"]
            best[opt] = max(scores, key=scores.get)
            print(f"tune {opt}: {scores} -> lr={best[opt]:g}", flush=True)
        (self.res_dir / "tuned_lr.json").write_text(json.dumps(best))
        return best

    def optimizers(self, lrs) -> str:
        val = {}
        for opt in OPTIMIZERS:
            rs = [self.run(RunConfig(PROPOSED, opt, lrs[opt], seed=s, epochs=self.epochs), save_ckpt=s == self.seeds[0])
                  for s in self.seeds]
            val[opt] = np.mean([r["val"]["i2i"] + r["val"]["i2i_strict"] for r in rs])
        best = max(val, key=val.get)  # chosen on VALIDATION, never on test
        print(f"optimizer selection (val): {val} -> {best}", flush=True)
        (self.res_dir / "best_optimizer.json").write_text(json.dumps({"optimizer": best, "lr": lrs[best]}))
        return best

    def methods(self, opt, lr):
        for m in METHODS:
            for s in self.seeds:
                self.run(RunConfig(m, opt, lr, seed=s, epochs=self.epochs), save_ckpt=s == self.seeds[0])

    def ablations(self, opt, lr):
        for key, change in ABLATIONS.items():
            for s in self.seeds:
                base = RunConfig(PROPOSED, opt, lr, seed=s, epochs=self.epochs)
                self.run(replace(base, name=f"abl_{key}_{base.run_name()}", **change))

    # ---- report
    def _rows(self):
        rows = []
        for f in sorted(self.runs_dir.glob("*.json")):
            r = json.loads(f.read_text())
            if "test" not in r:
                continue
            c = r["config"]
            rows.append({"run": r["run"], **{k: c[k] for k in ("method", "optimizer", "lr", "seed")},
                         "ablation": next((k for k in ABLATIONS if r["run"].startswith(f"abl_{k}_")), "full"),
                         **{f"test_{k}": r["test"].get(k, np.nan) for k in METRICS},
                         **{f"val_{k}": r["val"].get(k, np.nan) for k in METRICS},
                         "params": r["params"], "train_min": r["train_time_s"] / 60, "best_epoch": r["best_epoch"]})
        return pd.DataFrame(rows)

    def _perq(self, runs, metric):
        arrs = []
        for run in runs:
            z = np.load(self.runs_dir / f"{run}_perq.npz")
            if metric in z:
                arrs.append(z[metric])
        return np.mean(arrs, axis=0) if arrs else None   # average over seeds, per query

    def report(self, lrs, best_opt):
        df = self._rows()
        df.round(5).to_csv(self.res_dir / "all_runs.csv", index=False)
        main = df[(df["ablation"] == "full")]
        agg = lambda g: pd.Series({**{f"{k} mean": g[f"test_{k}"].mean() for k in METRICS},
                                   **{f"{k} std": g[f"test_{k}"].std() for k in METRICS},
                                   "seeds": len(g), "params (M)": g["params"].iloc[0] / 1e6,
                                   "train min": g["train_min"].mean()})
        # optimizer comparison (proposed model)
        opt_tab = main[(main["method"] == PROPOSED)].groupby("optimizer").apply(agg, include_groups=False)
        opt_tab.insert(0, "lr", [lrs[o] for o in opt_tab.index])
        opt_tab.round(4).to_csv(self.res_dir / "optimizer_comparison.csv")
        # method comparison (best optimizer)
        mt = main[(main["optimizer"] == best_opt) & (main["lr"] == lrs[best_opt])]
        meth_tab = mt.groupby("method").apply(agg, include_groups=False).reindex([m for m in METHODS if m in set(mt["method"])])
        meth_tab.insert(0, "name", [METHOD_NAMES[m] for m in meth_tab.index])
        meth_tab.round(4).to_csv(self.res_dir / "method_comparison.csv")
        # ablations
        ab = df[(df["method"] == PROPOSED) & (df["optimizer"] == best_opt) & (df["lr"] == lrs[best_opt])]
        abl_tab = ab.groupby("ablation").apply(agg, include_groups=False)
        abl_tab.insert(0, "name", [ABLATION_NAMES[a] for a in abl_tab.index])
        abl_tab.round(4).to_csv(self.res_dir / "ablations.csv")

        # significance: proposed vs every other method, per metric (per-query NDCG averaged over seeds)
        runs_of = {m: mt[mt["method"] == m]["run"].tolist() for m in meth_tab.index}
        sig = []
        for k in METRICS:
            pq = {m: self._perq(r, k) for m, r in runs_of.items()}
            pq = {m: v for m, v in pq.items() if v is not None}
            if not pq:
                continue
            ranked = sorted(pq, key=lambda m: pq[m].mean(), reverse=True)
            for other in ranked:
                for prop in ("gated", "xattn"):
                    if prop in pq and other != prop:
                        lo, hi = bootstrap_ci(pq[prop])
                        sig.append({"metric": k, "proposed": prop, "vs": other,
                                    "proposed_ndcg": pq[prop].mean(), "proposed_ci_low": lo, "proposed_ci_high": hi,
                                    "other_ndcg": pq[other].mean(), "diff": pq[prop].mean() - pq[other].mean(),
                                    "p_value": paired_bootstrap_p(pq[prop], pq[other]),
                                    "n_queries": len(pq[prop])})
        sig = pd.DataFrame(sig)
        sig.round(5).to_csv(self.res_dir / "significance.csv", index=False)

        summary = {"tuned_lr": lrs, "best_optimizer": best_opt, "seeds": self.seeds, "epochs": self.epochs,
                   "n_products": int(len(self.data.meta)), "n_test": int(len(self.data.split["test"])),
                   "winners": {k: {"method": meth_tab[f"{k} mean"].idxmax(),
                                   "ndcg": float(meth_tab[f"{k} mean"].max())}
                               for k in METRICS if meth_tab[f"{k} mean"].notna().any()}}
        (self.res_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        self.figures(df, opt_tab, meth_tab, abl_tab, lrs, best_opt)
        print("\n=== method comparison (test NDCG@10, mean over seeds) ===")
        print(meth_tab[["name"] + [f"{k} mean" for k in METRICS]].round(4).to_string())
        print("\n=== optimizer comparison (proposed model) ===")
        print(opt_tab[["lr"] + [f"{k} mean" for k in METRICS]].round(4).to_string())
        print("\n=== ablations ===")
        print(abl_tab[["name"] + [f"{k} mean" for k in METRICS]].round(4).to_string())
        return summary

    def figures(self, df, opt_tab, meth_tab, abl_tab, lrs, best_opt):
        colors = {"SGD": "#4C72B0", "Adam": "#DD8452", "AdamW": "#55A868"}
        # training curves of the proposed model per optimizer (first seed)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for opt in OPTIMIZERS:
            f = self.runs_dir / f"{RunConfig(PROPOSED, opt, lrs[opt], seed=self.seeds[0]).run_name()}.json"
            if not f.exists():
                continue
            h = pd.DataFrame(json.loads(f.read_text())["history"])
            axes[0].plot(h["epoch"], h["train_loss"], label=opt, color=colors[opt], lw=2)
            axes[1].plot(h["epoch"], h["val_i2i"], label=opt, color=colors[opt], lw=2)
        axes[0].set_title("Training loss (gated fusion)")
        axes[1].set_title("Validation NDCG@10 (more like this)")
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(self.res_dir / "optimizer_curves.png", dpi=140)
        plt.close(fig)

        def bars(tab, labels, path, title):
            ks = [k for k in METRICS if tab[f"{k} mean"].notna().any()]
            fig, axes = plt.subplots(1, len(ks), figsize=(4.6 * len(ks), 0.45 * len(tab) + 1.8), sharey=True)
            for ax, k in zip(np.atleast_1d(axes), ks):
                m, s = tab[f"{k} mean"], tab[f"{k} std"].fillna(0)
                y = np.arange(len(tab))[::-1]
                ax.barh(y, m.fillna(0), xerr=s, color=["#2E8B57" if "proposed" in str(l) or l.startswith("Gated")
                                                        else "#8DA0CB" for l in labels], capsize=3)
                for yy, v in zip(y, m):
                    ax.text((0 if pd.isna(v) else v) + 0.01, yy, "n/a" if pd.isna(v) else f"{v:.3f}", va="center",
                            fontsize=8)
                ax.set_yticks(y, labels)
                ax.set_xlim(0, 1.1)
                ax.set_title(METRICS[k], fontsize=10)
                ax.set_xlabel("NDCG@10 (mean ± std over seeds)")
            fig.suptitle(title, fontweight="bold")
            fig.tight_layout()
            fig.savefig(path, dpi=140)
            plt.close(fig)

        bars(meth_tab, list(meth_tab["name"]), self.res_dir / "method_comparison.png",
             f"From-scratch methods (optimizer: {best_opt})")
        bars(abl_tab, list(abl_tab["name"]), self.res_dir / "ablations.png", "Ablation of the gated fusion model")
        bars(opt_tab.assign(), [f"{o} (lr {lrs[o]:g})" for o in opt_tab.index], self.res_dir / "optimizer_comparison.png",
             "Optimizer comparison (gated fusion)")

    # ---- export for the app: first-seed model of every method + test-catalog embeddings
    def export(self, lrs, best_opt):
        from .models import MultimodalRec
        exp = {"optimizer": best_opt, "lr": lrs[best_opt], "methods": {}}
        emb_dir = self.out / "embeddings" / "scratch"
        emb_dir.mkdir(parents=True, exist_ok=True)
        te = self.data.split["test"]
        for m in METHODS:
            run = RunConfig(m, best_opt, lrs[best_opt], seed=self.seeds[0]).run_name()
            ck = self.ckpt_dir / f"{run}.pt"
            if not ck.exists():
                continue
            model = MultimodalRec(m, len(self.data.vocab), self.data.tokens.shape[1]).to(self.device)
            model.load_state_dict(torch.load(ck, map_location=self.device))
            np.save(emb_dir / f"{m}.npy", embed(model, self.data_t, te).cpu().numpy().astype(np.float32))
            exp["methods"][m] = {"run": run, "name": METHOD_NAMES[m], "checkpoint": f"{run}.pt"}
        self.data.meta.iloc[te].to_csv(self.out / "embeddings" / "scratch" / "test_meta.csv", index=False)
        (self.ckpt_dir / "export.json").write_text(json.dumps(exp, indent=2))
        print(f"exported {len(exp['methods'])} models for the app")


BALANCE_NAMES = {"base": "Gated fusion (no balancing)", "ogm": "+ gradient modulation (OGM)",
                 "uni": "+ uni-modal supervision", "noise": "+ noise-aware training",
                 "noise_uni": "+ noise-aware training + uni-modal supervision"}
OGM_ALPHAS = [0.3, 1.0]


def split_fingerprint(meta: pd.DataFrame) -> str:
    """Hash of the test product ids: the extension must use exactly the main study's split."""
    import hashlib
    ids = ",".join(map(str, sorted(meta.loc[meta["split"] == "test", "id"])))
    return hashlib.md5(ids.encode()).hexdigest()


def run_balance(st: "Study", opt: str, lr: float):
    """Modality-balancing variants of the proposed gated model; OGM's alpha is chosen on validation."""
    base = RunConfig(PROPOSED, opt, lr, seed=st.seeds[0], epochs=st.epochs)
    scores = {}
    for a in OGM_ALPHAS:
        r = st.run(replace(base, balance="ogm", ogm_alpha=a, name=f"bal_ogm_a{a:g}_s{st.seeds[0]}"), save_ckpt=True)
        scores[a] = r["val"]["i2i"] + r["val"]["i2i_strict"]
    alpha = max(scores, key=scores.get)
    (st.res_dir / "balance_alpha.json").write_text(json.dumps(
        {"scores": scores, "alpha": alpha, "test_split_fingerprint": split_fingerprint(st.data.meta)}))
    print(f"OGM alpha (validation): {scores} -> {alpha}", flush=True)
    for s in st.seeds:
        b = replace(base, seed=s)
        st.run(replace(b, balance="ogm", ogm_alpha=alpha, name=f"bal_ogm_a{alpha:g}_s{s}"), save_ckpt=s == st.seeds[0])
        st.run(replace(b, lambda_uni=0.5, name=f"bal_uni_s{s}"), save_ckpt=s == st.seeds[0])
        st.run(replace(b, title_noise=0.5, name=f"bal_noise_s{s}"), save_ckpt=s == st.seeds[0])
        st.run(replace(b, title_noise=0.5, lambda_uni=0.5, name=f"bal_noise_uni_s{s}"), save_ckpt=s == st.seeds[0])


def report_balance(root: Path, opt="SGD", lr=0.1, seeds=(0, 1, 2)):
    """Compare the balancing variants with the plain gated model. Needs only results/scratch/runs."""
    res_dir = root / "results" / "scratch"
    runs = res_dir / "runs"
    info = json.loads((res_dir / "balance_alpha.json").read_text())
    alpha = info["alpha"]
    meta_f = SCRATCH_DIR / "meta.csv"
    if "test_split_fingerprint" in info and meta_f.exists():
        same = split_fingerprint(pd.read_csv(meta_f)) == info["test_split_fingerprint"]
        print("test split identical to the main study:", same)
        if not same:
            raise SystemExit("The extension used a different test split; results are not comparable.")
    names = {"base": [f"{PROPOSED}_{opt}_lr{lr:g}_s{s}" for s in seeds],
             "ogm": [f"bal_ogm_a{alpha:g}_s{s}" for s in seeds],
             "uni": [f"bal_uni_s{s}" for s in seeds],
             "noise": [f"bal_noise_s{s}" for s in seeds],
             "noise_uni": [f"bal_noise_uni_s{s}" for s in seeds]}
    rows, sig = [], []
    perq = {}
    for v, rs in names.items():
        found = [r for r in rs if (runs / f"{r}.json").exists()]
        if not found:
            continue
        tests = [json.loads((runs / f"{r}.json").read_text())["test"] for r in found]
        row = {"variant": v, "name": BALANCE_NAMES[v] + (f" (alpha={alpha:g})" if "ogm" in v else ""),
               "seeds": len(found)}
        for k in METRICS:
            vals = [t.get(k, np.nan) for t in tests]
            row[f"{k} mean"], row[f"{k} std"] = float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1))
        rows.append(row)
        perq[v] = {k: np.mean([np.load(runs / f"{r}_perq.npz")[k] for r in found], axis=0) for k in METRICS}
    for v in perq:
        if v == "base" or "base" not in perq:
            continue
        for k in METRICS:
            a, b = perq[v][k], perq["base"][k]
            sig.append({"variant": v, "metric": k, "variant_ndcg": a.mean(), "base_ndcg": b.mean(),
                        "diff": a.mean() - b.mean(), "p_value": paired_bootstrap_p(a, b)})
    tab, sig = pd.DataFrame(rows), pd.DataFrame(sig)
    tab.round(4).to_csv(res_dir / "balance_comparison.csv", index=False)
    sig.round(5).to_csv(res_dir / "balance_significance.csv", index=False)
    # how strongly text dominated during training, and how OGM reacted (first seed)
    f = runs / f"bal_ogm_a{alpha:g}_s{seeds[0]}.json"
    if f.exists():
        h = pd.DataFrame(json.loads(f.read_text())["history"])
        if "ogm_ratio" in h:
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.plot(h["epoch"], h["ogm_ratio"], lw=2, label="text / image strength")
            ax.plot(h["epoch"], h["ogm_k_txt"], lw=2, label="gradient scale on text encoder")
            ax.axhline(1, ls="--", color="grey", lw=1)
            ax.set_xlabel("epoch")
            ax.set_title("Modality imbalance during training (gated + OGM)")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(res_dir / "balance_ogm_ratio.png", dpi=140)
            plt.close(fig)
    print(tab[["name"] + [f"{k} mean" for k in METRICS]].round(4).to_string(index=False))
    print(sig.round(4).to_string(index=False))
    return tab, sig


def bundle_balance(out: Path, zip_path: Path):
    """Zip only the extension's runs and checkpoints (unzip into the project, next to the main results)."""
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((out / "results" / "scratch" / "runs").glob("bal_*")):
            z.write(f, f"results/scratch/runs/{f.name}")
        for f in sorted((out / "results" / "scratch").glob("balance_*")):
            z.write(f, f"results/scratch/{f.name}")
        for f in sorted((out / "models" / "scratch").glob("bal_*.pt")):
            z.write(f, f"models/scratch/{f.name}")
    print(f"balance bundle -> {zip_path}")


def bundle(out: Path, zip_path: Path):
    """Zip everything the local app needs (no large training arrays)."""
    stage = zip_path.with_suffix("")
    if stage.exists():
        shutil.rmtree(stage)
    for rel in ["results/scratch", "models/scratch", "embeddings/scratch"]:
        if (out / rel).exists():
            shutil.copytree(out / rel, stage / rel, ignore=shutil.ignore_patterns("tune_*"))
    for rel in ["meta.csv", "vocab.json"]:
        (stage / "data/scratch").mkdir(parents=True, exist_ok=True)
        shutil.copy(SCRATCH_DIR / rel, stage / "data/scratch" / rel)
    shutil.copytree(SCRATCH_DIR / "test_images", stage / "data/scratch/test_images")
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", stage)
    shutil.rmtree(stage)
    print(f"bundle -> {zip_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT), help="where results/models go (e.g. a Google Drive folder)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tune-epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--stages", nargs="+", default=["tune", "optimizers", "methods", "ablations", "report", "export"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--smoke", action="store_true", help="tiny settings for a quick end-to-end check")
    ap.add_argument("--bundle", default=None, help="write a zip for the local app to this path")
    ap.add_argument("--bundle-balance", default=None, help="zip only the balancing extension to this path")
    ap.add_argument("--base-opt", default="SGD", help="optimizer of the main study (for the balance stage)")
    ap.add_argument("--base-lr", type=float, default=0.1)
    a = ap.parse_args()
    if a.stages == ["report_balance"]:  # local, no training data needed
        report_balance(Path(a.out), a.base_opt, a.base_lr, tuple(a.seeds))
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        a.epochs, a.tune_epochs, a.seeds, a.batch = 1, 1, [0, 1], 32
        for k in LR_GRID:
            LR_GRID[k] = LR_GRID[k][1:2]
    st = Study(Path(a.out), a.epochs, a.tune_epochs, a.seeds, device, a.smoke, a.batch)
    if "balance" in a.stages:
        run_balance(st, a.base_opt, a.base_lr)
    if "report_balance" in a.stages:
        report_balance(Path(a.out), a.base_opt, a.base_lr, tuple(a.seeds))
    if a.bundle_balance:
        bundle_balance(Path(a.out), Path(a.bundle_balance))
    main_stages = {"tune", "optimizers", "methods", "ablations", "report", "export"}
    if not main_stages & set(a.stages):
        return
    lrs_file, opt_file = st.res_dir / "tuned_lr.json", st.res_dir / "best_optimizer.json"
    lrs = st.tune() if "tune" in a.stages else json.loads(lrs_file.read_text())
    best_opt = st.optimizers(lrs) if "optimizers" in a.stages else json.loads(opt_file.read_text())["optimizer"]
    if "methods" in a.stages:
        st.methods(best_opt, lrs[best_opt])
    if "ablations" in a.stages:
        st.ablations(best_opt, lrs[best_opt])
    if "report" in a.stages:
        st.report(lrs, best_opt)
    if "export" in a.stages:
        st.export(lrs, best_opt)
    if a.bundle:
        bundle(Path(a.out), Path(a.bundle))


if __name__ == "__main__":
    main()
