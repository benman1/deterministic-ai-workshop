"""Shared, non-teaching helpers for the Meridian workshop notebooks.

Anything the workshop teaches (the conformal rule, the state machine, the
idempotent execute node) lives in notebook cells. This file holds only the
data loader, the interval formulas the slides quote, and a timing helper.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "meridian_triage.jsonl"
SEED = 20261030

ROUTES = [
    "baggage_claim",
    "cancellation_request",
    "out_of_scope",
    "payment_dispute",
    "rebooking_request",
    "refund_claim",
    "service_complaint",
    "visa_entry_question",
]
ROUTE_ID = {r: i for i, r in enumerate(ROUTES)}


def load(split: str | None = None) -> list[dict]:
    """Return the Meridian rows, optionally one of train / calibration / test."""
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    return [r for r in rows if split is None or r["split"] == split]


def texts_labels(rows: list[dict]) -> tuple[list[str], np.ndarray]:
    return [r["text"] for r in rows], np.array([ROUTE_ID[r["label"]] for r in rows])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k / n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_prop(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.3f} (Wilson 95%: {lo:.3f} to {hi:.3f})"


def cluster_bootstrap_ci(
    correct: np.ndarray, clusters: np.ndarray, n_boot: int = 5000, seed: int = SEED
) -> tuple[float, float]:
    """Percentile CI for accuracy, resampling whole clusters (scenarios)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(clusters)
    per = {c: correct[clusters == c] for c in ids}
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        vals = np.concatenate([per[c] for c in pick])
        stats[b] = vals.mean()
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def cluster_ids(rows: list[dict]) -> np.ndarray:
    """One id per (route, scenario): the five variants of a situation share it."""
    return np.array([f"{r['label']}:{r['scenario']}" for r in rows])


def hardware() -> str:
    """One line naming the machine, to print beside every latency number."""
    import os
    import subprocess

    cpu = platform.processor() or platform.machine()
    try:
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        pass
    return f"{cpu}, {os.cpu_count()} logical cores, {platform.system()} {platform.release()}"


def time_calls(fn, items, warmup: int = 3) -> dict:
    """Median and p95 wall-clock milliseconds per call of fn(item)."""
    for it in items[:warmup]:
        fn(it)
    ms = []
    for it in items:
        t = time.perf_counter()
        fn(it)
        ms.append((time.perf_counter() - t) * 1000)
    a = np.array(ms)
    return {"n": len(a), "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95))}


# ---- frozen-encoder embeddings (cached) ------------------------------------

ENCODERS = {
    "bge-small": dict(repo="BAAI/bge-small-en-v1.5", pool="cls"),
    "modernbert": dict(repo="answerdotai/ModernBERT-base", pool="mean"),
    "lfm2.5-enc": dict(repo="LiquidAI/LFM2.5-Encoder-230M", pool="mean", trust_remote_code=True),
}
CACHE = ROOT / "data" / "cache"


def load_encoder(name: str):
    """Return (tokenizer, model) for one of ENCODERS, in eval mode."""
    from transformers import AutoModel, AutoTokenizer

    cfg = ENCODERS[name]
    kw = {"trust_remote_code": True} if cfg.get("trust_remote_code") else {}
    tok = AutoTokenizer.from_pretrained(cfg["repo"], **kw)
    if name == "lfm2.5-enc":
        # The checkpoint stores the encoder under a "lfm2." prefix. AutoModel looks for no prefix,
        # finds nothing, and silently initializes every layer at random. Load the masked-LM class
        # and take its encoder body instead.
        from transformers import AutoModelForMaskedLM

        mod = AutoModelForMaskedLM.from_pretrained(cfg["repo"], **kw).eval().lfm2
    else:
        mod = AutoModel.from_pretrained(cfg["repo"], **kw).eval()
    return tok, mod


def encode(name: str, texts: list[str], batch_size: int = 32, max_length: int = 256, _loaded=None):
    """Frozen embeddings, L2-normalized, pooled as ENCODERS[name] says."""
    import torch

    tok, mod = _loaded or load_encoder(name)
    pool = ENCODERS[name]["pool"]
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            b = tok(texts[i : i + batch_size], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
            h = mod(**b).last_hidden_state
            if pool == "cls":
                e = h[:, 0]
            else:
                mask = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                e = (h * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).float().numpy())
    return np.concatenate(out)


def cached_embeddings(name: str) -> dict[str, np.ndarray]:
    """Embeddings of every row for one encoder, by split, cached under data/cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npz"
    rows = load()
    if path.exists():
        z = np.load(path)
        return {s: z[s] for s in ("train", "calibration", "test")}
    loaded = load_encoder(name)
    res = {}
    for s in ("train", "calibration", "test"):
        res[s] = encode(name, [r["text"] for r in rows if r["split"] == s], _loaded=loaded)
    np.savez(path, **res)
    return res
