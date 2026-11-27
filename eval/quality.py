"""Output-quality comparison: squid7800xt NVFP4 vs. a llama.cpp GGUF (e.g. unsloth UD-IQ4_XS).

Both engines score the *same token ids* teacher-forced, so every number is a paired comparison.

  python -I eval/quality.py prepare            # tokenize eval text via llama-server -> eval/out/tokens_*.i32
  python -I eval/quality.py gguf               # score with llama-server (must be running, --gguf-url)
  python -I eval/quality.py ours               # score with zig-out/bin/squid7800xt.exe score
  python -I eval/quality.py compare            # print the report (and write eval/out/report.json)

Metrics (per domain):
  ppl            perplexity on held-out text; lower = closer to the real data distribution
  dNLL           mean paired difference in -log p(true token), ours - gguf, with 95% bootstrap CI
  top1 agree     how often the two engines' greedy next token is the same
  top1 acc       how often each engine's greedy next token equals the real next token
  KL(g||o)       approx. KL divergence of ours from gguf over gguf's top-K tokens
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "eval", "data")
OUT = os.path.join(ROOT, "eval", "out")
EXE = os.path.join(ROOT, "zig-out", "bin", "squid7800xt.exe")
MODEL_DIR = os.path.join(ROOT, "models", "inkling-2-mini")

TOPK = 20
DOMAINS = ("wiki", "code")


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def rec_dtype(k):
    return np.dtype([("lp", "<f4"), ("ent", "<f4"), ("ids", "<i4", (k,)), ("lps", "<f4", (k,))])


# ---------------------------------------------------------------- prepare
def load_text(domain):
    if domain == "wiki":
        with open(os.path.join(DATA, "wikitext-2-raw", "wiki.test.raw"), encoding="utf-8") as f:
            return f.read()
    parts = []
    with open(os.path.join(DATA, "HumanEval.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            parts.append(o["prompt"] + o["canonical_solution"] + "\n\n" + o["test"])
    return "\n\n".join(parts)


def cmd_prepare(a):
    os.makedirs(OUT, exist_ok=True)
    for domain, n_win in (("wiki", a.wiki_windows), ("code", a.code_windows)):
        text = load_text(domain)
        toks = post(a.gguf_url + "/tokenize", {"content": text, "add_special": False})["tokens"]
        n_avail = len(toks) // a.window
        if n_avail < n_win:
            sys.exit(f"{domain}: only {n_avail} windows of {a.window} available")
        # evenly spaced, non-overlapping windows across the whole text
        starts = [int(i * (len(toks) - a.window) / max(n_win - 1, 1)) for i in range(n_win)]
        arr = np.concatenate([np.asarray(toks[s:s + a.window], dtype="<i4") for s in starts])
        path = os.path.join(OUT, f"tokens_{domain}.i32")
        arr.tofile(path)
        det = post(a.gguf_url + "/detokenize", {"tokens": arr[:24].tolist()})["content"]
        print(f"{domain}: {len(toks)} tokens total, {n_win} windows x {a.window} -> {path}")
        print(f"  gguf detok of first ids: {det!r}")
    json.dump({"window": a.window}, open(os.path.join(OUT, "meta.json"), "w"))


# ---------------------------------------------------------------- gguf
def cmd_gguf(a):
    window = json.load(open(os.path.join(OUT, "meta.json")))["window"]
    dt = rec_dtype(TOPK)
    for domain in DOMAINS:
        toks = np.fromfile(os.path.join(OUT, f"tokens_{domain}.i32"), dtype="<i4")
        n_win = len(toks) // window
        for w in range(n_win):
            path = os.path.join(OUT, f"gguf_{domain}_{w:03d}.bin")
            if os.path.exists(path):
                continue
            win = toks[w * window:(w + 1) * window].tolist()
            recs = np.zeros(window - 1, dtype=dt)
            t0 = time.time()
            for i in range(window - 1):
                target = win[i + 1]
                body = {
                    "prompt": win[:i + 1],
                    "n_predict": 1,
                    "n_probs": TOPK,
                    "cache_prompt": True,
                    "temperature": 0.0,
                    # Force the sampled token to be the true next token. The reported probability is
                    # computed by llama-server from the raw logits (softmax over the full vocab),
                    # before the logit-bias sampler, so this returns the exact p(target).
                    "logit_bias": [[target, 1000.0]],
                    "post_sampling_probs": False,
                }
                resp = post(a.gguf_url + "/completion", body)
                if "completion_probabilities" not in resp:
                    # llama-server withholds output (and its probabilities) when the token is a lone
                    # byte of an incomplete UTF-8 sequence. Mark it; compare drops it for both engines.
                    recs[i]["lp"] = np.nan
                    recs[i]["ent"] = np.nan
                    recs[i]["ids"] = -1
                    recs[i]["lps"] = -np.inf
                    continue
                cp = resp["completion_probabilities"][0]
                if cp["id"] != target:
                    sys.exit(f"forced token mismatch at {domain} w{w} i{i}: {cp['id']} != {target}")
                recs[i]["lp"] = cp["logprob"]
                recs[i]["ent"] = np.nan
                top = cp["top_logprobs"][:TOPK]
                recs[i]["ids"] = [t["id"] for t in top] + [-1] * (TOPK - len(top))
                recs[i]["lps"] = [t["logprob"] for t in top] + [-np.inf] * (TOPK - len(top))
            recs.tofile(path)
            ppl = math.exp(-float(np.nanmean(recs["lp"])))
            print(f"[gguf] {domain} window {w + 1}/{n_win}: ppl={ppl:.3f}  {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- ours
def cmd_ours(a):
    window = json.load(open(os.path.join(OUT, "meta.json")))["window"]
    for domain in DOMAINS:
        cmd = [EXE, "score", MODEL_DIR, "--tokens", os.path.join(OUT, f"tokens_{domain}.i32"),
               "--out", os.path.join(OUT, f"ours_{domain}.bin"), "--window", str(window),
               "--topk", str(TOPK), "--ctx", str(window + 64)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)


# ---------------------------------------------------------------- compare
def load_gguf(domain, n_win):
    return np.concatenate([np.fromfile(os.path.join(OUT, f"gguf_{domain}_{w:03d}.bin"), dtype=rec_dtype(TOPK))
                           for w in range(n_win)])


def bootstrap_ci(x, win_ids=None, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    if win_ids is not None:
        uniq = np.unique(win_ids)
        w_sums = np.array([x[win_ids == w].sum() for w in uniq], dtype=np.float64)
        w_cnts = np.array([(win_ids == w).sum() for w in uniq], dtype=np.float64)
        idx = rng.integers(0, len(uniq), size=(n, len(uniq)))
        means = w_sums[idx].sum(axis=1) / w_cnts[idx].sum(axis=1)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cmd_compare(a):
    window = json.load(open(os.path.join(OUT, "meta.json")))["window"]
    report = {}
    for domain in DOMAINS:
        toks = np.fromfile(os.path.join(OUT, f"tokens_{domain}.i32"), dtype="<i4")
        n_win = len(toks) // window
        o = np.fromfile(os.path.join(OUT, f"ours_{domain}.bin"), dtype=rec_dtype(TOPK))
        g = load_gguf(domain, n_win)
        assert len(o) == len(g) == n_win * (window - 1)
        targets = np.concatenate([toks[w * window + 1:(w + 1) * window] for w in range(n_win)])
        # skip the first positions of each window, where both models have almost no context
        pos = np.tile(np.arange(window - 1), n_win)
        win_ids = np.repeat(np.arange(n_win), window - 1)
        valid = ~np.isnan(g["lp"])
        n_dropped = int((~valid).sum())
        m = (pos >= a.skip) & valid
        o, g, targets, win_ids = o[m], g[m], targets[m], win_ids[m]

        nll_o, nll_g = -o["lp"].astype(np.float64), -g["lp"].astype(np.float64)
        d = nll_o - nll_g
        lo, hi = bootstrap_ci(d, win_ids)

        # approx KL(gguf || ours) over gguf's top-K; ours' logprob for tokens outside its top-K is
        # capped at its K-th logprob (an over-estimate of q, so this under-estimates KL slightly)
        kl = np.zeros(len(g))
        for j in range(len(g)):
            omap = dict(zip(o["ids"][j].tolist(), o["lps"][j].tolist()))
            ofloor = float(o["lps"][j][-1])
            gp = np.exp(g["lps"][j].astype(np.float64))
            oq = np.array([omap.get(t, ofloor) for t in g["ids"][j].tolist()])
            kl[j] = float(np.sum(gp * (g["lps"][j] - oq)))

        r = {
            "positions": int(len(d)),
            "dropped_partial_utf8": n_dropped,
            "ppl_ours": math.exp(nll_o.mean()),
            "ppl_gguf": math.exp(nll_g.mean()),
            "dnll_mean": float(d.mean()), "dnll_ci95": [lo, hi],
            "top1_agree": float((o["ids"][:, 0] == g["ids"][:, 0]).mean()),
            "top1_acc_ours": float((o["ids"][:, 0] == targets).mean()),
            "top1_acc_gguf": float((g["ids"][:, 0] == targets).mean()),
            "kl_gguf_ours_mean": float(kl.mean()),
            "kl_gguf_ours_p99": float(np.percentile(kl, 99)),
            "ours_better_frac": float((d < 0).mean()),
        }
        report[domain] = r
        print(f"\n== {domain}  ({r['positions']} scored tokens, {n_win} windows x {window}, skip first {a.skip}, "
              f"{n_dropped} partial-UTF-8 positions dropped) ==")
        print(f"  perplexity        ours {r['ppl_ours']:.4f}   gguf {r['ppl_gguf']:.4f}   "
              f"({100 * (r['ppl_ours'] / r['ppl_gguf'] - 1):+.2f}%)")
        print(f"  dNLL ours-gguf    {r['dnll_mean']:+.4f} nats/token  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"(negative = ours better)")
        print(f"  top-1 accuracy    ours {100 * r['top1_acc_ours']:.2f}%   gguf {100 * r['top1_acc_gguf']:.2f}%")
        print(f"  top-1 agreement   {100 * r['top1_agree']:.2f}%")
        print(f"  KL(gguf||ours)    mean {r['kl_gguf_ours_mean']:.4f}  p99 {r['kl_gguf_ours_p99']:.4f} nats (top-{TOPK} approx)")
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(OUT, 'report.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "gguf", "ours", "compare"])
    ap.add_argument("--gguf-url", default="http://127.0.0.1:8090")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--wiki-windows", type=int, default=24)
    ap.add_argument("--code-windows", type=int, default=12)
    ap.add_argument("--skip", type=int, default=32)
    a = ap.parse_args()
    {"prepare": cmd_prepare, "gguf": cmd_gguf, "ours": cmd_ours, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
