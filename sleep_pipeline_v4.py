#!/usr/bin/env python3
"""
Wave-1 add-on experiments (cache-only, no download). Reuses features2_enc/ (encoder) & features2/.
EVAL subjects only; subject-disjoint impostors; enrol/probe disjoint epochs. Leakage-safe.

Stages:
  python sleep_pipeline_v4.py multistage   # single-stage vs multi-stage enrolment template (the FIX)
  python sleep_pipeline_v4.py threshold     # threshold set on stage A applied to stage B -> 5x5 HTER
  python sleep_pipeline_v4.py det           # DET curve points + FAR/FRR operating points (within & cross)
  python sleep_pipeline_v4.py tsne          # 2D embedding coords by subject & stage (killer figure)
"""
import argparse, glob, json, os, datetime, sys
import numpy as np
STAGES = ["W", "N1", "N2", "N3", "REM"]
def log(m): print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)
def d(r, *p): return os.path.join(r, *p)
def L2(X): return X/(np.linalg.norm(X, axis=1, keepdims=True)+1e-12)

def eval_subjects(root, tf, seed):
    subs = sorted({os.path.basename(f)[:6] for f in glob.glob(d(root, "features2", "*.npz"))})
    rng = np.random.default_rng(seed); rng.shuffle(subs); n = int(round(len(subs)*tf)); return subs[n:]
def load(root, folder, keep):
    c = {}
    for f in sorted(glob.glob(d(root, folder, "*.npz"))):
        s = os.path.basename(f)[:6]
        if s in keep: z = np.load(f, allow_pickle=True); c[s] = (z["X"], z["y"])
    return c
def far_frr(gen, imp, tau): return float((imp >= tau).mean()*100), float((gen < tau).mean()*100)
def eer_tau(gen, imp):
    s = np.concatenate([gen, imp]); l = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    o = np.argsort(-s); ls = l[o]; ss = s[o]; P, N = l.sum(), (1-l).sum()
    far = np.cumsum(1-ls)/N; frr = 1-np.cumsum(ls)/P; k = np.nanargmin(np.abs(far-frr))
    return float((far[k]+frr[k])/2*100), float(ss[k])

def pools(cache): return {s: {st: cache[s][0][cache[s][1] == st] for st in STAGES} for s in cache}

def scores(p, enrol_stages, probe_stage, ids, fuse=10, seed=42, min_ep=10):
    """genuine/impostor cosine scores: template from union of enrol_stages, probe from probe_stage."""
    rng = np.random.default_rng(seed)
    templ = {}
    for s in ids:
        chunks = [p[s][st] for st in enrol_stages if len(p[s][st]) >= 1]
        if not chunks: continue
        templ[s] = np.concatenate(chunks).mean(0)
    if len(templ) < 3: return None, None
    tids = list(templ); Tn = L2(np.stack([templ[s] for s in tids]))
    gen, imp = [], []
    for s in tids:
        Xb = p[s][probe_stage]
        if len(Xb) < min_ep: continue
        fe = min(fuse, len(Xb)); nt = max(1, len(Xb)//fe)
        idx = rng.permutation(len(Xb))[:nt*fe].reshape(nt, fe); Pn = L2(Xb[idx].mean(1))
        sims = Pn @ Tn.T; ti = tids.index(s)
        gen.append(sims[:, ti]); imp.append(np.delete(sims, ti, 1).ravel())
    if not gen: return None, None
    return np.concatenate(gen), np.concatenate(imp)

def stage_multistage(a):
    ev = eval_subjects(a.root, a.train_frac, a.seed); p = pools(load(a.root, "features2_enc", set(ev)))
    ids = sorted(p)
    sets = {"single_match": None, "all5": STAGES, "N2N3REM": ["N2", "N3", "REM"], "N3only": ["N3"]}
    res = {}
    for name, enrol in sets.items():
        row = {}
        for B in STAGES:
            es = [B] if enrol is None else enrol
            g, i = scores(p, es, B, ids, a.fuse, a.seed)
            if g is not None:
                e, _ = eer_tau(g, i); row[B] = round(e, 2)
        res[name] = row;
        vals=[v for v in row.values()]; log(f"  enrol={name}: mean probe-EER {np.mean(vals):.2f}")
    json.dump({"enrol_strategy_probeEER": res,
               "note": "Template built from enrol stage-set; probe each stage. 'single_match'=enrol==probe (within). 'all5'/'N2N3REM' = multi-stage template -> tests if a multi-stage enrolment reduces cross-stage EER (the FIX)."},
              open(d(a.root, "multistage.json"), "w"), indent=2)
    log("DONE -> multistage.json")

def stage_threshold(a):
    ev = eval_subjects(a.root, a.train_frac, a.seed); p = pools(load(a.root, "features2_enc", set(ev)))
    ids = sorted(p)
    sc = {st: scores(p, [st], st, ids, a.fuse, a.seed) for st in STAGES}  # within-stage scores per stage
    HTER = np.full((5, 5), np.nan)
    for ai, A in enumerate(STAGES):
        gA, iA = sc[A]
        if gA is None: continue
        _, tau = eer_tau(gA, iA)                       # threshold calibrated on A
        for bi, B in enumerate(STAGES):
            gB, iB = sc[B]
            if gB is None: continue
            far, frr = far_frr(gB, iB, tau); HTER[ai, bi] = (far+frr)/2
    import csv
    with open(d(a.root, "threshold_transfer_HTER.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["calibA\\applyB"]+STAGES)
        for i, st in enumerate(STAGES):
            w.writerow([st]+[f"{HTER[i,j]:.2f}" if not np.isnan(HTER[i, j]) else "NA" for j in range(5)])
    diag = np.nanmean([HTER[i, i] for i in range(5)]); off = np.nanmean(HTER[~np.eye(5, dtype=bool)])
    json.dump({"matched_threshold_HTER_mean": round(float(diag), 2),
               "mismatched_threshold_HTER_mean": round(float(off), 2),
               "interpretation": "Threshold calibrated on stage A applied to stage B. If off-diagonal HTER >> diagonal, a fixed operating point does NOT transfer across stages -> security risk; stage-specific calibration needed."},
              open(d(a.root, "threshold_transfer.json"), "w"), indent=2)
    log(f"DONE threshold: matched HTER {diag:.2f} vs mismatched {off:.2f} -> threshold_transfer.json/.csv")

def stage_det(a):
    ev = eval_subjects(a.root, a.train_frac, a.seed); p = pools(load(a.root, "features2_enc", set(ev)))
    ids = sorted(p)
    # pooled within-stage and cross-stage scores
    gw, iw, gc, ic = [], [], [], []
    for A in STAGES:
        for B in STAGES:
            g, i = scores(p, [A], B, ids, a.fuse, a.seed)
            if g is None: continue
            (gw if A == B else gc).append(g); (iw if A == B else ic).append(i)
    def curve(g, i):
        g = np.concatenate(g); i = np.concatenate(i); taus = np.quantile(np.concatenate([g, i]), np.linspace(0, 1, 200))
        pts = [{"tau": float(t), "FAR": round(float((i >= t).mean()*100), 3), "FRR": round(float((g < t).mean()*100), 3)} for t in taus]
        e, _ = eer_tau(g, i)
        # FRR at FAR=1% and FAR at FRR=1%
        frr_at_far1 = min((q["FRR"] for q in pts if q["FAR"] <= 1.0), default=None)
        far_at_frr1 = min((q["FAR"] for q in pts if q["FRR"] <= 1.0), default=None)
        return {"EER": round(e, 2), "FRR_at_FAR1pct": frr_at_far1, "FAR_at_FRR1pct": far_at_frr1, "curve": pts}
    out = {"within_stage": curve(gw, iw), "cross_stage": curve(gc, ic)}
    json.dump(out, open(d(a.root, "det.json"), "w"), indent=2)
    log(f"DONE det: within EER {out['within_stage']['EER']} (FRR@FAR1%={out['within_stage']['FRR_at_FAR1pct']}) | cross EER {out['cross_stage']['EER']} -> det.json")

def stage_tsne(a):
    ev = eval_subjects(a.root, a.train_frac, a.seed); cache = load(a.root, "features2_enc", set(ev))
    # sample up to 40 epochs/subject/stage to keep t-SNE tractable
    X = []; subj = []; stg = []
    rng = np.random.default_rng(a.seed)
    for s in sorted(cache):
        Xs, ys = cache[s]
        for st in STAGES:
            xs = Xs[ys == st]
            if len(xs) == 0: continue
            take = xs[rng.permutation(len(xs))[:40]]
            X.append(take); subj += [s]*len(take); stg += [st]*len(take)
    X = np.concatenate(X)
    from sklearn.manifold import TSNE
    Y = TSNE(n_components=2, init="pca", perplexity=30, random_state=a.seed).fit_transform(X)
    import csv
    with open(d(a.root, "tsne_coords.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["x", "y", "subject", "stage"])
        for (x, y), s, st in zip(Y, subj, stg): w.writerow([round(float(x), 3), round(float(y), 3), s, st])
    log(f"DONE tsne: {len(X)} points -> tsne_coords.csv (color by subject=identity clusters, by stage=state)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["multistage", "threshold", "det", "tsne"])
    ap.add_argument("--root", default="./"); ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--fuse", type=int, default=10)
    a = ap.parse_args(); log(f"STAGE={a.stage}")
    {"multistage": stage_multistage, "threshold": stage_threshold, "det": stage_det, "tsne": stage_tsne}[a.stage](a)

if __name__ == "__main__":
    sys.exit(main())
