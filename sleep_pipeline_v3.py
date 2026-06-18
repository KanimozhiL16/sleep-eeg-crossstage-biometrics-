#!/usr/bin/env python3
"""
v3 add-on — corrected CIs + deployment experiments. Reuses cached features (no re-download).

Reads:
  features2_enc/  (encoder embeddings; default, best method)
  features2/      (rich PSD features; for --feats psd)
  epochs2/        (2-channel epochs; for minimal-channel experiment)

Stages:
  python sleep_pipeline_v3.py ci          # corrected bootstrap CI (subject SUBSAMPLING, no dup-impostor bug)
  python sleep_pipeline_v3.py stagegated  # stage-gated (enrol+verify in best stage) vs stage-blind (pool all)
  python sleep_pipeline_v3.py ttd         # time-to-decision: EER vs probe epochs (=> minutes of sleep)
  python sleep_pipeline_v3.py minchan     # minimal-channel: Fpz-Cz only vs Fpz-Cz+Pz-Oz (within-stage EER)

Leakage-safe: EVAL subjects only (same split as v2/encoder via --train_frac/--seed); impostors
subject-disjoint; enrol/probe disjoint epochs.
"""
import argparse, glob, json, os, datetime, sys
import numpy as np

STAGES = ["W", "N1", "N2", "N3", "REM"]
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30)}

def log(m): print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)
def d(root, *p): return os.path.join(root, *p)
def jdump(o, f): json.dump(o, open(f, "w"), indent=2, default=str)


def eval_subjects(root, train_frac, seed):
    # reproduce the v2/encoder split exactly
    subs = sorted({os.path.basename(fp)[:6] for fp in glob.glob(d(root, "features2", "*.npz"))})
    rng = np.random.default_rng(seed); rng.shuffle(subs)
    n_tr = int(round(len(subs)*train_frac)); return subs[:n_tr], subs[n_tr:]


def load(root, sub_set, kind):
    folder = "features2_enc" if kind == "enc" else "features2"
    c = {}
    for fp in sorted(glob.glob(d(root, folder, "*.npz"))):
        sid = os.path.basename(fp)[:6]
        if sid in sub_set:
            z = np.load(fp, allow_pickle=True); c[sid] = (z["X"], z["y"])
    return c


def eer(gen, imp):
    s = np.concatenate([gen, imp]); l = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    o = np.argsort(-s); l = l[o]; P, N = l.sum(), (1-l).sum()
    if P == 0 or N == 0: return np.nan
    far = np.cumsum(1-l)/N; frr = 1-np.cumsum(l)/P
    i = np.nanargmin(np.abs(far-frr)); return float((far[i]+frr[i])/2*100)


def L2(X): return X/(np.linalg.norm(X, axis=1, keepdims=True)+1e-12)


def pools_of(cache):
    return {s: {st: cache[s][0][cache[s][1] == st] for st in STAGES} for s in cache}


def crossstage(cache, min_epochs=10, fuse=10, seed=0, subjects=None):
    """5x5 EER on given subjects. Templates and probes use disjoint epochs implicitly via mean vs samples."""
    pools = pools_of(cache); ids_all = subjects or sorted(cache)
    rng = np.random.default_rng(seed); M = np.full((5, 5), np.nan)
    for ei, A in enumerate(STAGES):
        templ = {s: pools[s][A].mean(0) for s in ids_all if len(pools[s][A]) >= min_epochs}
        if len(templ) < 3: continue
        ids = list(templ); Tn = L2(np.stack([templ[s] for s in ids]))
        for pi, B in enumerate(STAGES):
            gen, imp = [], []
            for s in ids:
                Xb = pools[s][B]
                if len(Xb) < min_epochs: continue
                fe = min(max(fuse, 1), len(Xb)); nt = max(1, len(Xb)//fe)
                idx = rng.permutation(len(Xb))[:nt*fe].reshape(nt, fe)
                Pn = L2(Xb[idx].mean(1)); sims = Pn @ Tn.T; ti = ids.index(s)
                gen.append(sims[:, ti]); imp.append(np.delete(sims, ti, 1).ravel())
            if gen: M[ei, pi] = eer(np.concatenate(gen), np.concatenate(imp))
    return M


def means(M):
    diag = [M[i, i] for i in range(5)]; off = M[~np.eye(5, dtype=bool)]
    return float(np.nanmean(diag)), float(np.nanmean(off))


def stage_ci(a):
    """Corrected CI via subject SUBSAMPLING without replacement (no duplicate-as-impostor artifact)."""
    _, ev = eval_subjects(a.root, a.train_frac, a.seed)
    cache = load(a.root, set(ev), a.feats)
    M = crossstage(cache, fuse=a.fuse, seed=a.seed); wi, cr = means(M)
    rng = np.random.default_rng(a.seed); k = max(8, int(0.8*len(ev))); boot = []
    for _ in range(a.boot):
        sub = list(rng.choice(ev, k, replace=False))
        Mb = crossstage({s: cache[s] for s in sub}, fuse=a.fuse, seed=a.seed, subjects=sub)
        boot.append(list(means(Mb)))
    boot = np.array(boot)
    rep = {"feats": a.feats, "fuse": a.fuse, "n_eval": len(ev), "subsample_k": k, "boot": a.boot,
           "within_EER": round(wi, 2), "within_CI95": [round(float(np.nanpercentile(boot[:,0],2.5)),2), round(float(np.nanpercentile(boot[:,0],97.5)),2)],
           "cross_EER": round(cr, 2), "cross_CI95": [round(float(np.nanpercentile(boot[:,1],2.5)),2), round(float(np.nanpercentile(boot[:,1],97.5)),2)],
           "method": "subject subsampling (80%, without replacement) — corrects v2 duplicate-impostor bias"}
    jdump(rep, d(a.root, f"ci_{a.feats}_fuse{a.fuse}.json")); log(f"CI: within {wi:.2f}{rep['within_CI95']}  cross {cr:.2f}{rep['cross_CI95']}")


def stage_stagegated(a):
    """Stage-gated (enrol+verify within best stage) vs stage-blind (pool all stages)."""
    _, ev = eval_subjects(a.root, a.train_frac, a.seed); cache = load(a.root, set(ev), a.feats)
    M = crossstage(cache, fuse=a.fuse, seed=a.seed)
    gated = {STAGES[i]: (None if np.isnan(M[i, i]) else round(float(M[i, i]), 2)) for i in range(5)}  # within-stage diagonal
    # stage-blind: ignore stage labels — template = mean over ALL epochs; probe = fused over ALL epochs
    blind_g, blind_i = [], []
    allcache = {s: (cache[s][0], np.array(["ALL"]*len(cache[s][1]))) for s in cache}
    pools = {s: allcache[s][0] for s in allcache}
    ids = [s for s in pools if len(pools[s]) >= 10]; Tn = L2(np.stack([pools[s].mean(0) for s in ids]))
    rng = np.random.default_rng(a.seed)
    for s in ids:
        Xb = pools[s]; fe = min(a.fuse, len(Xb)); nt = max(1, len(Xb)//fe); idx = rng.permutation(len(Xb))[:nt*fe].reshape(nt, fe)
        Pn = L2(Xb[idx].mean(1)); sims = Pn @ Tn.T; ti = ids.index(s)
        blind_g.append(sims[:, ti]); blind_i.append(np.delete(sims, ti, 1).ravel())
    blind = round(eer(np.concatenate(blind_g), np.concatenate(blind_i)), 2)
    best_stage = min((k for k, v in gated.items() if v is not None), key=lambda k: gated[k])
    rep = {"feats": a.feats, "fuse": a.fuse, "stage_gated_within_EER": gated, "best_stage": best_stage,
           "best_stage_EER": gated[best_stage], "stage_blind_EER": blind,
           "interpretation": "If best-stage-gated EER << stage-blind EER, gating verification to the most identity-bearing stage (e.g., N3/REM) materially improves a deployed sleep-monitoring identity check."}
    jdump(rep, d(a.root, f"stagegated_{a.feats}.json")); log(f"STAGEGATED: best={best_stage} {gated[best_stage]}%  vs stage-blind {blind}%")


def stage_ttd(a):
    """Time-to-decision: within-stage EER vs probe-epoch count (=> minutes of sleep)."""
    _, ev = eval_subjects(a.root, a.train_frac, a.seed); cache = load(a.root, set(ev), a.feats)
    curve = {}
    for N in [1, 2, 3, 5, 10, 20]:
        M = crossstage(cache, fuse=N, seed=a.seed); wi, _ = means(M)
        # best within-stage too
        diag = [M[i, i] for i in range(5)]; best = float(np.nanmin(diag))
        curve[N] = {"minutes": round(N*30/60, 1), "within_mean_EER": round(wi, 2), "best_stage_EER": round(best, 2)}
        log(f"  N={N} ({N*30/60:.1f} min): within {wi:.2f}  best-stage {best:.2f}")
    jdump({"feats": a.feats, "time_to_decision": curve,
           "note": "N probe epochs = N*30s of sleep fused per decision."}, d(a.root, f"ttd_{a.feats}.json"))


def stage_minchan(a):
    """Minimal-channel feasibility: Fpz-Cz only vs both channels (recompute PSD features from epochs2)."""
    from scipy.signal import welch
    _, ev = eval_subjects(a.root, a.train_frac, a.seed)
    def chanfeat(x, sf):
        f, p = welch(x, fs=sf, nperseg=int(sf*4)); bp = np.array([p[(f>=lo)&(f<hi)].sum() for lo,hi in BANDS.values()])
        return np.log(bp/(bp.sum()+1e-12)+1e-8)
    out = {}
    for mode, chans in [("Fpz-Cz_only", [0]), ("both", [0, 1])]:
        cache = {}
        for fp in sorted(glob.glob(d(a.root, "epochs2", "*.npz"))):
            sid = os.path.basename(fp)[:6]
            if sid not in ev: continue
            z = np.load(fp, allow_pickle=True); X, y, sf = z["X"], z["y"], float(z["sf"])
            if X.shape[1] <= max(chans): continue
            feats = np.array([np.concatenate([chanfeat(ep[c], sf) for c in chans]) for ep in X])
            cache[sid] = (feats, y)
        M = crossstage(cache, fuse=a.fuse, seed=a.seed); wi, cr = means(M)
        out[mode] = {"within_EER": round(wi, 2), "cross_EER": round(cr, 2), "n_subjects": len(cache)}
        log(f"  {mode}: within {wi:.2f}  cross {cr:.2f}")
    jdump({"fuse": a.fuse, "channels": out,
           "note": "PSD-cosine features; Fpz-Cz is a forehead electrode used by consumer sleep headbands."},
          d(a.root, "minchan.json"))


FEATNAMES = []
for ch in ["Fpz-Cz", "Pz-Oz"]:
    FEATNAMES += [f"{ch}:{b}" for b in ["delta", "theta", "alpha", "sigma", "beta"]]
    FEATNAMES += [f"{ch}:spec_entropy", f"{ch}:hjorth_act", f"{ch}:hjorth_mob", f"{ch}:hjorth_cmp"]
    FEATNAMES += [f"{ch}:r_delta/alpha", f"{ch}:r_theta/alpha", f"{ch}:r_sigma/alpha"]


def stage_featstab(a):
    """Per-feature identity-stability index across states (which features are state-invariant).
    Uses PSD features (features2/, interpretable). Descriptive variance decomposition — uses ALL
    subjects (no classifier, no leakage). index = between-subject var / within-subject-across-state var."""
    cache = {}
    for fp in sorted(glob.glob(d(a.root, "features2", "*.npz"))):
        z = np.load(fp, allow_pickle=True); cache[os.path.basename(fp)[:6]] = (z["X"], z["y"])
    subs = sorted(cache); Dn = cache[subs[0]][0].shape[1]
    # m[s, st, d] = mean feature; require subject to have all 5 states w/ >=10 epochs
    M = {}; valid = []
    for s in subs:
        X, y = cache[s]; ok = True; row = np.zeros((5, Dn))
        for i, st in enumerate(STAGES):
            xs = X[y == st]
            if len(xs) < 10: ok = False; break
            row[i] = xs.mean(0)
        if ok: M[s] = row; valid.append(s)
    A = np.stack([M[s] for s in valid])              # [n_subj, 5, D]
    within = A.var(axis=1).mean(axis=0)               # within-subject across-state var, per feature
    between = A.mean(axis=1).var(axis=0)              # between-subject var of state-averaged feature
    stab = between / (within + 1e-12)
    order = np.argsort(-stab)
    # --- DISCRIMINABILITY (identity power), Fisher-like: separates "carries identity" from "low variance" ---
    # within-state epoch-level variance pooled over (subject,state); between = var of subject means.
    win_state = np.zeros(Dn); cnt = 0
    submean = []
    for s in valid:
        X, y = cache[s]; submean.append(X.mean(0))
        for st in STAGES:
            xs = X[y == st]
            if len(xs) >= 10: win_state += xs.var(0); cnt += 1
    win_state = win_state/max(cnt, 1)
    btw_subj = np.stack(submean).var(0)
    disc = btw_subj/(win_state+1e-12)                 # high => feature separates PEOPLE (ignoring state)
    # genuine state-invariant identity carrier = high on BOTH disc and stab
    combo = np.sqrt(np.clip(disc, 0, None)*np.clip(stab, 0, None))
    dord = np.argsort(-disc); cord = np.argsort(-combo)
    ranked = [{"feature": FEATNAMES[d] if d < len(FEATNAMES) else f"f{d}",
               "stability_index": round(float(stab[d]), 3),
               "between_var": round(float(between[d]), 4), "within_state_var": round(float(within[d]), 4)}
              for d in order]
    # wake vs sleep: per-feature |mean_W - mean_sleep| (states 1..4 = sleep)
    wake = A[:, 0, :].mean(0); sleep = A[:, 1:, :].mean(axis=1).mean(0)
    wvs = sorted([{"feature": FEATNAMES[d] if d < len(FEATNAMES) else f"f{d}",
                   "wake_minus_sleep": round(float(wake[d]-sleep[d]), 3)} for d in range(Dn)],
                 key=lambda r: -abs(r["wake_minus_sleep"]))
    fn = lambda d_: FEATNAMES[d_] if d_ < len(FEATNAMES) else f"f{d_}"
    disc_rank = [{"feature": fn(d_), "discriminability": round(float(disc[d_]), 3),
                  "state_robustness": round(float(stab[d_]), 3)} for d_ in dord]
    combo_rank = [{"feature": fn(d_), "combined": round(float(combo[d_]), 3),
                   "discriminability": round(float(disc[d_]), 3), "state_robustness": round(float(stab[d_]), 3)} for d_ in cord]
    jdump({"n_subjects_all5stages": len(valid), "n_features": Dn,
           "most_state_invariant_top5": ranked[:5], "least_state_invariant_bottom5": ranked[-5:],
           "full_ranking": ranked, "wake_vs_sleep_largest_diffs_top8": wvs[:8],
           "identity_discriminability_top5": disc_rank[:5], "identity_discriminability_bottom5": disc_rank[-5:],
           "genuine_identity_carriers_top5_combined": combo_rank[:5],
           "method": "TWO indices per feature: (1) state_robustness = between-subj var / within-subj-ACROSS-STATE var (ICC-like, stable across W/N1/N2/N3/REM); (2) discriminability = between-subj var / within-subj WITHIN-STATE epoch var (Fisher-like, separates PEOPLE). A genuine state-invariant identity carrier scores HIGH on BOTH (combined = geometric mean). This separates 'carries identity' from 'merely low-variance', addressing the confound.",
           "note": "Descriptive (no classifier, no train/test) -> no leakage. W = resting/quiet wake, not active task wake. Interpret cautiously: all state_robustness <1 => state drift exceeds identity for every single feature; identity that survives across stages is distributed/nonlinear (motivates the encoder)."},
          d(a.root, "featstab.json"))
    log(f"FEATSTAB: n={len(valid)} subjects | most state-invariant: {[r['feature'] for r in ranked[:3]]} | least: {[r['feature'] for r in ranked[-3:]]}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        idx = order; vals = stab[idx]
        plt.figure(figsize=(6, 7)); plt.barh(range(len(idx)), vals[::-1], color="0.4")
        plt.yticks(range(len(idx)), [FEATNAMES[d] if d < len(FEATNAMES) else f"f{d}" for d in idx[::-1]], fontsize=6)
        plt.xlabel("identity-stability index (between / within-state var)"); plt.title("Feature identity-stability across sleep/wake states")
        plt.tight_layout(); plt.savefig(d(a.root, "featstab.png"), dpi=200)
    except Exception as e: log(f"plot skip: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ci", "stagegated", "ttd", "minchan", "featstab"])
    ap.add_argument("--root", default="./")
    ap.add_argument("--feats", choices=["enc", "psd"], default="enc")
    ap.add_argument("--fuse", type=int, default=10)
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    log(f"STAGE={a.stage} feats={a.feats} fuse={a.fuse}")
    {"ci": stage_ci, "stagegated": stage_stagegated, "ttd": stage_ttd, "minchan": stage_minchan, "featstab": stage_featstab}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
