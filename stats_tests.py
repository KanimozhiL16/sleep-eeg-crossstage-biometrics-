#!/usr/bin/env python3
"""
Statistical validation (no new data/GPU). Per-subject EERs on EVAL subjects, then:
  - within-stage vs cross-stage gap: Wilcoxon signed-rank + paired bootstrap ΔEER 95% CI + Cohen's d
  - method comparison (encoder vs cosine): paired Wilcoxon on per-subject within-stage EER
Reuses features2_enc/ (encoder embeddings) and features2/ (PSD). EVAL split = same seed/train_frac.

Run:  python stats_tests.py --root ./ 2>&1 | tee stats.log
"""
import argparse, glob, json, os, datetime, sys
import numpy as np

STAGES = ["W", "N1", "N2", "N3", "REM"]
def log(m): print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)
def d(r, *p): return os.path.join(r, *p)
def L2(X): return X/(np.linalg.norm(X, axis=1, keepdims=True)+1e-12)

def eer(g, i):
    s = np.concatenate([g, i]); l = np.concatenate([np.ones_like(g), np.zeros_like(i)])
    o = np.argsort(-s); l = l[o]; P, N = l.sum(), (1-l).sum()
    if P == 0 or N == 0: return np.nan
    far = np.cumsum(1-l)/N; frr = 1-np.cumsum(l)/P
    k = np.nanargmin(np.abs(far-frr)); return float((far[k]+frr[k])/2*100)

def eval_subjects(root, tf, seed):
    subs = sorted({os.path.basename(f)[:6] for f in glob.glob(d(root, "features2", "*.npz"))})
    rng = np.random.default_rng(seed); rng.shuffle(subs); n = int(round(len(subs)*tf)); return subs[n:]

def load(root, folder, keep):
    c = {}
    for f in sorted(glob.glob(d(root, folder, "*.npz"))):
        s = os.path.basename(f)[:6]
        if s in keep: z = np.load(f, allow_pickle=True); c[s] = (z["X"], z["y"])
    return c

def per_subject_eer(cache, fuse=10, mode="within", min_ep=10, seed=42):
    """Return dict subj -> EER, pooling the relevant (enrolA,probeB) pairs. mode: within(A==B) or cross(A!=B)."""
    subs = sorted(cache); rng = np.random.default_rng(seed)
    pools = {s: {st: cache[s][0][cache[s][1] == st] for st in STAGES} for s in subs}
    templ = {s: {st: pools[s][st].mean(0) for st in STAGES if len(pools[s][st]) >= min_ep} for s in subs}
    out = {}
    for s in subs:
        gen, imp = [], []
        for A in STAGES:
            if A not in templ[s]: continue
            others = [o for o in subs if o != s and A in templ[o]]
            if not others: continue
            Tn_self = L2(templ[s][A][None])[0]
            Tn_oth = L2(np.stack([templ[o][A] for o in others]))
            for B in STAGES:
                if (mode == "within") != (A == B): continue
                Xb = pools[s][B]
                if len(Xb) < min_ep: continue
                fe = min(fuse, len(Xb)); nt = max(1, len(Xb)//fe)
                idx = rng.permutation(len(Xb))[:nt*fe].reshape(nt, fe)
                Pn = L2(Xb[idx].mean(1))
                gen.append(Pn @ Tn_self); imp.append((Pn @ Tn_oth.T).ravel())
        if gen and imp:
            e = eer(np.concatenate(gen), np.concatenate(imp))
            if not np.isnan(e): out[s] = e
    return out

def paired(aw, ac):
    """aw, ac: dict subj->EER (within, cross). Return aligned arrays."""
    k = sorted(set(aw) & set(ac)); return np.array([aw[s] for s in k]), np.array([ac[s] for s in k]), k

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="./")
    ap.add_argument("--train_frac", type=float, default=0.6); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fuse", type=int, default=10); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None; log("scipy.stats.wilcoxon unavailable -> reporting bootstrap only")
    ev = eval_subjects(a.root, a.train_frac, a.seed); log(f"EVAL n={len(ev)}")
    enc = load(a.root, "features2_enc", set(ev)); psd = load(a.root, "features2", set(ev))
    rng = np.random.default_rng(a.seed)

    def gap_test(cache, tag):
        w = per_subject_eer(cache, a.fuse, "within", seed=a.seed)
        c = per_subject_eer(cache, a.fuse, "cross", seed=a.seed)
        W, C, ks = paired(w, c); diff = C - W
        # paired bootstrap of mean difference
        bd = [np.mean(rng.choice(diff, len(diff), replace=True)) for _ in range(a.boot)]
        ci = [round(float(np.percentile(bd, 2.5)), 2), round(float(np.percentile(bd, 97.5)), 2)]
        d_cohen = float(np.mean(diff)/(np.std(diff, ddof=1)+1e-12))
        res = {"tag": tag, "n_paired": len(ks),
               "within_mean": round(float(W.mean()), 2), "cross_mean": round(float(C.mean()), 2),
               "mean_diff_cross_minus_within": round(float(diff.mean()), 2),
               "diff_95CI_bootstrap": ci, "cohens_d": round(d_cohen, 2),
               "within_median": round(float(np.median(W)), 2), "cross_median": round(float(np.median(C)), 2)}
        if wilcoxon is not None and len(ks) >= 6:
            try:
                st, p = wilcoxon(C, W, alternative="greater")
                res["wilcoxon_cross_gt_within_p"] = float(p); res["wilcoxon_stat"] = float(st)
            except Exception as e: res["wilcoxon_error"] = str(e)
        return res, w

    enc_gap, enc_within = gap_test(enc, "encoder")
    psd_gap, psd_within = gap_test(psd, "psd_cosine")
    # method comparison: encoder vs cosine, per-subject WITHIN-stage EER (paired)
    E, P, ks2 = paired(enc_within, psd_within); mdiff = P - E
    bdm = [np.mean(rng.choice(mdiff, len(mdiff), replace=True)) for _ in range(a.boot)]
    method = {"n_paired": len(ks2), "encoder_within_mean": round(float(E.mean()), 2),
              "cosine_within_mean": round(float(P.mean()), 2),
              "improvement_cosine_minus_encoder": round(float(mdiff.mean()), 2),
              "improvement_95CI_bootstrap": [round(float(np.percentile(bdm, 2.5)), 2), round(float(np.percentile(bdm, 97.5)), 2)]}
    if wilcoxon is not None and len(ks2) >= 6:
        try:
            st, p = wilcoxon(P, E, alternative="greater"); method["wilcoxon_cosine_gt_encoder_p"] = float(p)
        except Exception as e: method["wilcoxon_error"] = str(e)
    out = {"eval_n": len(ev), "fuse": a.fuse, "boot": a.boot,
           "gap_within_vs_cross_encoder": enc_gap, "gap_within_vs_cross_psd_cosine": psd_gap,
           "method_encoder_vs_cosine_within": method,
           "note": "Per-subject EERs (paired by subject). Wilcoxon signed-rank one-sided + paired bootstrap 95% CI of the difference + Cohen's d. Leakage-safe: EVAL subjects only, subject-disjoint impostors."}
    json.dump(out, open(d(a.root, "stats.json"), "w"), indent=2)
    log(f"ENCODER gap: within {enc_gap['within_mean']} vs cross {enc_gap['cross_mean']} | Δ {enc_gap['mean_diff_cross_minus_within']} CI{enc_gap['diff_95CI_bootstrap']} d={enc_gap['cohens_d']} p={enc_gap.get('wilcoxon_cross_gt_within_p')}")
    log(f"METHOD encoder {method['encoder_within_mean']} vs cosine {method['cosine_within_mean']} | p={method.get('wilcoxon_cosine_gt_encoder_p')}")
    log("DONE -> stats.json")

if __name__ == "__main__":
    sys.exit(main())
