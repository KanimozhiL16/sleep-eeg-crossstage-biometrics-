#!/usr/bin/env python3
"""
Wave-3a: DEMOGRAPHIC fairness. Does within-stage identifiability differ by age / sex?
Per-subject within-stage EER (encoder embeddings) vs age (median split) and sex (Mann-Whitney U).
Best-effort metadata: tries PhysioNet SC-subjects.xls; else accepts local sc_subjects.csv
(columns: subject,age,sex). Cache-only for EERs (no GPU/download for the EER part).

Run:  python demographics.py --root ./
"""
import argparse, glob, json, os
import numpy as np
STAGES = ["W", "N1", "N2", "N3", "REM"]
def d(r, *p): return os.path.join(r, *p)
def L2(X): return X/(np.linalg.norm(X, axis=1, keepdims=True)+1e-12)
def eer(g, i):
    s = np.concatenate([g, i]); l = np.concatenate([np.ones_like(g), np.zeros_like(i)]); o = np.argsort(-s); l = l[o]
    P, N = l.sum(), (1-l).sum()
    if P == 0 or N == 0: return np.nan
    far = np.cumsum(1-l)/N; frr = 1-np.cumsum(l)/P; k = np.nanargmin(np.abs(far-frr)); return float((far[k]+frr[k])/2*100)

def per_subject_within(cache, fuse=10, seed=42, min_ep=10):
    subs = sorted(cache); pools = {s: {st: cache[s][0][cache[s][1] == st] for st in STAGES} for s in subs}
    templ = {s: {st: pools[s][st].mean(0) for st in STAGES if len(pools[s][st]) >= min_ep} for s in subs}
    rng = np.random.default_rng(seed); out = {}
    for s in subs:
        gen, imp = [], []
        for S in STAGES:
            if S not in templ[s]: continue
            others = [o for o in subs if o != s and S in templ[o]]
            if not others: continue
            Tself = L2(templ[s][S][None])[0]; Toth = L2(np.stack([templ[o][S] for o in others]))
            Xb = pools[s][S]; fe = min(fuse, len(Xb)); nt = max(1, len(Xb)//fe)
            idx = rng.permutation(len(Xb))[:nt*fe].reshape(nt, fe); Pn = L2(Xb[idx].mean(1))
            gen.append(Pn@Tself); imp.append((Pn@Toth.T).ravel())
        if gen:
            e = eer(np.concatenate(gen), np.concatenate(imp))
            if not np.isnan(e): out[s] = e
    return out

def load_meta(root):
    # subject number -> (age, sex). Try local csv, then PhysioNet xls.
    p = d(root, "sc_subjects.csv")
    meta = {}
    if os.path.exists(p):
        import csv
        for r in csv.DictReader(open(p)):
            try: meta[int(r["subject"])] = (float(r["age"]), str(r["sex"]).strip())
            except Exception: pass
        return meta
    try:
        import pandas as pd
        url = "https://physionet.org/files/sleep-edfx/1.0.0/SC-subjects.xls"
        df = pd.read_excel(url)
        df.columns = [str(c).strip().lower() for c in df.columns]
        sc = [c for c in df.columns if "subject" in c][0]; ac = [c for c in df.columns if "age" in c][0]
        sx = [c for c in df.columns if c in ("sex", "gender") or "sex" in c][0]
        for _, row in df.iterrows():
            meta[int(row[sc])] = (float(row[ac]), str(row[sx]))
    except Exception as e:
        print("metadata load failed:", e, "-> provide sc_subjects.csv (subject,age,sex)")
    return meta

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="./"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cache = {}
    for f in sorted(glob.glob(d(a.root, "features2_enc", "*.npz"))):
        z = np.load(f, allow_pickle=True); cache[os.path.basename(f)[:6]] = (z["X"], z["y"])
    eers = per_subject_within(cache, seed=a.seed)
    meta = load_meta(a.root)
    rows = []
    for sid, e in eers.items():
        sn = int(sid[3:5])
        if sn in meta: rows.append((sid, e, meta[sn][0], meta[sn][1]))
    if not rows:
        json.dump({"error": "no demographic metadata mapped; provide sc_subjects.csv"}, open(d(a.root, "demographics.json"), "w"), indent=2)
        print("no metadata -> demographics.json error"); return
    E = np.array([r[1] for r in rows]); A = np.array([r[2] for r in rows]); S = np.array([str(r[3]) for r in rows])
    med = np.median(A); young = E[A <= med]; old = E[A > med]
    out = {"n": len(rows), "age_median": float(med),
           "EER_young_mean": round(float(np.mean(young)), 2), "EER_old_mean": round(float(np.mean(old)), 2)}
    # sex groups (handle 1/2 or M/F encodings)
    grp = {}
    for r in rows: grp.setdefault(str(r[3]), []).append(r[1])
    out["EER_by_sex_mean"] = {k: round(float(np.mean(v)), 2) for k, v in grp.items() if len(v) >= 3}
    try:
        from scipy.stats import mannwhitneyu
        if len(young) >= 3 and len(old) >= 3:
            out["age_MannWhitney_p"] = float(mannwhitneyu(young, old).pvalue)
        ks = [k for k in grp if len(grp[k]) >= 3]
        if len(ks) == 2: out["sex_MannWhitney_p"] = float(mannwhitneyu(grp[ks[0]], grp[ks[1]]).pvalue)
    except Exception as e:
        out["stats_note"] = f"scipy unavailable: {e}"
    out["interpretation"] = "Tests whether per-subject within-stage EER differs by age (median split) or sex. Non-significant p => no strong demographic bias (fairness)."
    json.dump(out, open(d(a.root, "demographics.json"), "w"), indent=2)
    print("DONE -> demographics.json", out)
if __name__ == "__main__":
    main()
