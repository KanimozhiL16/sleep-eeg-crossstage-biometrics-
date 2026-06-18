#!/usr/bin/env python3
"""Export encoder 5x5 cross-stage EER matrix to enc_eer_fuse10.csv (for the main heatmap).
Reuses features2_enc/ (EVAL subjects) — cache-only, no GPU/download.
Run on Brev:  python export_enc_matrix.py --root ./"""
import argparse, glob, os, csv
import numpy as np
STAGES = ["W", "N1", "N2", "N3", "REM"]
def L2(X): return X/(np.linalg.norm(X, axis=1, keepdims=True)+1e-12)
def eer(g, i):
    s = np.concatenate([g, i]); l = np.concatenate([np.ones_like(g), np.zeros_like(i)])
    o = np.argsort(-s); l = l[o]; P, N = l.sum(), (1-l).sum()
    if P == 0 or N == 0: return np.nan
    far = np.cumsum(1-l)/N; frr = 1-np.cumsum(l)/P; k = np.nanargmin(np.abs(far-frr)); return float((far[k]+frr[k])/2*100)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="./"); ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--fuse", type=int, default=10); ap.add_argument("--min_ep", type=int, default=10)
    a = ap.parse_args()
    subs = sorted({os.path.basename(f)[:6] for f in glob.glob(os.path.join(a.root, "features2", "*.npz"))})
    rng = np.random.default_rng(a.seed); rng.shuffle(subs); ev = subs[int(round(len(subs)*a.train_frac)):]
    cache = {}
    for f in sorted(glob.glob(os.path.join(a.root, "features2_enc", "*.npz"))):
        s = os.path.basename(f)[:6]
        if s in ev: z = np.load(f, allow_pickle=True); cache[s] = (z["X"], z["y"])
    ids = sorted(cache); pools = {s: {st: cache[s][0][cache[s][1] == st] for st in STAGES} for s in ids}
    rng2 = np.random.default_rng(a.seed); M = np.full((5, 5), np.nan)
    for ei, A in enumerate(STAGES):
        templ = {s: pools[s][A].mean(0) for s in ids if len(pools[s][A]) >= a.min_ep}
        if len(templ) < 3: continue
        tid = list(templ); Tn = L2(np.stack([templ[s] for s in tid]))
        for pi, B in enumerate(STAGES):
            gen, imp = [], []
            for s in tid:
                Xb = pools[s][B]
                if len(Xb) < a.min_ep: continue
                fe = min(a.fuse, len(Xb)); nt = max(1, len(Xb)//fe); idx = rng2.permutation(len(Xb))[:nt*fe].reshape(nt, fe)
                Pn = L2(Xb[idx].mean(1)); sims = Pn @ Tn.T; ti = tid.index(s)
                gen.append(sims[:, ti]); imp.append(np.delete(sims, ti, 1).ravel())
            if gen: M[ei, pi] = eer(np.concatenate(gen), np.concatenate(imp))
    with open(os.path.join(a.root, "enc_eer_fuse10.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["enrol\\probe"]+STAGES)
        for i, st in enumerate(STAGES): w.writerow([st]+[f"{M[i,j]:.2f}" if not np.isnan(M[i, j]) else "NA" for j in range(5)])
    print("wrote enc_eer_fuse10.csv\n", M.round(1))
if __name__ == "__main__":
    main()
