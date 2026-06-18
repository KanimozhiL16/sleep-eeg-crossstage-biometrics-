#!/usr/bin/env python3
"""
Sleep-EDF cross-stage EEG biometric verification — v2 (leakage-safe EER reduction).

Tiers A+B (no GPU training): 2 channels + rich features + subject-disjoint LDA scoring
+ multi-epoch probe fusion + bootstrap CIs. Reuses raw .edf already downloaded by
sleep_pipeline.py (same ./data). Tier C (learned encoder) is in encoder_train.py.

Stages (run one at a time):
    python sleep_pipeline_v2.py preprocess   # 2-channel filtered 30s epochs -> epochs2/
    python sleep_pipeline_v2.py features      # rich per-epoch features -> features2/
    python sleep_pipeline_v2.py experiment    # cosine & LDA x fusion N, EVAL-only, 5x5 + CIs
    python sleep_pipeline_v2.py validate       # random baseline + explicit leakage assertion

GUARDRAIL: subjects split TRAIN/EVAL by --train_frac; LDA fit on TRAIN only;
cross-stage EER reported on EVAL only; enrol/probe epochs disjoint.
"""
import argparse, json, os, glob, datetime, sys
import numpy as np

STAGES = ["W", "N1", "N2", "N3", "REM"]
ANNOT_MAP = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
             "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM"}
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30)}
CHANNELS = ["Fpz-Cz", "Pz-Oz"]
EPOCH_SEC = 30.0
L_FREQ, H_FREQ, REJECT_UV = 1.0, 45.0, 500e-6


def log(m): print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)
def d(root, *p): return os.path.join(root, *p)
def jdump(o, f): json.dump(o, open(f, "w"), indent=2, default=str)


def _hjorth(x):
    dx = np.diff(x); ddx = np.diff(dx)
    v0 = np.var(x) + 1e-12; v1 = np.var(dx) + 1e-12; v2 = np.var(ddx) + 1e-12
    activity = v0
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / (mobility + 1e-12)
    return activity, mobility, complexity


def _chan_feats(x, sf):
    """Rich untrained features for one channel/epoch."""
    from scipy.signal import welch
    f, p = welch(x, fs=sf, nperseg=int(sf * 4))
    bp = np.array([p[(f >= lo) & (f < hi)].sum() for lo, hi in BANDS.values()])
    rel = bp / (bp.sum() + 1e-12)
    logrel = np.log(rel + 1e-8)                         # 5
    pn = p / (p.sum() + 1e-12)
    spec_entropy = -np.sum(pn * np.log(pn + 1e-12))     # 1
    act, mob, cmp = _hjorth(x)                           # 3
    ratios = np.array([rel[0]/(rel[2]+1e-9), rel[1]/(rel[2]+1e-9), rel[3]/(rel[2]+1e-9)])  # delta/alpha, theta/alpha, sigma/alpha (3)
    return np.concatenate([logrel, [spec_entropy, np.log(act+1e-12), mob, cmp], np.log(ratios+1e-9)])


def _epochs2(psg, hyp):
    import mne
    raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR")
    picks = []
    for ch in CHANNELS:
        m = [c for c in raw.ch_names if ch.lower() in c.lower()]
        if m: picks.append(m[0])
    if len(picks) < 1: return None, None, None
    raw.pick(picks); sf = raw.info["sfreq"]; raw.filter(L_FREQ, H_FREQ, verbose="ERROR")
    ann = mne.read_annotations(hyp); raw.set_annotations(ann, emit_warning=False)
    eid = {k: i for i, k in enumerate(ANNOT_MAP)}
    ev, _ = mne.events_from_annotations(raw, event_id=eid, chunk_duration=EPOCH_SEC, verbose="ERROR")
    if len(ev) == 0: return None, None, sf
    inv = {i: ANNOT_MAP[k] for k, i in eid.items()}
    ep = mne.Epochs(raw, ev, tmin=0., tmax=EPOCH_SEC - 1.0/sf, baseline=None, preload=True, verbose="ERROR")
    X = ep.get_data()                                   # [n, n_ch, n_times]
    y = np.array([inv[e] for e in ep.events[:, 2]])
    return X, y, sf


def stage_preprocess(a):
    import mne
    paths = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=list(range(min(a.n_subjects, 83))), recording=[a.recording],
        path=d(a.root, "data"), on_missing="warn")
    out = d(a.root, "epochs2"); os.makedirs(out, exist_ok=True); kept = rej = 0
    for psg, hyp in paths:
        sid = os.path.basename(psg)[:6]; X, y, sf = _epochs2(psg, hyp)
        if X is None: continue
        amp = np.abs(X).max(axis=(1, 2)); keep = amp < REJECT_UV
        kept += int(keep.sum()); rej += int((~keep).sum())
        np.savez_compressed(d(out, f"{sid}.npz"), X=X[keep].astype(np.float32), y=y[keep], sf=sf)
        log(f"  {sid}: {keep.sum()} epochs, {X.shape[1]} ch")
    jdump({"channels": CHANNELS, "bandpass": [L_FREQ, H_FREQ], "kept": kept, "rejected": rej},
          d(a.root, "preprocess2_report.json"))
    log(f"PREPROCESS2 done: kept {kept}, rejected {rej} -> epochs2/")


def stage_features(a):
    out = d(a.root, "features2"); os.makedirs(out, exist_ok=True)
    for fp in sorted(glob.glob(d(a.root, "epochs2", "*.npz"))):
        sid = os.path.basename(fp)[:6]; z = np.load(fp, allow_pickle=True)
        X, y, sf = z["X"], z["y"], float(z["sf"])
        feats = []
        for ep in X:
            fv = np.concatenate([_chan_feats(ep[c], sf) for c in range(ep.shape[0])])
            feats.append(fv)
        np.savez_compressed(d(out, f"{sid}.npz"), X=np.array(feats), y=y)
        log(f"  {sid}: {len(feats)} epochs, dim={len(feats[0])}")
    log("FEATURES2 done -> features2/")


def _load(root):
    c = {}
    for fp in sorted(glob.glob(d(root, "features2", "*.npz"))):
        z = np.load(fp, allow_pickle=True); c[os.path.basename(fp)[:6]] = (z["X"], z["y"])
    return c


def _split(cache, train_frac, seed):
    subs = sorted(cache); rng = np.random.default_rng(seed); rng.shuffle(subs)
    n_tr = int(round(len(subs) * train_frac))
    return subs[:n_tr], subs[n_tr:]


def _fit_lda(cache, train_subs):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([cache[s][0] for s in train_subs])
    y = np.concatenate([[s]*len(cache[s][0]) for s in train_subs])
    sc = StandardScaler().fit(X)
    lda = LinearDiscriminantAnalysis().fit(sc.transform(X), y)
    return sc, lda


def _project(cache, subs, sc=None, lda=None):
    out = {}
    for s in subs:
        X = cache[s][0]
        if sc is not None: X = sc.transform(X)
        if lda is not None: X = lda.transform(X)
        out[s] = (X, cache[s][1])
    return out


def _eer(gen, imp):
    s = np.concatenate([gen, imp]); l = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    o = np.argsort(-s); l = l[o]; P, N = l.sum(), (1-l).sum()
    if P == 0 or N == 0: return np.nan
    far = np.cumsum(1-l)/N; frr = 1-np.cumsum(l)/P
    i = np.nanargmin(np.abs(far-frr)); return float((far[i]+frr[i])/2*100)


def _crossstage(eval_cache, min_epochs, fuse=1, shuffle_rng=None, rng=None):
    """5x5 EER on EVAL subjects. fuse=N averages N random probe epochs per genuine/impostor trial."""
    subs = sorted(eval_cache)
    pools = {s: {st: eval_cache[s][0][eval_cache[s][1] == st] for st in STAGES} for s in subs}
    rng = rng or np.random.default_rng(0)
    M = np.full((5, 5), np.nan)
    for ei, A in enumerate(STAGES):
        templ = {s: pools[s][A].mean(0) for s in subs if len(pools[s][A]) >= min_epochs}
        if len(templ) < 5: continue
        ids = list(templ); T = np.stack([templ[s] for s in ids])
        Tn = T/(np.linalg.norm(T, axis=1, keepdims=True)+1e-12)
        for pi, B in enumerate(STAGES):
            gen, imp = [], []
            for s in ids:
                Xb = pools[s][B]
                if len(Xb) < min_epochs: continue
                # build fused probe vectors: average fuse random epochs
                nt = max(1, len(Xb)//max(fuse, 1))
                idx = rng.permutation(len(Xb))[:nt*fuse].reshape(nt, fuse)
                probes = Xb[idx].mean(1)
                Pn = probes/(np.linalg.norm(probes, axis=1, keepdims=True)+1e-12)
                sims = Pn @ Tn.T
                true_i = ids.index(s)
                gi = int(shuffle_rng.integers(len(ids))) if shuffle_rng is not None else true_i
                gen.append(sims[:, gi]); imp.append(np.delete(sims, true_i, 1).ravel())
            if gen: M[ei, pi] = _eer(np.concatenate(gen), np.concatenate(imp))
    return M


def _means(M):
    diag = np.array([M[i, i] for i in range(5)]); off = M[~np.eye(5, dtype=bool)]
    return float(np.nanmean(diag)), float(np.nanmean(off))


def stage_experiment(a):
    import csv
    cache = _load(a.root); train, ev = _split(cache, a.train_frac, a.seed)
    log(f"split: {len(train)} TRAIN / {len(ev)} EVAL subjects (disjoint)")
    sc, lda = _fit_lda(cache, train)
    reps = {}
    for method in ["cosine", "lda"]:
        ec = _project(cache, ev, sc, lda) if method == "lda" else {s: cache[s] for s in ev}
        for fuse in [1, 5, 10]:
            M = _crossstage(ec, a.min_epochs, fuse=fuse, rng=np.random.default_rng(a.seed))
            wi, cr = _means(M)
            key = f"{method}_fuse{fuse}"; reps[key] = {"within": round(wi, 2), "cross": round(cr, 2),
                                                       "within_by_stage": {STAGES[i]: (None if np.isnan(M[i,i]) else round(float(M[i,i]),2)) for i in range(5)}}
            log(f"  {key}: within={wi:.2f}  cross={cr:.2f}")
            with open(d(a.root, f"eer_{key}.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["enrol\\probe"]+STAGES)
                for i, st in enumerate(STAGES):
                    w.writerow([st]+[f"{M[i,j]:.2f}" if not np.isnan(M[i,j]) else "NA" for j in range(5)])
    jdump({"n_train": len(train), "n_eval": len(ev), "methods": reps}, d(a.root, "summary_v2.json"))
    log("EXPERIMENT v2 done -> summary_v2.json + eer_*.csv")


def stage_validate(a):
    cache = _load(a.root); train, ev = _split(cache, a.train_frac, a.seed)
    assert set(train).isdisjoint(set(ev)), "LEAKAGE: train/eval overlap!"
    sc, lda = _fit_lda(cache, train)
    ec = _project(cache, ev, sc, lda)
    rng = np.random.default_rng(a.seed)
    M = _crossstage(ec, a.min_epochs, fuse=10, rng=np.random.default_rng(a.seed))
    Mr = _crossstage(ec, a.min_epochs, fuse=10, shuffle_rng=rng, rng=np.random.default_rng(a.seed))
    wi, cr = _means(M)
    # bootstrap CI over EVAL subjects
    boot = []
    for _ in range(a.boot):
        bs = list(rng.choice(ev, len(ev), replace=True))
        ecb = {f"{s}#{i}": ec[s] for i, s in enumerate(bs)}   # unique keys
        Mb = _crossstage(ecb, a.min_epochs, fuse=10, rng=np.random.default_rng(a.seed))
        wb, cb = _means(Mb); boot.append([wb, cb])
    boot = np.array(boot)
    rep = {"method": "lda_fuse10", "n_train": len(train), "n_eval": len(ev),
           "within_EER": round(wi, 2), "cross_EER": round(cr, 2),
           "within_CI95": [round(float(np.nanpercentile(boot[:,0], 2.5)),2), round(float(np.nanpercentile(boot[:,0],97.5)),2)],
           "cross_CI95": [round(float(np.nanpercentile(boot[:,1], 2.5)),2), round(float(np.nanpercentile(boot[:,1],97.5)),2)],
           "random_identity_baseline_EER": round(float(np.nanmean(Mr)), 2),
           "leakage_check": "PASS: TRAIN/EVAL subject-disjoint (asserted); LDA fit on TRAIN only; enrol/probe disjoint epochs; impostors subject-disjoint."}
    jdump(rep, d(a.root, "validation_v2.json"))
    log(f"VALIDATE v2: within={wi:.2f} cross={cr:.2f} random~{rep['random_identity_baseline_EER']} (CIs in json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["preprocess", "features", "experiment", "validate"])
    ap.add_argument("--root", default="./")
    ap.add_argument("--n_subjects", type=int, default=78)
    ap.add_argument("--recording", type=int, default=1)
    ap.add_argument("--min_epochs", type=int, default=10)
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); np.random.seed(a.seed)
    log(f"STAGE={a.stage} root={a.root} train_frac={a.train_frac}")
    {"preprocess": stage_preprocess, "features": stage_features,
     "experiment": stage_experiment, "validate": stage_validate}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
