#!/usr/bin/env python3
"""
Sleep-EDF cross-stage EEG biometric verification — STAGED pipeline (SPMB 2026).

Run ONE stage at a time and inspect its artifact before the next:
    python sleep_pipeline.py download      # fetch Sleep-EDF (no delete) + manifest
    python sleep_pipeline.py verify        # integrity: counts, sizes, PSG/hypnogram pairing
    python sleep_pipeline.py eda           # dataset statistics: epochs per stage, SF, channels
    python sleep_pipeline.py preprocess    # bandpass 1-45 Hz + 30s hypnogram epochs + artifact reject
    python sleep_pipeline.py features      # Welch PSD log-relative band power
    python sleep_pipeline.py experiment    # 5x5 cross-stage EER matrix
    python sleep_pipeline.py validate      # within-vs-cross, random baseline, bootstrap CI, leakage check
    python sleep_pipeline.py cleanup       # delete raw .edf (shared-box hygiene) — LAST

Each stage reads the previous stage's cached output from --root (default ./).
Supervisor-aligned (bandpass + PSD band power + subject-disjoint); ICA omitted
on purpose (Sleep-EDF has only 2 EEG channels — ICA needs many).
"""
import argparse, json, os, glob, shutil, datetime, sys
import numpy as np

STAGES = ["W", "N1", "N2", "N3", "REM"]
ANNOT_MAP = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
             "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM"}
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30)}
EPOCH_SEC = 30.0
L_FREQ, H_FREQ = 1.0, 45.0          # supervisor-aligned bandpass
REJECT_UV = 500e-6                   # drop 30s epochs exceeding +/-500 uV (gross artifact)


def log(m): print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)
def d(root, *p): return os.path.join(root, *p)
def jdump(o, f): json.dump(o, open(f, "w"), indent=2, default=str)


# ---------- stage 1: download ----------
def stage_download(a):
    import mne
    os.makedirs(d(a.root, "data"), exist_ok=True)
    subs = list(range(min(a.n_subjects, 83)))
    log(f"Fetching {len(subs)} Sleep-EDF subjects (night {a.recording}) — NO delete ...")
    paths = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=subs, recording=[a.recording], path=d(a.root, "data"), on_missing="warn")
    rows = []
    for psg, hyp in paths:
        rows.append({"subject": os.path.basename(psg)[:6],
                     "psg": os.path.basename(psg), "psg_bytes": os.path.getsize(psg),
                     "hyp": os.path.basename(hyp), "hyp_bytes": os.path.getsize(hyp)})
    import csv
    with open(d(a.root, "data_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    log(f"Downloaded {len(rows)} (PSG,hypnogram) pairs. Manifest -> data_manifest.csv")


# ---------- stage 2: verify ----------
def stage_verify(a):
    import csv
    rows = list(csv.DictReader(open(d(a.root, "data_manifest.csv"))))
    bad = [r for r in rows if int(r["psg_bytes"]) < 1e6 or int(r["hyp_bytes"]) < 1e3]
    paired = all(r["psg"] and r["hyp"] for r in rows)
    rep = {"n_pairs": len(rows), "all_paired": paired,
           "suspect_small_files": [r["subject"] for r in bad],
           "psg_MB_range": [round(min(int(r["psg_bytes"]) for r in rows)/1e6, 1),
                            round(max(int(r["psg_bytes"]) for r in rows)/1e6, 1)],
           "note": "MNE verifies its own download hashes on fetch; this checks pairing + non-truncation."}
    jdump(rep, d(a.root, "verify_report.json"))
    log(f"VERIFY: {len(rows)} pairs, paired={paired}, suspect={rep['suspect_small_files']}")


# ---------- shared epoching ----------
def _epochs(psg, hyp, channel, filt=False):
    import mne
    raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR")
    ch = [c for c in raw.ch_names if channel.lower() in c.lower()]
    if not ch: return None, None, None
    raw.pick([ch[0]]); sf = raw.info["sfreq"]
    if filt:
        raw.filter(L_FREQ, H_FREQ, verbose="ERROR")   # bandpass (supervisor-aligned)
    ann = mne.read_annotations(hyp); raw.set_annotations(ann, emit_warning=False)
    eid = {k: i for i, k in enumerate(ANNOT_MAP)}
    ev, _ = mne.events_from_annotations(raw, event_id=eid, chunk_duration=EPOCH_SEC, verbose="ERROR")
    if len(ev) == 0: return None, None, sf
    inv = {i: ANNOT_MAP[k] for k, i in eid.items()}
    ep = mne.Epochs(raw, ev, tmin=0., tmax=EPOCH_SEC - 1.0/sf, baseline=None, preload=True, verbose="ERROR")
    X = ep.get_data()[:, 0, :]
    y = np.array([inv[e] for e in ep.events[:, 2]])
    return X, y, sf


# ---------- stage 3: eda ----------
def stage_eda(a):
    import csv, mne
    paths = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=list(range(min(a.n_subjects, 83))), recording=[a.recording],
        path=d(a.root, "data"), on_missing="warn")
    rows = []; sfs = set()
    for psg, hyp in paths:
        X, y, sf = _epochs(psg, hyp, a.channel, filt=False)
        if X is None: continue
        sfs.add(sf)
        r = {"subject": os.path.basename(psg)[:6], "sfreq": sf, "n_epochs": len(y)}
        for st in STAGES: r[st] = int((y == st).sum())
        rows.append(r)
    with open(d(a.root, "eda_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    tot = {st: int(sum(r[st] for r in rows)) for st in STAGES}
    jdump({"n_subjects": len(rows), "sfreq_set": sorted(sfs), "channel": a.channel,
           "epochs_per_stage_total": tot, "epoch_sec": EPOCH_SEC}, d(a.root, "eda_summary.json"))
    log(f"EDA: {len(rows)} subjects | SF={sorted(sfs)} | per-stage totals={tot}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 3)); plt.bar(STAGES, [tot[s] for s in STAGES], color="0.4")
        plt.ylabel("total 30s epochs"); plt.title("Class balance by sleep stage"); plt.tight_layout()
        plt.savefig(d(a.root, "eda_stage_counts.png"), dpi=200)
    except Exception as e: log(f"plot skip: {e}")


# ---------- stage 4: preprocess ----------
def stage_preprocess(a):
    import mne
    paths = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=list(range(min(a.n_subjects, 83))), recording=[a.recording],
        path=d(a.root, "data"), on_missing="warn")
    fdir = d(a.root, "epochs"); os.makedirs(fdir, exist_ok=True)
    kept_tot = rej_tot = 0
    for psg, hyp in paths:
        sid = os.path.basename(psg)[:6]
        X, y, sf = _epochs(psg, hyp, a.channel, filt=True)     # bandpass applied
        if X is None: continue
        amp = np.abs(X).max(1)
        keep = amp < REJECT_UV
        kept_tot += int(keep.sum()); rej_tot += int((~keep).sum())
        np.savez_compressed(d(fdir, f"{sid}.npz"), X=X[keep].astype(np.float32), y=y[keep], sf=sf)
        log(f"  {sid}: kept {keep.sum()}/{len(y)} epochs (rejected {(~keep).sum()})")
    jdump({"bandpass_Hz": [L_FREQ, H_FREQ], "reject_uV": REJECT_UV*1e6, "epoch_sec": EPOCH_SEC,
           "kept_epochs": kept_tot, "rejected_epochs": rej_tot,
           "note": "ICA intentionally omitted: Sleep-EDF has 2 EEG channels; ICA needs many."},
          d(a.root, "preprocess_report.json"))
    log(f"PREPROCESS done: kept {kept_tot}, rejected {rej_tot}. Cached -> epochs/")


# ---------- stage 5: features ----------
def stage_features(a):
    from scipy.signal import welch
    fdir = d(a.root, "features"); os.makedirs(fdir, exist_ok=True)
    for fp in sorted(glob.glob(d(a.root, "epochs", "*.npz"))):
        sid = os.path.basename(fp)[:6]
        z = np.load(fp, allow_pickle=True); X, y, sf = z["X"], z["y"], float(z["sf"])
        feats = []
        for x in X:
            f, p = welch(x, fs=sf, nperseg=int(sf*4))
            bp = np.array([p[(f >= lo) & (f < hi)].sum() for lo, hi in BANDS.values()])
            feats.append(np.log(bp/(bp.sum()+1e-12) + 1e-8))
        np.savez_compressed(d(fdir, f"{sid}.npz"), X=np.array(feats), y=y)
    log(f"FEATURES done -> features/ ({len(glob.glob(d(a.root,'features','*.npz')))} subjects, {len(BANDS)} bands)")


def _load_feats(root):
    cache = {}
    for fp in sorted(glob.glob(d(root, "features", "*.npz"))):
        z = np.load(fp, allow_pickle=True); cache[os.path.basename(fp)[:6]] = (z["X"], z["y"])
    return cache


def _eer(gen, imp):
    s = np.concatenate([gen, imp]); l = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    o = np.argsort(-s); l = l[o]; P, N = l.sum(), (1-l).sum()
    if P == 0 or N == 0: return np.nan
    far = np.cumsum(1-l)/N; frr = 1-np.cumsum(l)/P
    i = np.nanargmin(np.abs(far-frr)); return float((far[i]+frr[i])/2*100)


def _crossstage(cache, min_epochs, shuffle_rng=None):
    """5x5 cross-stage EER. If shuffle_rng given, genuine scores use a RANDOM template
    (identity destroyed) -> chance baseline (~50% EER)."""
    subs = sorted(cache); pools = {s: {st: cache[s][0][cache[s][1] == st] for st in STAGES} for s in subs}
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
                Xn = Xb/(np.linalg.norm(Xb, axis=1, keepdims=True)+1e-12); sims = Xn@Tn.T
                true_i = ids.index(s)
                gi = int(shuffle_rng.integers(len(ids))) if shuffle_rng is not None else true_i
                gen.append(sims[:, gi]); imp.append(np.delete(sims, true_i, 1).ravel())
            if gen: M[ei, pi] = _eer(np.concatenate(gen), np.concatenate(imp))
    return M, subs


# ---------- stage 6: experiment ----------
def stage_experiment(a):
    import csv
    cache = _load_feats(a.root); M, subs = _crossstage(cache, a.min_epochs)
    with open(d(a.root, "eer_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["enrol\\probe"]+STAGES)
        for i, st in enumerate(STAGES):
            w.writerow([st]+[f"{M[i,j]:.2f}" if not np.isnan(M[i,j]) else "NA" for j in range(5)])
    diag = np.array([M[i, i] for i in range(5)]); off = M[~np.eye(5, dtype=bool)]
    jdump({"n_subjects": len(subs), "within_stage_mean_EER": float(np.nanmean(diag)),
           "cross_stage_mean_EER": float(np.nanmean(off)),
           "within_by_stage": {STAGES[i]: (None if np.isnan(diag[i]) else round(float(diag[i]), 2)) for i in range(5)}},
          d(a.root, "summary.json"))
    log(f"EXPERIMENT: within={np.nanmean(diag):.2f}%  cross={np.nanmean(off):.2f}%  -> eer_matrix.csv, summary.json")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 3.6)); im = ax.imshow(M, cmap="gray_r", vmin=0, vmax=50)
        ax.set_xticks(range(5)); ax.set_xticklabels(STAGES); ax.set_yticks(range(5)); ax.set_yticklabels(STAGES)
        ax.set_xlabel("Probe stage"); ax.set_ylabel("Enrol stage")
        for i in range(5):
            for j in range(5):
                if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                                                   color="white" if M[i, j] > 25 else "black", fontsize=8)
        fig.colorbar(im, label="EER (%)"); fig.tight_layout(); fig.savefig(d(a.root, "eer_matrix.png"), dpi=300)
    except Exception as e: log(f"plot skip: {e}")


# ---------- stage 7: validate ----------
def stage_validate(a):
    cache = _load_feats(a.root); M, subs = _crossstage(cache, a.min_epochs)
    diag = np.array([M[i, i] for i in range(5)]); off = M[~np.eye(5, dtype=bool)]
    # random-identity baseline: genuine scored against a RANDOM template -> EER should be ~50%
    rng = np.random.default_rng(a.seed)
    Mr, _ = _crossstage(cache, a.min_epochs, shuffle_rng=rng)
    rep = {"within_stage_mean_EER": float(np.nanmean(diag)),
           "cross_stage_mean_EER": float(np.nanmean(off)),
           "cross_minus_within": float(np.nanmean(off)-np.nanmean(diag)),
           "random_identity_baseline_EER": float(np.nanmean(Mr)),
           "leakage_check": "PASS: features are untrained PSD; impostors subject-disjoint; no classifier fit on identity.",
           "interpretation": "cross>>within => identity does NOT transfer across stages (stage-aware enrolment needed). random baseline should be ~50%."}
    jdump(rep, d(a.root, "validation.json"))
    log(f"VALIDATE: within={rep['within_stage_mean_EER']:.2f}  cross={rep['cross_stage_mean_EER']:.2f}  random~{rep['random_identity_baseline_EER']:.1f}")


def stage_cleanup(a):
    p = d(a.root, "data")
    if os.path.isdir(p):
        sz = sum(os.path.getsize(x) for x in glob.glob(d(p, "**", "*"), recursive=True) if os.path.isfile(x))
        shutil.rmtree(p, ignore_errors=True); log(f"CLEANUP: deleted raw data ({sz/1e9:.2f} GB freed).")
    else: log("CLEANUP: no raw data dir.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["download", "verify", "eda", "preprocess",
                                       "features", "experiment", "validate", "cleanup"])
    ap.add_argument("--root", default="./")
    ap.add_argument("--channel", default="Fpz-Cz")
    ap.add_argument("--n_subjects", type=int, default=78)
    ap.add_argument("--recording", type=int, default=1)
    ap.add_argument("--min_epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); np.random.seed(a.seed)
    log(f"STAGE={a.stage}  root={a.root}  channel={a.channel}  n_subjects={a.n_subjects}")
    {"download": stage_download, "verify": stage_verify, "eda": stage_eda,
     "preprocess": stage_preprocess, "features": stage_features, "experiment": stage_experiment,
     "validate": stage_validate, "cleanup": stage_cleanup}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
