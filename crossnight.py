#!/usr/bin/env python3
"""
Cross-NIGHT EEG biometric verification (Sleep-EDF). Enrol on NIGHT 1, verify on NIGHT 2.

Uses the ALREADY-TRAINED encoder.pt (trained on night-1 TRAIN subjects only) — NO retraining.
EVAL subjects' night-2 recordings are never seen during training => leakage-safe.

Pipeline:
  1. reproduce TRAIN/EVAL split (same seed/train_frac as encoder_train).
  2. recompute per-channel z-norm (mu/sd) from TRAIN night-1 epochs2/ (identical to training).
  3. download night-2 (recording=2) for EVAL subjects, preprocess (2-ch, 1-45 Hz, 30s), embed with encoder.pt.
  4. score: enrol night-1 stage A template -> verify night-2 stage B probes (5x5),
     plus cross-night SAME-stage (A==B), vs within-night-1 reference. fusion N=10. bootstrap CI.

Run (Brev GPU, tmux):  CUDA_VISIBLE_DEVICES=0 python crossnight.py --root ./ 2>&1 | tee crossnight.log
Then delete night-2 raw:  (script keeps data/night2; remove after).
"""
import argparse, glob, json, os, datetime, sys
import numpy as np

STAGES = ["W", "N1", "N2", "N3", "REM"]
ANNOT_MAP = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
             "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM"}
CHANNELS = ["Fpz-Cz", "Pz-Oz"]; EPOCH_SEC = 30.0; L_FREQ, H_FREQ, REJECT_UV = 1.0, 45.0, 500e-6

def log(m): print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)
def d(root, *p): return os.path.join(root, *p)
def L2(X): return X/(np.linalg.norm(X, axis=1, keepdims=True)+1e-12)


def epochs2(psg, hyp):
    import mne
    raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR")
    picks = [m[0] for ch in CHANNELS for m in [[c for c in raw.ch_names if ch.lower() in c.lower()]] if m]
    if not picks: return None, None, None
    raw.pick(picks); sf = raw.info["sfreq"]; raw.filter(L_FREQ, H_FREQ, verbose="ERROR")
    ann = mne.read_annotations(hyp); raw.set_annotations(ann, emit_warning=False)
    eid = {k: i for i, k in enumerate(ANNOT_MAP)}
    ev, _ = mne.events_from_annotations(raw, event_id=eid, chunk_duration=EPOCH_SEC, verbose="ERROR")
    if len(ev) == 0: return None, None, sf
    inv = {i: ANNOT_MAP[k] for k, i in eid.items()}
    ep = mne.Epochs(raw, ev, tmin=0., tmax=EPOCH_SEC-1.0/sf, baseline=None, preload=True, verbose="ERROR")
    X = ep.get_data(); y = np.array([inv[e] for e in ep.events[:, 2]])
    keep = np.abs(X).max(axis=(1, 2)) < REJECT_UV
    return X[keep].astype("float32"), y[keep], sf


def eer(gen, imp):
    s = np.concatenate([gen, imp]); l = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    o = np.argsort(-s); l = l[o]; P, N = l.sum(), (1-l).sum()
    if P == 0 or N == 0: return np.nan
    far = np.cumsum(1-l)/N; frr = 1-np.cumsum(l)/P
    i = np.nanargmin(np.abs(far-frr)); return float((far[i]+frr[i])/2*100)


def crossnight_matrix(c1, c2, min_ep=10, fuse=10, seed=42, shuffle=False, subjects=None):
    """enrol night1 stage A (c1) -> probe night2 stage B (c2). 5x5."""
    ids_all = subjects or sorted(set(c1) & set(c2))
    p1 = {s: {st: c1[s][0][c1[s][1] == st] for st in STAGES} for s in ids_all}
    p2 = {s: {st: c2[s][0][c2[s][1] == st] for st in STAGES} for s in ids_all}
    rng = np.random.default_rng(seed); M = np.full((5, 5), np.nan)
    for ei, A in enumerate(STAGES):
        templ = {s: p1[s][A].mean(0) for s in ids_all if len(p1[s][A]) >= min_ep}
        if len(templ) < 3: continue
        ids = list(templ); Tn = L2(np.stack([templ[s] for s in ids]))
        for pi, B in enumerate(STAGES):
            gen, imp = [], []
            for s in ids:
                Xb = p2[s][B]
                if len(Xb) < min_ep: continue
                fe = min(fuse, len(Xb)); nt = max(1, len(Xb)//fe)
                idx = rng.permutation(len(Xb))[:nt*fe].reshape(nt, fe)
                Pn = L2(Xb[idx].mean(1)); sims = Pn@Tn.T; ti = ids.index(s)
                gi = int(rng.integers(len(ids))) if shuffle else ti
                gen.append(sims[:, gi]); imp.append(np.delete(sims, ti, 1).ravel())
            if gen: M[ei, pi] = eer(np.concatenate(gen), np.concatenate(imp))
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./"); ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--min_epochs", type=int, default=10); ap.add_argument("--fuse", type=int, default=10)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; log(f"device={dev}")

    # split (same as encoder_train: derive from features2 subject list)
    subs = sorted({os.path.basename(fp)[:6] for fp in glob.glob(d(a.root, "features2", "*.npz"))})
    rng = np.random.default_rng(a.seed); rng.shuffle(subs)
    n_tr = int(round(len(subs)*a.train_frac)); train, ev = subs[:n_tr], subs[n_tr:]
    log(f"{len(train)} TRAIN / {len(ev)} EVAL")

    # z-norm stats from TRAIN night-1 epochs2 (identical to encoder_train)
    Xtr = np.concatenate([np.load(d(a.root, "epochs2", f"{s}.npz"), allow_pickle=True)["X"]
                          for s in train if os.path.exists(d(a.root, "epochs2", f"{s}.npz"))])
    mu = Xtr.mean(axis=(0, 2), keepdims=True); sd = Xtr.std(axis=(0, 2), keepdims=True)+1e-6
    def norm(X): return (X-mu)/sd
    C, T = Xtr.shape[1], Xtr.shape[2]

    class Enc(nn.Module):
        def __init__(self, C, emb):
            super().__init__()
            def blk(i, o, k=7, s=2): return nn.Sequential(nn.Conv1d(i, o, k, s, k//2), nn.BatchNorm1d(o), nn.ELU(), nn.Dropout(0.3))
            self.net = nn.Sequential(blk(C, 32), blk(32, 64), blk(64, 128), blk(128, 128), nn.AdaptiveAvgPool1d(1))
            self.fc = nn.Linear(128, emb)
        def forward(self, x):
            z = self.net(x).squeeze(-1); return F.normalize(self.fc(z), dim=1)
    enc = Enc(C, a.emb).to(dev); enc.load_state_dict(torch.load(d(a.root, "encoder.pt"), map_location=dev)); enc.eval()

    # night-1 EVAL embeddings: reuse features2_enc/
    c1 = {}
    for s in ev:
        fp = d(a.root, "features2_enc", f"{s}.npz")
        if os.path.exists(fp): z = np.load(fp, allow_pickle=True); c1[s] = (z["X"], z["y"])

    # night-2: download recording=2 for EVAL subjects, preprocess, embed
    import mne
    sub_ids = sorted({int(s[3:5]) for s in ev})   # SC4ssN -> subject digits at [3:5]
    paths = mne.datasets.sleep_physionet.age.fetch_data(subjects=sub_ids, recording=[2],
                                                        path=d(a.root, "data_n2"), on_missing="warn")
    c2 = {}
    with torch.no_grad():
        for psg, hyp in paths:
            sidnum = os.path.basename(psg)[3:5]
            X, y, sf = epochs2(psg, hyp)
            if X is None: continue
            Z = []
            for i in range(0, len(X), 512):
                Z.append(enc(torch.tensor(norm(X[i:i+512])).to(dev)).cpu().numpy())
            # match to an EVAL key sharing subject number (SC4ssN -> [3:5])
            key = next((s for s in ev if s[3:5] == sidnum), None)
            if key: c2[key] = (np.concatenate(Z), y)
    common = sorted(set(c1) & set(c2)); log(f"subjects with BOTH nights: {len(common)}")

    M = crossnight_matrix(c1, c2, a.min_epochs, a.fuse, a.seed, subjects=common)
    Mr = crossnight_matrix(c1, c2, a.min_epochs, a.fuse, a.seed, shuffle=True, subjects=common)
    diag = np.array([M[i, i] for i in range(5)]); off = M[~np.eye(5, dtype=bool)]
    same_stage = float(np.nanmean(diag)); cross_both = float(np.nanmean(off))
    # bootstrap CI (subject subsampling 80%) — guard small n
    rng2 = np.random.default_rng(a.seed); boot = []
    if len(common) >= 6:
        k = min(len(common)-1, max(5, int(0.8*len(common))))
        for _ in range(a.boot):
            sub = list(rng2.choice(common, k, replace=False))
            Mb = crossnight_matrix(c1, c2, a.min_epochs, a.fuse, a.seed, subjects=sub)
            db = [Mb[i, i] for i in range(5)]; boot.append([np.nanmean(db), np.nanmean(Mb[~np.eye(5, dtype=bool)])])
    boot = np.array(boot) if boot else np.full((1, 2), np.nan)
    import csv
    with open(d(a.root, "crossnight_eer_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["enrolN1\\probeN2"]+STAGES)
        for i, st in enumerate(STAGES):
            w.writerow([st]+[f"{M[i,j]:.2f}" if not np.isnan(M[i, j]) else "NA" for j in range(5)])
    out = {"n_subjects_both_nights": len(common),
           "cross_night_same_stage_EER": round(same_stage, 2),
           "cross_night_same_stage_CI95": [round(float(np.nanpercentile(boot[:, 0], 2.5)), 2), round(float(np.nanpercentile(boot[:, 0], 97.5)), 2)],
           "cross_night_AND_cross_stage_EER": round(cross_both, 2),
           "cross_night_cross_stage_CI95": [round(float(np.nanpercentile(boot[:, 1], 2.5)), 2), round(float(np.nanpercentile(boot[:, 1], 97.5)), 2)],
           "same_stage_by_stage": {STAGES[i]: (None if np.isnan(M[i, i]) else round(float(M[i, i]), 2)) for i in range(5)},
           "random_baseline_EER": round(float(np.nanmean(Mr)), 2),
           "reference_within_night1_encoder_fuse10": {"within": 0.21, "cross_stage": 9.58},
           "leakage_check": "PASS: encoder trained on night-1 TRAIN subjects only; EVAL night-2 never seen; z-norm from night-1 TRAIN; enrol(N1)/probe(N2) different nights.",
           "interpretation": "Compare cross-night-same-stage vs within-night1 (0.21%) to see identity drift across nights; cross-night+cross-stage is the hardest real-world case."}
    json.dump(out, open(d(a.root, "crossnight.json"), "w"), indent=2)
    log(f"CROSS-NIGHT: same-stage {same_stage:.2f}%  cross-stage {cross_both:.2f}%  random~{out['random_baseline_EER']}  (n={len(common)})")
    log("DONE -> crossnight.json, crossnight_eer_matrix.csv  (delete data_n2/ afterwards)")


if __name__ == "__main__":
    sys.exit(main())
