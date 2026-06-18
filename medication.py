#!/usr/bin/env python3
"""
Wave-3b: MEDICATION effect. Sleep-EDF Telemetry subset (temazepam vs placebo nights, 22 subjects).
Embed both nights per subject with the cassette-trained encoder.pt (cross-corpus transfer — CAVEAT),
then verify enrol-placebo / probe-temazepam (cross-drug) vs same-night. Does a hypnotic drug
change EEG identity? Telemetry subjects are fully held out (never in encoder training) -> no leakage.

CAVEATS (state in paper): cross-corpus encoder (trained on cassette); drug-night labelling via
ST-subjects metadata is best-effort; small n.

Run (Brev GPU):  CUDA_VISIBLE_DEVICES=0 python medication.py --root ./ 2>&1 | tee medication.log
"""
import argparse, glob, json, os, datetime, sys
import numpy as np
STAGES = ["W", "N1", "N2", "N3", "REM"]
ANNOT_MAP = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
             "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM"}
CH = ["Fpz-Cz", "Pz-Oz"]; ESEC = 30.0; LF, HF, REJ = 1.0, 45.0, 500e-6
def log(m): print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)
def d(r, *p): return os.path.join(r, *p)
def L2(X): return X/(np.linalg.norm(X, axis=1, keepdims=True)+1e-12)
def eer(g, i):
    s = np.concatenate([g, i]); l = np.concatenate([np.ones_like(g), np.zeros_like(i)]); o = np.argsort(-s); l = l[o]
    P, N = l.sum(), (1-l).sum()
    if P == 0 or N == 0: return np.nan
    far = np.cumsum(1-l)/N; frr = 1-np.cumsum(l)/P; k = np.nanargmin(np.abs(far-frr)); return float((far[k]+frr[k])/2*100)
def epochs2(psg, hyp):
    import mne
    raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR")
    pk = [m[0] for c in CH for m in [[x for x in raw.ch_names if c.lower() in x.lower()]] if m]
    if not pk: return None, None, None
    raw.pick(pk); sf = raw.info["sfreq"]; raw.filter(LF, HF, verbose="ERROR")
    raw.set_annotations(mne.read_annotations(hyp), emit_warning=False)
    eid = {k: i for i, k in enumerate(ANNOT_MAP)}
    ev, _ = mne.events_from_annotations(raw, event_id=eid, chunk_duration=ESEC, verbose="ERROR")
    if len(ev) == 0: return None, None, sf
    inv = {i: ANNOT_MAP[k] for k, i in eid.items()}
    ep = mne.Epochs(raw, ev, tmin=0., tmax=ESEC-1.0/sf, baseline=None, preload=True, verbose="ERROR")
    X = ep.get_data(); y = np.array([inv[e] for e in ep.events[:, 2]]); keep = np.abs(X).max(axis=(1, 2)) < REJ
    return X[keep].astype("float32"), y[keep], sf
def xstage(c1, c2, fuse, seed, min_ep=10):
    """enrol c1 (recA) -> probe c2 (recB), same-stage mean EER over common subjects."""
    common = sorted(set(c1) & set(c2)); rng = np.random.default_rng(seed)
    p1 = {s: {st: c1[s][0][c1[s][1] == st] for st in STAGES} for s in common}
    p2 = {s: {st: c2[s][0][c2[s][1] == st] for st in STAGES} for s in common}
    diag = []
    for S in STAGES:
        templ = {s: p1[s][S].mean(0) for s in common if len(p1[s][S]) >= min_ep}
        if len(templ) < 3: continue
        ids = list(templ); Tn = L2(np.stack([templ[s] for s in ids])); gen, imp = [], []
        for s in ids:
            Xb = p2[s][S]
            if len(Xb) < min_ep: continue
            fe = min(fuse, len(Xb)); nt = max(1, len(Xb)//fe); idx = rng.permutation(len(Xb))[:nt*fe].reshape(nt, fe)
            Pn = L2(Xb[idx].mean(1)); sims = Pn@Tn.T; ti = ids.index(s)
            gen.append(sims[:, ti]); imp.append(np.delete(sims, ti, 1).ravel())
        if gen: diag.append(eer(np.concatenate(gen), np.concatenate(imp)))
    return float(np.nanmean(diag)) if diag else np.nan
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="./"); ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--emb", type=int, default=128); ap.add_argument("--fuse", type=int, default=10); ap.add_argument("--n_subj", type=int, default=22)
    a = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F, mne
    np.random.seed(a.seed); torch.manual_seed(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    # z-norm stats from cassette TRAIN night-1 epochs2 (same as encoder training)
    subs = sorted({os.path.basename(f)[:6] for f in glob.glob(d(a.root, "features2", "*.npz"))})
    rng = np.random.default_rng(a.seed); rng.shuffle(subs); train = subs[:int(round(len(subs)*a.train_frac))]
    Xtr = np.concatenate([np.load(d(a.root, "epochs2", f"{s}.npz"), allow_pickle=True)["X"] for s in train if os.path.exists(d(a.root, "epochs2", f"{s}.npz"))])
    mu = Xtr.mean(axis=(0, 2), keepdims=True); sd = Xtr.std(axis=(0, 2), keepdims=True)+1e-6
    def nrm(X): return (X-mu)/sd
    C = Xtr.shape[1]
    class Enc(nn.Module):
        def __init__(s, C, e):
            super().__init__()
            def b(i, o, k=7, st=2): return nn.Sequential(nn.Conv1d(i, o, k, st, k//2), nn.BatchNorm1d(o), nn.ELU(), nn.Dropout(0.3))
            s.net = nn.Sequential(b(C, 32), b(32, 64), b(64, 128), b(128, 128), nn.AdaptiveAvgPool1d(1)); s.fc = nn.Linear(128, e)
        def forward(s, x): z = s.net(x).squeeze(-1); return F.normalize(s.fc(z), dim=1)
    enc = Enc(C, a.emb).to(dev); enc.load_state_dict(torch.load(d(a.root, "encoder.pt"), map_location=dev)); enc.eval()
    try:
        paths = mne.datasets.sleep_physionet.temazepam.fetch_data(subjects=list(range(a.n_subj)), path=d(a.root, "data_st"))
    except TypeError:
        paths = mne.datasets.sleep_physionet.temazepam.fetch_data(subjects=list(range(a.n_subj)))
    # group by subject; recordings per subject = [placebo, temazepam] (order per MNE)
    bys = {}
    for psg, hyp in paths:
        sn = os.path.basename(psg)[3:5]; bys.setdefault(sn, []).append((psg, hyp))
    recA, recB = {}, {}   # A=first recording, B=second
    with torch.no_grad():
        for sn, recs in bys.items():
            recs = sorted(recs)
            for ridx, (psg, hyp) in enumerate(recs[:2]):
                X, y, sf = epochs2(psg, hyp)
                if X is None: continue
                Z = np.concatenate([enc(torch.tensor(nrm(X[i:i+512])).to(dev)).cpu().numpy() for i in range(0, len(X), 512)])
                (recA if ridx == 0 else recB)[sn] = (Z, y)
    log(f"subjects with both telemetry nights: {len(set(recA)&set(recB))}")
    same = xstage(recA, recA, a.fuse, a.seed)          # within-recording (upper bound)
    cross = xstage(recA, recB, a.fuse, a.seed)          # enrol nightA -> verify nightB (cross-drug)
    out = {"n_subjects": len(set(recA) & set(recB)),
           "within_recording_sameStage_EER": round(same, 2),
           "cross_recording_sameStage_EER": round(cross, 2),
           "labels_note": "recA/recB = the two telemetry nights (one placebo, one temazepam). Map drug via ST-subjects.xls if exact placebo->temazepam direction needed.",
           "caveats": "cross-corpus encoder (trained on Sleep-Cassette); held-out subjects (no leakage); small n; supplementary result.",
           "interpretation": "If cross-recording EER >> within-recording, identity degrades across the two telemetry nights (one drugged) -> medication/another-night changes the brainprint."}
    json.dump(out, open(d(a.root, "medication.json"), "w"), indent=2)
    log(f"MEDICATION: within-rec {same:.2f} vs cross-rec {cross:.2f} -> medication.json  (delete data_st/ after)")
if __name__ == "__main__":
    sys.exit(main())
