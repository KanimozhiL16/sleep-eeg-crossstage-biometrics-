#!/usr/bin/env python3
"""
Wave-2: RE-ENROLMENT simulation. After night-to-night drift, how much fresh (night-2) data
restores accuracy? Template = night-1 enrol + fraction f of night-2 epochs; probe = held-out
night-2 epochs. EER vs f (same-stage). Reuses encoder.pt + data_n2 (no new download).

Leakage-safe: encoder trained on night-1 TRAIN only; EVAL subjects; night-2 enrol-add epochs
are DISJOINT from night-2 probe epochs; impostors subject-disjoint.

Run (Brev GPU):  CUDA_VISIBLE_DEVICES=0 python reenroll.py --root ./ 2>&1 | tee reenroll.log
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
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="./"); ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--emb", type=int, default=128); ap.add_argument("--fuse", type=int, default=10); ap.add_argument("--min_ep", type=int, default=10)
    a = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    np.random.seed(a.seed); torch.manual_seed(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    subs = sorted({os.path.basename(f)[:6] for f in glob.glob(d(a.root, "features2", "*.npz"))})
    rng = np.random.default_rng(a.seed); rng.shuffle(subs); n = int(round(len(subs)*a.train_frac)); train, ev = subs[:n], subs[n:]
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
    # night-1 EVAL embeddings
    c1 = {}
    for s in ev:
        fp = d(a.root, "features2_enc", f"{s}.npz")
        if os.path.exists(fp): z = np.load(fp, allow_pickle=True); c1[s] = (z["X"], z["y"])
    # night-2 embeddings from data_n2
    import mne
    paths = mne.datasets.sleep_physionet.age.fetch_data(subjects=sorted({int(s[3:5]) for s in ev}), recording=[2], path=d(a.root, "data_n2"), on_missing="warn")
    c2 = {}
    with torch.no_grad():
        for psg, hyp in paths:
            sn = os.path.basename(psg)[3:5]; X, y, sf = epochs2(psg, hyp)
            if X is None: continue
            Z = [enc(torch.tensor(nrm(X[i:i+512])).to(dev)).cpu().numpy() for i in range(0, len(X), 512)]
            key = next((s for s in ev if s[3:5] == sn), None)
            if key: c2[key] = (np.concatenate(Z), y)
    common = sorted(set(c1) & set(c2)); log(f"both-night EVAL subjects: {len(common)}")
    p1 = {s: {st: c1[s][0][c1[s][1] == st] for st in STAGES} for s in common}
    p2 = {s: {st: c2[s][0][c2[s][1] == st] for st in STAGES} for s in common}
    rng2 = np.random.default_rng(a.seed); curve = {}
    for f in [0.0, 0.05, 0.1, 0.25, 0.5]:
        diag = []
        for S in STAGES:
            # split night-2 stage-S epochs into enrol-add (f) and probe (rest)
            templ = {}; probes = {}
            for s in common:
                n1 = p1[s][S]; n2 = p2[s][S]
                if len(n1) < a.min_ep or len(n2) < a.min_ep+5: continue
                perm = rng2.permutation(len(n2)); k = int(f*len(n2)); add, prb = n2[perm[:k]], n2[perm[k:]]
                if len(prb) < a.min_ep: continue
                templ[s] = np.concatenate([n1, add]).mean(0); probes[s] = prb
            if len(templ) < 3: continue
            ids = list(templ); Tn = L2(np.stack([templ[s] for s in ids])); gen, imp = [], []
            for s in ids:
                Xb = probes[s]; fe = min(a.fuse, len(Xb)); nt = max(1, len(Xb)//fe)
                idx = rng2.permutation(len(Xb))[:nt*fe].reshape(nt, fe); Pn = L2(Xb[idx].mean(1)); sims = Pn@Tn.T; ti = ids.index(s)
                gen.append(sims[:, ti]); imp.append(np.delete(sims, ti, 1).ravel())
            if gen: diag.append(eer(np.concatenate(gen), np.concatenate(imp)))
        curve[f] = round(float(np.nanmean(diag)), 2); log(f"  re-enrol f={f:.2f} ({int(f*100)}% night-2): same-stage EER {curve[f]}")
    json.dump({"n_subjects": len(common), "reenroll_fraction_to_sameStageEER": curve,
               "baseline_f0_is_night1only": curve.get(0.0),
               "interpretation": "f=0 -> night-1-only template (cross-night drift). As f grows (more night-2 data folded into template), EER should drop -> quantifies how much re-enrolment restores accuracy after night drift."},
              open(d(a.root, "reenroll.json"), "w"), indent=2)
    log("DONE -> reenroll.json")
if __name__ == "__main__":
    sys.exit(main())
