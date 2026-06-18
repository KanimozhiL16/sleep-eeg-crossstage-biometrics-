#!/usr/bin/env python3
"""
Tier C — learned EEG embedding encoder (subject-disjoint, GPU/PyTorch).

Trains a compact 1D-CNN with an ArcFace head to classify TRAIN subjects from 2-channel
30s epochs (epochs2/ from sleep_pipeline_v2.py preprocess), then extracts L2-normalised
embeddings for EVAL subjects and runs the SAME cross-stage EER (cosine + multi-epoch
fusion + bootstrap CI) reusing sleep_pipeline_v2's scoring.

LEAKAGE-SAFE: encoder is trained ONLY on TRAIN subjects; cross-stage EER reported ONLY on
EVAL subjects (disjoint). Identity labels never seen for EVAL subjects.

Run (on Brev GPU, inside tmux):
    CUDA_VISIBLE_DEVICES=0 python encoder_train.py --root ./ --epochs 40 2>&1 | tee enc.log
"""
import argparse, glob, json, os, datetime, sys
import numpy as np

def log(m): print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)
def d(root, *p): return os.path.join(root, *p)


def load_epochs(root):
    """Return dict sid -> (X[n,ch,T] float32, y_stage[n]) from epochs2/."""
    c = {}
    for fp in sorted(glob.glob(d(root, "epochs2", "*.npz"))):
        z = np.load(fp, allow_pickle=True); c[os.path.basename(fp)[:6]] = (z["X"].astype("float32"), z["y"])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./")
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_epochs", type=int, default=10)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; log(f"device={dev}")

    cache = load_epochs(a.root)
    subs = sorted(cache); rng = np.random.default_rng(a.seed); rng.shuffle(subs)
    n_tr = int(round(len(subs)*a.train_frac)); train, ev = subs[:n_tr], subs[n_tr:]
    assert set(train).isdisjoint(ev), "LEAKAGE"
    log(f"{len(train)} TRAIN / {len(ev)} EVAL subjects (disjoint)")

    # per-channel z-norm using TRAIN stats only (no leakage)
    Xtr = np.concatenate([cache[s][0] for s in train])
    mu = Xtr.mean(axis=(0, 2), keepdims=True); sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    def norm(X): return (X - mu) / sd
    sub2idx = {s: i for i, s in enumerate(train)}
    Xtr_t = torch.tensor(norm(Xtr)); ytr_t = torch.tensor(np.concatenate([[sub2idx[s]]*len(cache[s][0]) for s in train]))

    C = Xtr.shape[1]; T = Xtr.shape[2]

    class Enc(nn.Module):
        def __init__(self, C, emb):
            super().__init__()
            def blk(i, o, k=7, s=2): return nn.Sequential(nn.Conv1d(i, o, k, s, k//2), nn.BatchNorm1d(o), nn.ELU(), nn.Dropout(0.3))
            self.net = nn.Sequential(blk(C, 32), blk(32, 64), blk(64, 128), blk(128, 128), nn.AdaptiveAvgPool1d(1))
            self.fc = nn.Linear(128, emb)
        def forward(self, x):
            z = self.net(x).squeeze(-1); return F.normalize(self.fc(z), dim=1)

    class ArcFace(nn.Module):
        def __init__(self, emb, n, s=30.0, m=0.30):
            super().__init__(); self.W = nn.Parameter(torch.randn(n, emb)); self.s, self.m = s, m
        def forward(self, z, y):
            W = F.normalize(self.W, dim=1); cos = z @ W.t()
            th = torch.acos(torch.clamp(cos, -1+1e-7, 1-1e-7))
            target = torch.cos(th + self.m)
            oh = F.one_hot(y, cos.size(1)).float()
            return self.s * (oh*target + (1-oh)*cos)

    enc = Enc(C, a.emb).to(dev); head = ArcFace(a.emb, len(train)).to(dev)
    opt = torch.optim.Adam(list(enc.parameters())+list(head.parameters()), lr=a.lr, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, drop_last=True)
    for ep in range(a.epochs):
        enc.train(); head.train(); tot = 0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad()
            loss = F.cross_entropy(head(enc(xb), yb), yb); loss.backward(); opt.step(); tot += loss.item()
        if (ep+1) % 5 == 0 or ep == 0: log(f"  epoch {ep+1}/{a.epochs} loss={tot/len(dl):.3f}")

    # extract EVAL embeddings -> features2_enc/ (X=embeddings, y=stage)
    enc.eval(); out = d(a.root, "features2_enc"); os.makedirs(out, exist_ok=True)
    with torch.no_grad():
        for s in ev:
            X, y = cache[s]; Z = []
            for i in range(0, len(X), 512):
                xb = torch.tensor(norm(X[i:i+512])).to(dev); Z.append(enc(xb).cpu().numpy())
            np.savez_compressed(d(out, f"{s}.npz"), X=np.concatenate(Z), y=y)
    log(f"embeddings saved -> features2_enc/ ({len(ev)} EVAL subjects, dim={a.emb})")

    # score with v2's validated cross-stage logic
    import importlib.util
    sp = importlib.util.spec_from_file_location("v2", d(a.root, "sleep_pipeline_v2.py"))
    v2 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v2)
    ec = {}
    for fp in sorted(glob.glob(d(out, "*.npz"))):
        z = np.load(fp, allow_pickle=True); ec[os.path.basename(fp)[:6]] = (z["X"], z["y"])
    res = {}
    for fuse in [1, 5, 10]:
        M = v2._crossstage(ec, a.min_epochs, fuse=fuse, rng=np.random.default_rng(a.seed))
        wi, cr = v2._means(M); res[f"encoder_fuse{fuse}"] = {"within": round(wi, 2), "cross": round(cr, 2),
            "within_by_stage": {v2.STAGES[i]: (None if np.isnan(M[i,i]) else round(float(M[i,i]),2)) for i in range(5)}}
        log(f"  encoder_fuse{fuse}: within={wi:.2f} cross={cr:.2f}")
    rng2 = np.random.default_rng(a.seed)
    M10 = v2._crossstage(ec, a.min_epochs, fuse=10, rng=np.random.default_rng(a.seed))
    Mr = v2._crossstage(ec, a.min_epochs, fuse=10, shuffle_rng=rng2, rng=np.random.default_rng(a.seed))
    boot = []
    for _ in range(a.boot):
        bs = list(rng2.choice(ev, len(ev), replace=True)); ecb = {f"{s}#{i}": ec[s] for i, s in enumerate(bs)}
        wb, cb = v2._means(v2._crossstage(ecb, a.min_epochs, fuse=10, rng=np.random.default_rng(a.seed))); boot.append([wb, cb])
    boot = np.array(boot); wi, cr = v2._means(M10)
    out_json = {"n_train": len(train), "n_eval": len(ev), "emb_dim": a.emb, "epochs": a.epochs,
                "methods": res, "best_method": "encoder_fuse10",
                "within_EER": round(wi, 2), "cross_EER": round(cr, 2),
                "within_CI95": [round(float(np.nanpercentile(boot[:,0],2.5)),2), round(float(np.nanpercentile(boot[:,0],97.5)),2)],
                "cross_CI95": [round(float(np.nanpercentile(boot[:,1],2.5)),2), round(float(np.nanpercentile(boot[:,1],97.5)),2)],
                "random_identity_baseline_EER": round(float(np.nanmean(Mr)), 2),
                "leakage_check": "PASS: encoder trained on TRAIN subjects only; EVAL disjoint; z-norm stats from TRAIN; enrol/probe disjoint epochs."}
    json.dump(out_json, open(d(a.root, "summary_encoder.json"), "w"), indent=2)
    torch.save(enc.state_dict(), d(a.root, "encoder.pt"))
    log("DONE -> summary_encoder.json, encoder.pt, features2_enc/")


if __name__ == "__main__":
    sys.exit(main())
