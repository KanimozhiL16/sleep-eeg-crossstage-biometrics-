#!/usr/bin/env python3
"""
Consolidate ALL paper figures + tables from result files (100% from measured outputs).
No re-computation — only reads the saved JSON/CSV. Grayscale, IEEE-friendly.

Usage:  python make_paper_assets.py --indir <dir with result json/csv> --outdir paper_assets
Reads (if present): stats.json, validation_v2.json, crossnight.json, crossnight_eer_matrix.csv,
det.json, ttd_enc.json, featstab.json, multistage.json, threshold_transfer_HTER.csv,
eer_lda_fuse10.csv, enc_eer_fuse10.csv (optional), tsne_coords.csv.
Writes: fig_*.png/.pdf + table_master_results.csv + table_perstage.csv (+ .tex).
"""
import argparse, json, os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "font.family": "serif", "savefig.dpi": 300, "savefig.bbox": "tight"})
STAGES = ["W", "N1", "N2", "N3", "REM"]

def jload(p):
    try: return json.load(open(p))
    except Exception: return None
def loadmat(p):
    try:
        rows = list(csv.reader(open(p)))
        M = np.array([[np.nan if c in ("NA", "") else float(c) for c in r[1:]] for r in rows[1:]])
        return M
    except Exception: return None
def save(fig, out, name):
    for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig); print("wrote", name)

def heat(M, out, name, title, vmax=50):
    if M is None: return
    fig, ax = plt.subplots(figsize=(3.4, 3.0)); im = ax.imshow(M, cmap="gray_r", vmin=0, vmax=vmax)
    ax.set_xticks(range(5)); ax.set_xticklabels(STAGES); ax.set_yticks(range(5)); ax.set_yticklabels(STAGES)
    ax.set_xlabel("probe stage"); ax.set_ylabel("enrol stage"); ax.set_title(title)
    for i in range(5):
        for j in range(5):
            if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                                              color="white" if M[i, j] > vmax/2 else "black", fontsize=7)
    fig.colorbar(im, label="EER (%)"); save(fig, out, name)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--indir", default="."); ap.add_argument("--outdir", default="paper_assets")
    a = ap.parse_args(); I = a.indir; O = a.outdir; os.makedirs(O, exist_ok=True)
    P = lambda f: os.path.join(I, f)

    # ---- Fig: degradation ladder (headline) ----
    val = jload(P("validation_v2.json")); cn = jload(P("crossnight.json")); st = jload(P("stats.json"))
    bars, vals, los, his = [], [], [], []
    enc = jload(P("summary_encoder.json")); cienc = jload(P("ci_enc_fuse10.json"))  # corrected CIs
    if enc:
        wci = cienc["within_CI95"] if cienc else enc["within_CI95"]
        cci = cienc["cross_CI95"] if cienc else enc["cross_CI95"]
        bars.append("within\nstage"); vals.append(enc["within_EER"]); los.append(wci[0]); his.append(wci[1])
        bars.append("cross\nstage"); vals.append(enc["cross_EER"]); los.append(cci[0]); his.append(cci[1])
    if cn:
        bars.append("cross\nnight"); vals.append(cn["cross_night_same_stage_EER"]); los.append(cn["cross_night_same_stage_CI95"][0]); his.append(cn["cross_night_same_stage_CI95"][1])
        bars.append("cross night\n+ stage"); vals.append(cn["cross_night_AND_cross_stage_EER"]); los.append(cn["cross_night_cross_stage_CI95"][0]); his.append(cn["cross_night_cross_stage_CI95"][1])
    if bars:
        v = np.array(vals); lo = np.clip(v-np.array(los), 0, None); hi = np.clip(np.array(his)-v, 0, None)
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        ax.bar(range(len(bars)), v, yerr=[lo, hi], capsize=3, color="0.5", edgecolor="k")
        ax.axhline(50, ls="--", c="k", lw=0.8, label="chance"); ax.set_xticks(range(len(bars))); ax.set_xticklabels(bars)
        ax.set_ylabel("EER (%)"); ax.set_title("Identity degradation (encoder)"); ax.legend(fontsize=7)
        for i, x in enumerate(v): ax.text(i, x+hi[i]+1, f"{x:.1f}", ha="center", fontsize=7)
        save(fig, O, "fig_degradation_ladder")

    # ---- Heatmaps ----
    heat(loadmat(P("enc_eer_fuse10.csv")) if os.path.exists(P("enc_eer_fuse10.csv")) else loadmat(P("eer_lda_fuse10.csv")),
         O, "fig_crossstage_heatmap", "Cross-stage EER")
    heat(loadmat(P("crossnight_eer_matrix.csv")), O, "fig_crossnight_heatmap", "Cross-night EER (enrol N1, probe N2)")
    heat(loadmat(P("threshold_transfer_HTER.csv")), O, "fig_threshold_HTER", "Threshold transfer HTER", vmax=5)

    # ---- Time-to-decision ----
    ttd = jload(P("ttd_enc.json"))
    if ttd:
        c = ttd["time_to_decision"]; ks = sorted(c, key=lambda k: int(k))
        mins = [c[k]["minutes"] for k in ks]; wm = [c[k]["within_mean_EER"] for k in ks]; bs = [c[k]["best_stage_EER"] for k in ks]
        fig, ax = plt.subplots(figsize=(3.4, 2.6)); ax.plot(mins, wm, "o-k", label="within-stage mean"); ax.plot(mins, bs, "s--", c="0.5", label="best stage")
        ax.set_xlabel("probe duration (min of sleep)"); ax.set_ylabel("EER (%)"); ax.set_title("Time-to-decision"); ax.legend(fontsize=7); save(fig, O, "fig_time_to_decision")

    # ---- DET ----
    det = jload(P("det.json"))
    if det:
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        for key, sty in [("within_stage", "-k"), ("cross_stage", "--")]:
            cur = det[key]["curve"]; far = [p["FAR"] for p in cur]; frr = [p["FRR"] for p in cur]
            ax.plot(far, frr, sty, label=f"{key.replace('_',' ')} (EER {det[key]['EER']}%)", color=None if sty=="-k" else "0.5")
        ax.set_xlabel("FAR (%)"); ax.set_ylabel("FRR (%)"); ax.set_title("DET"); ax.legend(fontsize=7); save(fig, O, "fig_det")

    # ---- Feature stability (discriminability vs state-robustness) ----
    fs = jload(P("featstab.json"))
    if fs and "full_ranking" in fs:
        # need disc; combined ranking has both
        comb = {r["feature"]: r for r in fs.get("genuine_identity_carriers_top5_combined", [])}
        rank = fs["full_ranking"]
        # build from identity_discriminability lists if present
        disc = {}
        for key in ("identity_discriminability_top5", "identity_discriminability_bottom5"):
            for r in fs.get(key, []): disc[r["feature"]] = r["discriminability"]
        xs, ys, names = [], [], []
        for r in rank:
            f = r["feature"]
            if f in disc: xs.append(disc[f]); ys.append(r["stability_index"]); names.append(f)
        if xs:
            fig, ax = plt.subplots(figsize=(3.6, 3.0)); ax.scatter(xs, ys, c="0.4")
            for x, y, n in zip(xs, ys, names):
                if x > 1.0 or y > 0.6: ax.annotate(n.split(":")[-1], (x, y), fontsize=5)
            ax.set_xlabel("identity discriminability"); ax.set_ylabel("state-robustness (ICC-like)")
            ax.set_title("Feature identity vs state-robustness"); save(fig, O, "fig_featstab")

    # ---- multistage ----
    ms = jload(P("multistage.json"))
    if ms:
        d = ms["enrol_strategy_probeEER"]; strat = list(d.keys()); means = [np.mean(list(d[s].values())) for s in strat]
        fig, ax = plt.subplots(figsize=(3.4, 2.6)); ax.bar(range(len(strat)), means, color="0.5", edgecolor="k")
        ax.set_xticks(range(len(strat))); ax.set_xticklabels(strat, rotation=20, fontsize=7); ax.set_ylabel("mean probe EER (%)")
        ax.set_title("Enrolment strategy"); save(fig, O, "fig_multistage")

    # ---- re-enrolment curve ----
    rr = jload(P("reenroll.json"))
    if rr and "reenroll_fraction_to_sameStageEER" in rr:
        c = rr["reenroll_fraction_to_sameStageEER"]; fs = sorted(c, key=lambda k: float(k))
        x = [float(k)*100 for k in fs]; y = [c[k] for k in fs]
        fig, ax = plt.subplots(figsize=(3.4, 2.6)); ax.plot(x, y, "o-k")
        ax.set_xlabel("night-2 data folded into template (%)"); ax.set_ylabel("cross-night EER (%)")
        ax.set_title("Re-enrolment recovers night-drift")
        for xi, yi in zip(x, y): ax.text(xi, yi+0.4, f"{yi:.1f}", ha="center", fontsize=6)
        save(fig, O, "fig_reenroll")

    # ---- t-SNE ----
    if os.path.exists(P("tsne_coords.csv")):
        rows = list(csv.DictReader(open(P("tsne_coords.csv"))))
        X = np.array([[float(r["x"]), float(r["y"])] for r in rows]); stg = [r["stage"] for r in rows]; sub = [r["subject"] for r in rows]
        cmap = {s: plt.cm.viridis(i/4) for i, s in enumerate(STAGES)}
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        for s in STAGES:
            m = [i for i, v in enumerate(stg) if v == s]; ax.scatter(X[m, 0], X[m, 1], s=3, color=cmap[s], label=s)
        ax.legend(fontsize=6, markerscale=2); ax.set_title("t-SNE by sleep stage"); ax.set_xticks([]); ax.set_yticks([]); save(fig, O, "fig_tsne_stage")
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        us = sorted(set(sub)); col = {u: plt.cm.tab20(i % 20) for i, u in enumerate(us)}
        ax.scatter(X[:, 0], X[:, 1], s=3, c=[col[u] for u in sub]); ax.set_title("t-SNE by subject (identity)"); ax.set_xticks([]); ax.set_yticks([]); save(fig, O, "fig_tsne_subject")

    # ---- MASTER TABLE ----
    rows = []
    if enc: rows += [["Within-stage (encoder, fuse10)", enc["within_EER"], enc.get("within_CI95")],
                     ["Cross-stage (encoder, fuse10)", enc["cross_EER"], enc.get("cross_CI95")]]
    if cn: rows += [["Cross-night same-stage", cn["cross_night_same_stage_EER"], cn["cross_night_same_stage_CI95"]],
                    ["Cross-night + cross-stage", cn["cross_night_AND_cross_stage_EER"], cn["cross_night_cross_stage_CI95"]]]
    if det: rows += [["Within-stage FRR@FAR=1%", det["within_stage"]["FRR_at_FAR1pct"], None]]
    if ms: rows += [["Multi-stage (all5) enrol, mean probe", round(np.mean(list(ms["enrol_strategy_probeEER"]["all5"].values())), 2), None]]
    with open(os.path.join(O, "table_master_results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Condition", "EER_%", "95%CI"]);
        for r in rows: w.writerow([r[0], r[1], r[2]])
    # per-stage table
    if enc and cn:
        with open(os.path.join(O, "table_perstage.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["Stage", "within_EER", "crossnight_same_stage_EER"])
            for s in STAGES: w.writerow([s, enc["methods"]["encoder_fuse10"]["within_by_stage"].get(s), cn["same_stage_by_stage"].get(s)])
    if st:
        json.dump({"cross_stage_gap_p": st["gap_within_vs_cross_encoder"].get("wilcoxon_cross_gt_within_p"),
                   "cohens_d": st["gap_within_vs_cross_encoder"]["cohens_d"],
                   "encoder_vs_cosine_p": st["method_encoder_vs_cosine_within"].get("wilcoxon_cosine_gt_encoder_p")},
                  open(os.path.join(O, "stats_summary.json"), "w"), indent=2)
    print("DONE. assets in", O)

if __name__ == "__main__":
    main()
