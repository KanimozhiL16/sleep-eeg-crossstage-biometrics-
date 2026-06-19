# Sleep-Stage and Night Dependence of EEG Biometric Identity

Reference implementation for the paper:

> **EEG Biometric Identity Is Sleep-Stage and Night Dependent: Per-Stage and Cross-Night Verification on Sleep-EDF**
> L. Kanimozhi and S. Shridevi, Vellore Institute of Technology, Chennai.
> *IEEE Signal Processing in Medicine and Biology Symposium (SPMB) 2026.*

This repository contains the full pipeline used to evaluate EEG-based biometric
verification **within sleep stage, across sleep stages, and across nights** on the
public Sleep-EDF Expanded database. It reproduces every number, table, and figure
reported in the paper from the raw recordings.

---

## Summary

We quantify how the identity information carried by sleep EEG depends on the sleep
stage and the recording night. Using a subject- and session-disjoint protocol with a
one-dimensional convolutional ArcFace encoder (128-D embeddings) and N-epoch probe
fusion, verification is near-perfect within a single stage and night but degrades
sharply when stage or night changes:

| Condition | Equal Error Rate (EER) |
|---|---|
| Within-stage, within-night | 0.21% |
| Across stages | 9.58% |
| Across nights | 17.28% |
| Across stages **and** nights | 23.87% |
| Random baseline | ~49.6% |

A spectral band-ablation shows the identity signal is distributed across frequency
bands, with the theta band contributing the largest single share. Multi-stage
enrolment and periodic re-enrolment recover much of the lost performance. The
practical message: **sleep-stage-blind biometric systems silently fail across stages
and nights.**

---

## Repository structure

```
sleep_pipeline.py        Core cross-stage / cross-night verification pipeline
sleep_pipeline_v2.py     Probe-fusion + per-stage EER protocol (paper version)
sleep_pipeline_v3.py     Multi-stage enrolment experiment
sleep_pipeline_v4.py     Periodic re-enrolment experiment
encoder_train.py         1D-CNN ArcFace encoder training
band_ablation2.py        Spectral band-ablation (which bands carry identity)
stats_tests.py           Paired Wilcoxon test, Cohen's d, bootstrap CIs
make_paper_assets.py     Generates all paper figures and tables
```

---

## Data

- **Database:** Sleep-EDF Expanded (Sleep Cassette), PhysioNet.
  https://physionet.org/content/sleep-edfx/
- **Channel:** Fpz–Cz EEG.
- **Epochs:** 30-second, stage-labelled (W, N1, N2, N3, REM) per the accompanying
  hypnogram annotations.
- The raw EDF files are **not** redistributed here; download them directly from
  PhysioNet. The pipeline reads the standard `*-PSG.edf` / `*-Hypnogram.edf` pairs.

---

## Protocol (leakage controls)

- **Subject-disjoint** train/test split — no subject appears in both.
- **Session-disjoint** cross-night evaluation — enrolment and probe come from
  different nights.
- **Enrolment/probe separation** — templates and probes never share epochs.
- Equal error rate (EER) is reported within stage, across stages, and across nights.

---

## Requirements

```
python >= 3.9
numpy, scipy, scikit-learn
mne
torch
matplotlib
```

```bash
pip install numpy scipy scikit-learn mne torch matplotlib
```

---

## Reproducing the results

1. Download Sleep-EDF Expanded from PhysioNet and set the data path inside the
   pipeline script.
2. Train the encoder:
   ```bash
   python encoder_train.py
   ```
3. Run the verification protocol (per-stage, cross-stage, cross-night):
   ```bash
   python sleep_pipeline_v2.py
   ```
4. Multi-stage enrolment and re-enrolment:
   ```bash
   python sleep_pipeline_v3.py
   python sleep_pipeline_v4.py
   ```
5. Spectral band-ablation:
   ```bash
   python band_ablation2.py
   ```
6. Statistical tests and paper figures/tables:
   ```bash
   python stats_tests.py
   python make_paper_assets.py
   ```

Trained checkpoints and preprocessed features are additionally archived on
Hugging Face for convenience:
https://huggingface.co/KanimozhiL16

---

## Hardware

Experiments were run on NVIDIA A100 GPUs provided through the NVIDIA Academic Grant
Program (awarded to S. Shridevi).

---

## Citation

```bibtex
@inproceedings{kanimozhi2026sleepeeg,
  title     = {EEG Biometric Identity Is Sleep-Stage and Night Dependent:
               Per-Stage and Cross-Night Verification on Sleep-EDF},
  author    = {Kanimozhi, L. and Shridevi, S.},
  booktitle = {IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
  year      = {2026}
}
```

---

## Acknowledgements

We thank PhysioNet for the Sleep-EDF Expanded database. This work used NVIDIA A100
GPUs provided through the NVIDIA Academic Grant Program (awarded to S. Shridevi).

## License

Released for academic and research use. Please cite the paper above if you use this
code or build on it.
