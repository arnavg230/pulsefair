# PulseFair: ECG explainability and fairness audit

Ethics in AI college project · Python · scikit-learn · SHAP · LIME · Fairlearn

PulseFair trains a Random Forest on **real, annotated ECG beats** from the MIT-BIH Arrhythmia Database. It explains model scores with SHAP and LIME, compares group error metrics by the age and sex metadata available in the records, and demonstrates an FPR-parity threshold postprocessor. The dashboard also shows approximate P/QRS/ST/T beat windows and links the related FDA pulse-oximeter safety case.

> **Educational project only.** This is not a medical device or a clinical model. MIT-BIH is a small historical dataset. Its public metadata has no skin-pigmentation labels, so this project does not test skin-tone disparities. The waveform windows are educational approximations, not clinical ECG delineation.

## Run locally

Use Python 3.10–3.12. Install the dependencies, then run the pipeline:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\pulsefair.py
```

On macOS/Linux, activate with `source .venv/bin/activate` and run `python src/pulsefair.py`.

The first run downloads the MIT-BIH signal files and beat annotations from PhysioNet into `data/mitdb/` (about 95 MB). It then creates:

- `output/index.html` — interactive audit dashboard, annotated waveform, and SHAP/LIME local explanations.
- `output/report.md` — tables and interpretation notes for a report or presentation.
- `output/mitbih_beats.csv` — extracted beat features, labels, age/sex audit fields, and subject-level split.

The data folder is ignored by Git. Teammates can reproduce it by running the script; the downloaded database has its own attribution terms. See [DATA_SOURCES.md](docs/DATA_SOURCES.md).

If creating a virtual environment fails on your Python installation, install into that Python directly with `python -m pip install -r requirements.txt`, then rerun the script. A fresh GitHub clone will need this dependency setup and the one-time dataset download.

## What the implementation does

1. Downloads PhysioNet MIT-BIH and reads ECG waveforms plus annotated beat labels using WFDB.
2. Builds fixed, approximate P-wave, QRS, ST, and T-wave window features, RR interval features, and a signal-noise proxy.
3. Splits by subject before training so beats from one person cannot leak across train, validation, and test sets. Records 201/202 are kept together because they belong to one subject.
4. Fits a scikit-learn Random Forest to distinguish normal (`N`) beats from other annotated beat types.
5. Produces SHAP TreeExplainer global and local feature contributions and LIME local surrogate explanations.
6. Audits accuracy, FPR, FNR, TPR, selection rate, demographic-parity difference, and equalized-odds difference by available sex and age-band metadata.
7. Fits Fairlearn `ThresholdOptimizer` on validation beats to target false-positive-rate parity, then reports test-set trade-offs.

## Important interpretation limits

- “Abnormal” means a beat annotation other than `N`; it is not an AF diagnosis or a patient-level clinical outcome.
- MIT-BIH has only 47 subjects and historical recordings. Beat counts are not independent patient counts; group results are exploratory.
- The dataset provides limited age/sex metadata, not skin tone. The FDA pulse-oximeter case is adjacent context about optical sensors, not evidence about these ECG results.
- The fixed wave windows and width estimate are approximate signal features, not validated P/QRS/ST segmentation.
- Fairness post-processing uses group membership at decision time and may randomize predictions. Equal FPR alone does not establish fairness.
- SHAP and LIME describe model behavior, not physiological causes or clinical reasoning.

## Repo map

```text
src/pulsefair.py       Dataset download, beat features, model, XAI, audits, report output
requirements.txt       Python libraries
docs/PROJECT_GUIDE.md  Pipeline, metrics, and presentation outline
docs/ETHICS_NOTES.md   Ethical limitations and discussion questions
docs/DATA_SOURCES.md   Dataset citation, license, FDA case-study sources
output/                Generated dashboard, results, and extracted feature CSV
```

## Team presentation outline

1. State the question: can error rates differ across demographic cohorts, and what can an explanation reveal?
2. Walk through one annotated beat and its approximate signal windows.
3. Compare FPR/FNR across age and sex groups before and after threshold adjustment.
4. Compare SHAP and LIME for the same beat; discuss why explanations can differ.
5. Explain why this dataset cannot answer the screenshot's specific skin-pigmentation question, then connect to the FDA pulse-oximeter case study.

## License

Code and documentation in this repository are provided under the MIT License. MIT-BIH data are separate and remain subject to the PhysioNet Open Data Commons Attribution License v1.0; cite the database when using or presenting its data.
