# PulseFair: ECG explainability and fairness audit

**Ethics in AI college project** · Python · scikit-learn · SHAP · LIME · Fairlearn

PulseFair is a small, reproducible experiment about how an AI model classifies heartbeats, how we can inspect its decisions, and whether its errors differ across demographic groups. It uses annotated ECG recordings from the MIT-BIH Arrhythmia Database and provides a browser dashboard plus a written report.

> **Educational project only.** This is not a medical device or a clinical model. MIT-BIH is a small historical dataset. The waveform windows are educational approximations, not clinical ECG delineation.

## The question this project explores

When a model flags an ECG beat, what signal measurements influenced its score? Do its errors look different across the age and sex groups available in the dataset? Can a post-processing rule reduce one group error gap, and does the change still hold for subjects kept out of development?

The project is designed to make those questions inspectable. It does not claim that any one fairness metric defines fairness or that an explanation proves a model is safe.

## Start here for a project overview

For a teammate-facing explanation of the project question, pipeline, dashboard, fairness methods, and interpretation limits, see [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md).

## What happens when you run it

The main program is `src/pulsefair.py`. In broad terms, it:

1. **Gets the data.** WFDB downloads and reads ECG recordings and beat annotations from PhysioNet's MIT-BIH Arrhythmia Database.
2. **Makes beat examples.** For each annotated beat, it extracts a short signal window and calculates features such as segment means/ranges, a QRS-width proxy, RR timing, and a signal-noise proxy. P/QRS/ST/T windows are fixed approximate regions around the annotation.
3. **Splits by subject.** Subjects are assigned to training, validation, or held-out test groups before model fitting. This prevents beats from the same person appearing in both training and test. Records 201 and 202 are grouped because they belong to one subject.
4. **Trains the classifier.** A scikit-learn Random Forest uses the signal features to distinguish `N` (normal-annotated) beats from all other included beat annotations. Age and sex are not model inputs.
5. **Explains model behavior.** SHAP provides local feature contributions for a selected beat and average global importance. LIME provides a second local explanation method.
6. **Audits group metrics.** Fairlearn calculates performance measures across available age and sex groups. A validation-fitted threshold post-processor targets false-positive-rate parity. The script then evaluates it on held-out subjects to show possible trade-offs and generalization limits.
7. **Creates outputs.** It writes a standalone HTML dashboard, Markdown report, and CSV of beat-level features and labels.

## Run locally

Use Python 3.10–3.12. For a full clean-clone walkthrough, troubleshooting, and explanations of the generated files, see [docs/TEAMMATE_SETUP.md](docs/TEAMMATE_SETUP.md).

### Windows PowerShell

Run these commands from the repository folder. Using the virtual environment's Python directly means activation is optional.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src\pulsefair.py
```

### macOS/Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/pulsefair.py
```

The first run downloads the MIT-BIH signal files and beat annotations into `data/mitdb/` (about 95 MB), so internet access is needed. The download may take a few minutes. When it completes, open `output/index.html` in a browser.

If creating a virtual environment fails, see the fallback instructions in [docs/TEAMMATE_SETUP.md](docs/TEAMMATE_SETUP.md). The dataset itself is not included in the repository; the data folder is ignored by Git and is recreated by running the script. See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for attribution and data terms.

## Understanding the outputs

- **`output/index.html`** — interactive dashboard. It contains age/sex audit tables, a key validation-versus-held-out comparison, global SHAP importance, and a selector for inspecting one held-out beat with its waveform and SHAP/LIME explanations.
- **`output/report.md`** — text summary and metric tables suitable for reading alongside the dashboard or adapting for a report.
- **`output/mitbih_beats.csv`** — one row per extracted beat with features, annotation-derived label, data split, and available audit metadata. It is useful for understanding what the model saw.
- **`data/mitdb/`** — downloaded recordings and annotations used to rebuild the other outputs.

The dashboard is generated locally as a standalone HTML file; it does not require a web server. Run the Python script again to regenerate the results and outputs.

## Reading the model and fairness metrics

The task is **beat-level annotation classification**, not patient diagnosis. In this project, `N` is the normal class and any other included beat annotation is grouped into the positive “other” class.

- **Accuracy:** fraction of beats whose prediction matches the annotation-derived reference label.
- **FPR (false-positive rate):** fraction of reference-normal beats incorrectly flagged as “other.” A high FPR means more false alarms in that group.
- **FNR (false-negative rate):** fraction of reference-other beats incorrectly classified as normal. A high FNR means more misses.
- **TPR (true-positive rate):** fraction of reference-other beats correctly flagged; also called sensitivity or recall.
- **Flag/selection rate:** fraction of beats predicted as “other.”
- **Demographic parity difference:** difference between groups' flag rates.
- **Equalized odds difference:** a summary of the largest group difference across FPR and TPR.

Rates are calculated over beats, and the dashboard also shows the number of subjects in each group. Beat observations from the same person are correlated, so a large beat count does not equal a large independent patient sample.

The **FPR-parity post-processor** is fitted on validation data to try to make group false-positive rates closer. It can use group membership at decision time and may randomize decisions. Check all metrics before and after adjustment: reducing one gap can affect other errors, accuracy, or flag rates. The held-out comparison matters because an improvement on validation data may not carry over to subjects the method did not use during development.

## Explainability in the dashboard

- **Global SHAP importance** summarizes the average magnitude of feature contributions across held-out examples. It shows what the fitted model tends to use; it is not a ranking of medical importance.
- **Local SHAP** assigns contributions to features for one selected model score, relative to a baseline. Positive and negative contributions push that score in different directions.
- **LIME** approximates the model near one selected beat with a simpler local model. It may give a different explanation from SHAP because the approaches make different assumptions.

Both methods describe model behavior. Neither establishes physiological cause, clinical reasoning, or safety.

## Important interpretation limits

- “Other/abnormal” means an annotation other than `N`; it is not atrial fibrillation, a disease diagnosis, or a patient-level outcome.
- MIT-BIH is historical and has only 47 subjects in this project. Group results are exploratory and should not be generalized to a population.
- Group comparisons use the age and sex metadata available in MIT-BIH; they are exploratory and limited by the small number of subjects.
- The FDA pulse-oximeter case study is related context about optical sensors, not evidence that this ECG dataset or model has the same disparity. Race is not a direct measurement of skin tone.
- The fixed waveform windows and width estimate are approximate signal features, not validated P/QRS/ST segmentation.
- Fairness post-processing and equal FPR do not by themselves establish fairness; thresholding can involve trade-offs and use group membership.
- SHAP and LIME are feature-attribution tools, not causal or clinical explanations.

## Repository map

```text
src/pulsefair.py       Downloads/reads data, builds features, trains model, audits fairness, writes outputs
requirements.txt       Python libraries needed to run the project

docs/PROJECT_GUIDE.md  Pipeline details, metric definitions, and presentation outline
docs/ETHICS_NOTES.md   Ethical interpretation, limits, and discussion questions
docs/DATA_SOURCES.md   Dataset attribution, license, and case-study sources
docs/TEAMMATE_SETUP.md Fresh-computer setup, troubleshooting, and output/CSV guide

output/                Generated dashboard, report, and extracted beat CSV
data/mitdb/            Downloaded dataset files (created on first run; ignored by Git)
```

## Suggested presentation flow

1. Introduce the question: can ECG model errors vary across groups, and what can explanations show?
2. Explain the dataset labels and why the model predicts beat annotations rather than a diagnosis.
3. Show how subjects are separated into train, validation, and held-out sets.
4. Inspect a waveform and compare SHAP and LIME for the same held-out beat.
5. Compare age/sex error metrics before and after the validation-fitted threshold adjustment, then discuss what happened on held-out subjects.
6. Explain that the audit uses available age and sex metadata, and introduce the pulse-oximeter case as related healthcare-sensor context.

## License

Code and documentation in this repository are provided under the MIT License. MIT-BIH data are separate and remain subject to the PhysioNet Open Data Commons Attribution License v1.0; cite the database when using or presenting its data.
