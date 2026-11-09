# PulseFair project overview

This document gives teammates the project-level story: what problem PulseFair investigates, how its pipeline is organized, what the dashboard shows, and how to interpret its evidence. For a machine-specific installation walkthrough, see [TEAMMATE_SETUP.md](TEAMMATE_SETUP.md). For implementation details and metric definitions, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## Project summary

PulseFair is an educational explainable-AI and fairness-auditing pipeline for beat-level ECG classification. It uses annotated recordings from the MIT-BIH Arrhythmia Database, trains a Random Forest on engineered signal features, explains selected model outputs with SHAP and LIME, and compares performance across available age and sex cohorts. It also demonstrates a Fairlearn post-processing method that targets false-positive-rate parity.

The central question is:

> When an ECG model flags a beat, what features influenced the score, do error rates differ across the available groups, and does a fairness adjustment generalize to held-out subjects?

The project makes model behavior and fairness trade-offs visible. It does not build a clinical system or claim to establish that one metric is a complete definition of fairness.

## Scope and task definition

The unit of prediction is an annotated **heartbeat**, not a patient. The labels are simplified from the database annotations into a binary task:

- `N` annotations are treated as normal.
- Other included beat annotations are grouped into the positive “other” class.

The Random Forest receives ECG-derived signal features. Age and sex are reserved for group auditing and are not inputs to the classifier. This lets the project compare error rates across cohorts without using those demographic fields to make the beat prediction.

Calling the positive class “abnormal” in the dashboard is shorthand for this annotation grouping. It should not be interpreted as atrial fibrillation, a diagnosis, or a patient-level outcome.

## Data and signal representation

The script downloads the MIT-BIH Arrhythmia Database and reads ECG recordings and beat annotations through WFDB. Each annotated beat is represented by features calculated from a short window around the annotation. Features include summaries of approximate P/QRS/ST/T windows, RR interval timing and change, a rough QRS-width proxy, and a noise proxy.

The window boundaries are fixed engineering choices. They support a visible, explainable demonstration but are not clinically validated waveform delineations. The data contain 48 records from 47 subjects; repeated beats from a subject are correlated observations.

PulseFair uses the available age and sex metadata for its cohort audit. The pulse-oximeter case study supplies broader healthcare-sensor context; it is a separate evidence source from this ECG experiment.

## Pipeline at a glance

1. **Read and annotate.** Download records, read signal channels and beat annotation positions, and retrieve available demographic metadata.
2. **Assign subjects to splits.** Keep all beats from one subject in a single partition. Records 201 and 202 are kept together. This avoids train/test leakage from the same person's signal patterns.
3. **Build features and labels.** Extract approximate waveform and timing features for each beat and map annotations to the binary target.
4. **Fit the classifier.** Train a scikit-learn Random Forest on the training subjects using signal features.
5. **Explain predictions.** Use SHAP TreeExplainer for feature contributions and global importance; use LIME for an additional local surrogate explanation.
6. **Measure group performance.** Use Fairlearn MetricFrame and related calculations to compare accuracy, FPR, FNR, TPR, selection rate, demographic-parity difference, and equalized-odds difference across age and sex groups.
7. **Demonstrate mitigation.** Fit Fairlearn ThresholdOptimizer on validation data with an FPR-parity constraint and balanced-accuracy objective. Report its test performance on unseen subjects to expose generalization and trade-offs.
8. **Write artifacts.** Generate the standalone dashboard, a Markdown report, and a beat-level CSV for inspection.

## Why the subject split matters

If beat rows were randomly split, beats from a person's record could appear in both model training and evaluation. The model could then benefit from patterns specific to that person, making its performance appear stronger than it is on unseen people. Subject-level splitting makes the held-out evaluation more meaningful, even though the small number of subjects still limits conclusions.

The validation set is used during development, including fitting the post-processing rule. The held-out test set is intended to show how these choices perform on subjects not used for that fitting. The comparison is a teaching illustration, not a definitive clinical validation.

## Explainability methods

### SHAP

SHAP attributes a prediction to its input features relative to a baseline. The local view shows which features push one selected score up or down. Global importance aggregates the magnitude of feature contributions over held-out examples. It describes the Random Forest's learned behavior, not the physiological cause of a beat or clinical importance of a waveform region.

### LIME

LIME fits a simpler local approximation around a selected example and uses it to rank influential features. Because it uses a different approximation strategy and assumptions, its explanation may differ from SHAP. The comparison is useful for discussing the fact that explanations are method-dependent.

## Fairness measures and intervention

The dashboard reports group-level metrics, including:

- **FPR:** rate of reference-normal beats incorrectly flagged as positive.
- **FNR:** rate of reference-positive beats incorrectly classified as normal.
- **TPR:** rate of reference-positive beats correctly flagged.
- **Selection/flag rate:** share of beats predicted positive.
- **Demographic parity difference:** group difference in selection rate.
- **Equalized odds difference:** largest group difference across FPR and TPR.

Metrics are beat-weighted, while tables also show subject counts. A group may contain many beats but only a few independent people, so uncertainty and correlated observations matter.

The threshold postprocessor targets similar FPRs between groups on validation data. That objective does not guarantee broader fairness. Threshold policies can change FNR, accuracy, and selection rates; they also use group membership at decision time and may randomize predictions. PulseFair therefore displays multiple metrics and compares validation with held-out outcomes instead of reporting only the targeted metric.

## Website walkthrough

- **Header and three-step summary:** introduce the research question and the data → explanation → audit workflow.
- **Key finding:** compares the validation FPR gap before/after adjustment with the held-out gap. A smaller validation gap that does not persist on held-out subjects illustrates limited generalization.
- **Sex and age audit tables:** compare group metrics under a shared threshold and after post-processing; inspect both error rates and subject counts.
- **Global SHAP importance:** shows which engineered features most influence the fitted model on average.
- **Held-out beat inspector:** select one test beat, view its waveform and model score, and switch between local SHAP and LIME explanations. The reference class comes from the annotation-derived label.
- **Pulse-oximeter case study:** links external context on optical-sensor measurement disparities, clearly separate from this ECG experiment.
- **Source footer:** identifies the ECG database and points to the case-study references.

## How to present the result responsibly

A clear presentation can follow this sequence:

1. State the question and define the beat-level target.
2. Describe the dataset and explain why splitting by subject is important.
3. Show one waveform and its engineered features; clarify that colored windows are approximate.
4. Compare SHAP and LIME for a held-out beat and describe them as model-behavior explanations.
5. Compare group FPR/FNR and related measures before and after the adjustment.
6. Compare validation and held-out results, then discuss why parity on one split is not proof of a fair system.
7. Describe the age/sex cohorts used in the audit and distinguish the pulse-oximeter case study as related context.

## Limitations to keep visible in discussion

- MIT-BIH is historical and small, with 47 subjects in this project.
- Repeated beat rows are not independent patient samples; group rates are exploratory.
- Audits use available age/sex metadata and small subject cohorts.
- The binary annotation task is not disease diagnosis, AF detection, or patient-level risk prediction.
- P/QRS/ST/T feature windows and QRS-width proxy are approximate.
- SHAP and LIME do not establish cause, clinical reasoning, or safety.
- A threshold adjustment may improve the targeted validation metric but worsen other metrics or fail to generalize.
- The pulse-oximeter study is contextual evidence about another sensor and does not establish the same disparity in MIT-BIH or PulseFair.

## Generated artifacts and code location

- `src/pulsefair.py` contains the data workflow, feature extraction, model, explanations, fairness analysis, and output generation.
- `output/index.html` is the generated interactive dashboard.
- `output/report.md` contains the generated report and metric summaries.
- `output/mitbih_beats.csv` contains extracted features, labels, split assignments, and audit metadata.
- `data/mitdb/` contains downloaded PhysioNet files and is recreated locally; it is not committed to Git.

Refer to [DATA_SOURCES.md](DATA_SOURCES.md) for citations and data terms, and [ETHICS_NOTES.md](ETHICS_NOTES.md) for discussion questions and a concise ethical analysis.
