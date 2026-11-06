# Project guide

## Research question

This classroom prototype asks two questions: do normal/abnormal ECG beat-classification errors vary across available age and sex cohorts, and can local explanations help people inspect individual model behavior?

The original idea focuses on PPG skin-pigmentation disparities. MIT-BIH provides real ECG waveforms and some age/sex information, but no skin-pigmentation field. This project therefore demonstrates a related demographic audit without claiming to test that original skin-tone hypothesis.

## Data and labels

The script downloads MIT-BIH Arrhythmia Database records and `atr` annotations from PhysioNet using WFDB. For each annotated beat, it crops a short waveform centered at the annotation sample and extracts measurements from fixed windows:

- P window: 200 to 100 ms before the annotated beat.
- QRS window: 80 ms before to 80 ms after.
- ST window: 80 to 180 ms after.
- T window: 180 to 360 ms after.

The windows yield mean, range, RMS, a rough QRS-width proxy, ST slope, RR interval, RR change, and a signal-noise proxy. These are fixed-window engineering features, not clinician-reviewed fiducials or clinical measurements. A positive label means the beat annotation is not `N`; it does not mean the patient has atrial fibrillation.

The subject is assigned to a split before beat extraction. This avoids the serious leakage that would occur if beats from one patient's record appeared in both train and test. Records 201 and 202 are grouped as one subject.

## Model and explanations

A scikit-learn Random Forest predicts the binary beat label using the signal features only. Age and sex are not model inputs. SHAP TreeExplainer provides local contributions and mean absolute global importance. LIME creates a local interpretable surrogate around a selected test example. Both methods describe the fitted model; neither explains biological causality.

The dashboard plots a selected annotated beat and colors approximate P, QRS, ST, and T windows. SHAP and LIME tabs show local feature contributions for the same example. The methods may differ because they use different explanation approaches and assumptions.

## Fairness audit and mitigation

Fairlearn MetricFrame reports accuracy, false-positive rate (FPR), false-negative rate (FNR), true-positive rate (TPR), and selection rate for available sex and age-band groups. The report also calculates demographic-parity and equalized-odds differences. Tables show both the number of distinct subjects and the number of beat rows in each group. The rates themselves are beat-weighted, so the beat count is not an independent sample size; subjects can contribute many correlated beats.

Fairlearn ThresholdOptimizer is fitted only on validation beats with an FPR-parity constraint and balanced-accuracy objective. Predictions are evaluated on held-out subjects. The postprocessor can randomize decisions and requires group membership at prediction time. Compare all metrics: matching FPR can increase FNR, change selection rates, or lower accuracy. The small cohort counts and repeated beats mean these estimates are exploratory, not population-level evidence.

## Generated files

- `data/mitdb/`: downloaded source recordings and annotations (ignored by Git).
- `output/mitbih_beats.csv`: extracted beats, signal features, annotation label, split, and audit metadata.
- `output/index.html`: standalone dashboard with cohort tables, waveform display, global SHAP summary, and local SHAP/LIME explanations.
- `output/report.md`: reproducible text report.

## Presentation outline

1. Show the source database and explain patient-level splitting.
2. Walk through the waveform and approximate component windows.
3. Explain why beat-label classification is not AF diagnosis.
4. Compare FPR/FNR before and after post-processing across age and sex cohorts.
5. Contrast SHAP with LIME on one beat.
6. Discuss the missing pigmentation metadata and connect the separate FDA pulse-oximeter case study.

## Code tour

`download_if_needed` gets the records; `subject_info` reads available age/sex comments; `summarize_segment` builds waveform features; `build_dataset` creates subject splits; `metric_table` and `fit_fpr_optimizer` run the fairness analysis; SHAP and LIME are used in `main`; `write_dashboard` creates the browser view.
