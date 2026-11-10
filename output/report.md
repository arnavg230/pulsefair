# PulseFair · real-data results

> Educational demonstration; not a medical device or clinical model.

- Source: PhysioNet MIT-BIH Arrhythmia Database (48 records, 47 subjects).
- Beat windows: 109,464; subject-level train/validation/test split to avoid the same person appearing across splits.
- Task: distinguish normal (`N`) beats from other beat annotations. This is not an atrial-fibrillation diagnosis.
- Cohorts: sex and broad age bands parsed from available headers.
- Group metrics are beat-weighted; subject counts are shown separately because beats from one person are correlated.
- FPR parity is calibrated on validation data and evaluated on held-out subjects; Fairlearn may randomize post-processed predictions.

## Sex cohorts

| Policy | Group | Subjects | Beats | Accuracy | FPR | FNR | TPR | Selection rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | F | 6 | 11797 | 0.662 | 0.060 | 0.646 | 0.354 | 0.200 |
| base | M | 3 | 9734 | 0.758 | 0.263 | 0.050 | 0.950 | 0.330 |
| adjusted | F | 6 | 11797 | 0.782 | 0.083 | 0.367 | 0.633 | 0.345 |
| adjusted | M | 3 | 9734 | 0.715 | 0.312 | 0.036 | 0.964 | 0.376 |
Unknown cohort metadata, omitted from this audit: 1 test subjects / 2154 test beats.

## Age cohorts

| Policy | Group | Subjects | Beats | Accuracy | FPR | FNR | TPR | Selection rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | 60 and over | 6 | 14101 | 0.686 | 0.096 | 0.607 | 0.393 | 0.223 |
| base | Under 60 | 3 | 7430 | 0.741 | 0.277 | 0.033 | 0.967 | 0.327 |
| adjusted | 60 and over | 6 | 14101 | 0.737 | 0.111 | 0.467 | 0.533 | 0.291 |
| adjusted | Under 60 | 3 | 7430 | 0.700 | 0.322 | 0.022 | 0.978 | 0.369 |
Unknown cohort metadata, omitted from this audit: 1 test subjects / 2154 test beats.

## Fairness gaps

| Attribute | Policy | FPR gap | Demographic-parity difference | Equalized-odds difference |
|---|---|---:|---:|---:|
| sex | Validation · shared threshold | 0.071 | 0.099 | 0.215 |
| sex | Validation · FPR parity | 0.003 | 0.111 | 0.111 |
| sex | Test · shared threshold | 0.203 | 0.130 | 0.596 |
| sex | Test · FPR parity | 0.229 | 0.031 | 0.331 |
| age_band | Validation · shared threshold | 0.044 | 0.029 | 0.160 |
| age_band | Validation · FPR parity | 0.000 | 0.120 | 0.079 |
| age_band | Test · shared threshold | 0.182 | 0.105 | 0.574 |
| age_band | Test · FPR parity | 0.211 | 0.079 | 0.445 |

## SHAP global feature importance

Mean absolute contribution across a sample of held-out beats; model importance is not medical importance.

| Feature | Mean absolute SHAP |
|---|---:|
| qrs_width_proxy_ms | 0.17318 |
| qrs_rms | 0.09706 |
| st_segment_range | 0.09140 |
| rr_interval_ms | 0.05117 |
| p_wave_range | 0.03977 |
| p_wave_rms | 0.03747 |
| rr_change_ms | 0.03330 |
| t_wave_range | 0.03178 |
| st_segment_slope | 0.02302 |
| signal_noise_proxy | 0.01969 |
| t_wave_mean | 0.01900 |
| t_wave_rms | 0.01398 |
| qrs_range | 0.01134 |
| st_segment_mean | 0.00762 |
| p_wave_mean | 0.00644 |

## Limits

MIT-BIH is a small historical ECG dataset with limited demographic metadata. The cohort audit uses available age and sex fields. Fixed P/QRS/ST/T windows are approximate signal features, not clinically validated delineation. SHAP and LIME describe model behavior, not causality or clinical reasoning. See `docs/ETHICS_NOTES.md` for the FDA pulse-oximeter case study and references.
