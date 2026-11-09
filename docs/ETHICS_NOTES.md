# Ethics notes and discussion

## Principle in the implementation

The project combines explainability with fairness auditing. SHAP and LIME expose how a model arrives at example scores; group metrics ask whether people in different cohorts experience different errors. Neither method alone establishes that a model is safe or fair.

## What the results can support

- A demonstration of beat-level ECG classification using real annotated signals.
- An exploratory comparison of error rates across the limited age and sex metadata available in the database.
- A visible trade-off when a validation-fitted group threshold policy targets FPR parity.
- A comparison of local SHAP and LIME explanations for one example.

## What the results cannot support

- A claim about clinical ECG devices or patient outcomes.
- An atrial-fibrillation diagnosis: the task is `N` versus other beat annotation symbols.
- Population-level conclusions from 47 subjects, especially when beat observations from one subject are correlated.
- Causal claims from SHAP/LIME feature attribution.
- A claim that equal FPR is the only or correct fairness goal.

## Pulse-oximeter case study

During the COVID-19 pandemic, a 2020 study reported that occult hypoxemia missed by pulse oximetry occurred nearly three times as often among Black as White patients in two hospital cohorts. The FDA's safety materials describe skin pigmentation among factors that can affect readings, and its January 2025 draft guidance proposes performance-evaluation recommendations across skin tones. This case is relevant because pulse oximeters use optical sensing. It provides context for discussing healthcare sensor fairness alongside the project’s separate ECG cohort audit. See [DATA_SOURCES.md](DATA_SOURCES.md) for sources.

## Questions for classmates

1. Should error parity be evaluated at the beat level, patient level, or both?
2. Which error matters most here: false alarms or missed abnormal beats? Who should set that priority?
3. What trade-offs appear when Fairlearn adjusts predictions to target equal FPR?
4. Why might SHAP and LIME assign different feature contributions to the same beat?
5. How should small subject counts and repeated beats affect confidence in group fairness metrics?
6. What validation, clinical oversight, privacy protections, and regulatory evaluation would be required before real-world use?
