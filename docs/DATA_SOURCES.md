# Data and case-study sources

## MIT-BIH Arrhythmia Database

Moody GB, Mark RG. *The MIT-BIH Arrhythmia Database*. PhysioNet. The database contains 48 approximately half-hour two-channel ECG recordings from 47 subjects with beat annotations. The two recordings 201 and 202 are from one subject, so this project keeps them in one data split.

- [PhysioNet database overview and access](https://physionet.org/content/mitdb/1.0.0/)
- [WFDB Python download API](https://wfdb.readthedocs.io/en/latest/wfdb.html)
- [PhysioNet MIT-BIH directory and record details](https://physionet.mit.edu/physiobank/database/html/mitdbdir/intro.htm)
- DOI: [10.13026/C2F305](https://doi.org/10.13026/C2F305)
- Data license: Open Data Commons Attribution License v1.0, as stated on the PhysioNet database page.

The project downloads the records on first run rather than bundling the data files in this repository. The public dataset description includes age and sex information, but no skin-pigmentation metadata. The audit uses age bands and sex only; it does not infer or fabricate skin tone.

## FDA pulse-oximeter case study

- [FDA: Pulse Oximeters](https://www.fda.gov/medical-devices/products-and-medical-procedures/pulse-oximeters) — describes accuracy limitations, the 2021 safety communication, and FDA work concerning skin pigmentation.
- [FDA 2025 draft guidance: Pulse Oximeters for Medical Purposes](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/pulse-oximeters-medical-purposes-non-clinical-and-clinical-performance-testing-labeling-and) — draft, non-binding recommendations for performance testing and labeling.

Pulse oximetry is the adjacent real-world example from the project brief. It uses optical sensing and is not the same modality as the ECG dataset used for model training.
