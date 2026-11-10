# Teammate setup: from a fresh clone to the dashboard

This guide assumes a new computer with no Python packages and no PhysioNet data downloaded. You do not need Jupyter, a GPU, an IDE, or a GitHub account to run the project.

## 1. Get the project folder

Either clone the GitHub repository:

```powershell
git clone https://github.com/arnavg230/pulsefair.git
cd pulsefair
```

Or download the repository ZIP from GitHub, extract it, and open PowerShell in the extracted project folder (the folder containing `README.md`).

## 2. Check Python

PulseFair supports Python 3.10–3.12. Check the version:

```powershell
python --version
```

If `python` is not recognized on Windows, install Python 3.12 from [python.org](https://www.python.org/downloads/) and enable the installer option that adds Python to PATH. Close and reopen PowerShell, then check again.

## 3. Create an isolated environment and install packages

These commands use the environment's Python directly, so PowerShell does not need permission to run an activation script:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The first install needs internet access and may take a few minutes because the scientific Python libraries include compiled packages. On macOS/Linux, use:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

If creating a virtual environment fails, the fallback is to install into your regular Python with `python -m pip install -r requirements.txt` (use `python3` on macOS/Linux). A virtual environment is preferable because it keeps this project's packages separate from other work.

## 4. Run PulseFair

On Windows:

```powershell
.venv\Scripts\python.exe src\pulsefair.py
```

On macOS/Linux:

```bash
.venv/bin/python src/pulsefair.py
```

On the first run, WFDB downloads the MIT-BIH recordings and beat annotations from PhysioNet into `data/mitdb/` (about 95 MB). This can take a few minutes depending on the connection. The script then extracts features, trains the model, calculates the audits, and writes the outputs. Keep internet access available for this first run.

The dataset is not bundled in the Git repository. The `data/mitdb/` folder is ignored by Git; teammates each download their own copy by running the program once. Later runs reuse the local data cache.

## 5. View the results

The script prints the output folder when it finishes. Open the dashboard:

```powershell
start output\index.html
```

On macOS, use `open output/index.html`; on Linux, use `xdg-open output/index.html`.

The dashboard is a local HTML file. Its tables, waveform, and explanations work offline after generation. The FDA and research links in the case-study section need internet access.

## Files to read

1. Start with `README.md` for the project summary and safety boundary.
2. Read `docs/PROJECT_GUIDE.md` for the dataset, model, feature windows, fairness metrics, and code tour.
3. Read `docs/ETHICS_NOTES.md` for limitations and class discussion prompts.
4. Read `docs/DATA_SOURCES.md` for the dataset citation, data license, and pulse-oximeter references.
5. Review `output/report.md`, then explore `output/index.html`.

## Generated outputs

- `output/index.html`: interactive cohort tables, fairness-gap comparisons, global SHAP feature summary, a waveform with approximate P/QRS/ST/T windows, and SHAP/LIME explanations for selected test beats.
- `output/report.md`: text tables for held-out cohort performance and fairness gaps on both validation and test data. Cohort tables show distinct subject count beside beat count; metrics remain beat-weighted.
- `output/mitbih_beats.csv`: one row per extracted beat. `sample_id` identifies a record and annotation sample; `record_id` is the source recording; `subject_id` keeps repeated records from one person together; `split` is train/validation/test; `sex`, `age`, and `age_band` are audit metadata; `annotation` is the WFDB beat symbol; `target` is 0 for `N` and 1 for any other included beat symbol. Remaining columns are extracted waveform features.

The waveform CSV contains derived features, not raw ECG samples. The original WFDB signals remain in `data/mitdb/` under their own license.

## Command-line options

The script accepts optional output and data folder paths:

```powershell
.venv\Scripts\python.exe src\pulsefair.py --out output --data-dir data\mitdb
```

Defaults already point to those folders in the repository, so most teammates can run the simple command in step 4.

## Common problems

### `No module named ...`

Make sure the install command used the same `.venv` Python as the run command. Re-run `.venv\Scripts\python.exe -m pip install -r requirements.txt` from the project folder.

### PhysioNet download fails

Check internet access, then rerun the script. WFDB can resume or reuse files in `data/mitdb/`. If the network is blocked on your campus, open the MIT-BIH access page linked in `docs/DATA_SOURCES.md` on an allowed connection and ask the team to transfer the downloaded database folder; place its record files under `data/mitdb/`. The script checks for all 48 recordings before training and reports an incomplete download instead of silently producing results from a partial dataset.

### Dashboard is not refreshed

The browser may still show the previous file. Rerun the script, then reload `output/index.html` in the browser.

### Want to publish a copy of the repo

Create an empty GitHub repository, then add it as the `origin` remote and push the `main` branch. Do not add `data/mitdb/` or `.local_packages/`; `.gitignore` intentionally excludes them.

## Reproducibility note

The subject split and model seeds are fixed in the script. Fairlearn's threshold optimizer is calibrated on validation subjects and evaluated on held-out subjects. The grouped beat observations are correlated within each person, the database has only 47 subjects, and the test FPR gap may differ from the validation gap. Treat the results as a classroom demonstration, not population evidence.
