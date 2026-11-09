#!/usr/bin/env python3
"""PulseFair: real ECG beat classification with SHAP, LIME, and Fairlearn."""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT / ".local_packages"
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
import wfdb
from fairlearn.metrics import (MetricFrame, count, demographic_parity_difference,
                               equalized_odds_difference, false_negative_rate, false_positive_rate,
                               selection_rate, true_positive_rate)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = ROOT / "data" / "mitdb"
OUTPUT_DIR = ROOT / "output"
BEAT_LABELS = {"N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q"}
NON_BEAT = {"+", "[", "]", "!", "|", "~", "x", "s", "T", "*", "D", "=", "?"}
FEATURES = [
    "p_wave_mean", "p_wave_range", "p_wave_rms",
    "qrs_width_proxy_ms", "qrs_range", "qrs_rms",
    "st_segment_mean", "st_segment_range", "st_segment_slope",
    "t_wave_mean", "t_wave_range", "t_wave_rms",
    "rr_interval_ms", "rr_change_ms", "signal_noise_proxy",
]
METRICS = {"accuracy": accuracy_score, "fpr": false_positive_rate,
           "fnr": false_negative_rate, "tpr": true_positive_rate,
           "selection_rate": selection_rate, "n_beats": count}


def download_if_needed() -> None:
    """Download MIT-BIH records and beat annotations once, into ignored data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if (DATA_DIR / "100.hea").exists() and (DATA_DIR / "100.atr").exists():
        return
    print("Downloading MIT-BIH Arrhythmia Database from PhysioNet (about 110 MB)...")
    wfdb.dl_database("mitdb", str(DATA_DIR), records="all", annotators=["atr"], keep_subdirs=False)


def subject_info(record_id: str) -> tuple[str, int | None]:
    """Read available age and sex from WFDB header comments; missing stays missing."""
    header = DATA_DIR / f"{record_id}.hea"
    text = header.read_text(encoding="latin-1", errors="replace") if header.exists() else ""
    for line in text.splitlines():
        match = re.match(r"\s*#\s*(\d{1,3})\s+([MF])(?:\s|$)", line, re.I)
        if match:
            return match.group(2).upper(), int(match.group(1))
    return "Unknown", None


def summarize_segment(segment: np.ndarray, fs: int) -> dict[str, float]:
    """Extract illustrative P/QRS/ST/T windows around annotated R peaks.

    These fixed windows are educational signal features, not clinical fiducial
    delineation or validated measurements of ECG intervals.
    """
    t = np.arange(-int(0.20 * fs), int(0.36 * fs)) / fs
    baseline = float(np.median(segment[(t >= -0.20) & (t < -0.12)]))
    centered = segment - baseline
    windows = {
        "p": centered[(t >= -0.20) & (t < -0.10)],
        "qrs": centered[(t >= -0.08) & (t < 0.08)],
        "st": centered[(t >= 0.08) & (t < 0.18)],
        "t": centered[(t >= 0.18) & (t < 0.36)],
    }
    def stats(values: np.ndarray) -> tuple[float, float, float]:
        return float(np.mean(values)), float(np.ptp(values)), float(np.sqrt(np.mean(values ** 2)))
    pmean, prange, prms = stats(windows["p"])
    qmean, qrange, qrms = stats(windows["qrs"])
    stmean, strange, _ = stats(windows["st"])
    tmean, trange, trms = stats(windows["t"])
    qrs = windows["qrs"]
    threshold = max(float(np.max(np.abs(qrs))) * 0.20, 0.02)
    width_proxy = float(np.sum(np.abs(qrs) >= threshold) / fs * 1000)
    st = windows["st"]
    st_slope = float(np.polyfit(np.arange(len(st)) / fs, st, 1)[0]) if len(st) > 1 else 0.0
    noise_proxy = float(np.std(np.diff(centered))) if len(centered) > 1 else 0.0
    return {"p_wave_mean": pmean, "p_wave_range": prange, "p_wave_rms": prms,
            "qrs_width_proxy_ms": width_proxy, "qrs_range": qrange, "qrs_rms": qrms,
            "st_segment_mean": stmean, "st_segment_range": strange, "st_segment_slope": st_slope,
            "t_wave_mean": tmean, "t_wave_range": trange, "t_wave_rms": trms,
            "signal_noise_proxy": noise_proxy}


def build_dataset() -> tuple[pd.DataFrame, dict[str, dict]]:
    download_if_needed()
    records = sorted(p.stem for p in DATA_DIR.glob("*.hea") if p.stem.isdigit())
    if not records:
        raise RuntimeError(f"No MIT-BIH headers found in {DATA_DIR}; see README for download help.")
    record_metadata = {}
    records_and_subjects = []
    for record_id in records:
        sex, age = subject_info(record_id)
        # Records 201 and 202 are two recordings from the same person.
        subject_id = "201_202" if record_id in {"201", "202"} else record_id
        age_band = "Unknown" if age is None else ("Under 60" if age < 60 else "60 and over")
        record_metadata[record_id] = {"sex": sex, "age": age, "age_band": age_band, "subject_id": subject_id}
        records_and_subjects.append((record_id, subject_id))

    # Assign whole subjects to splits before extracting beats to avoid patient leakage.
    unique_subjects = sorted({subject for _, subject in records_and_subjects})
    ids = np.asarray(unique_subjects)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    trainval_idx, test_idx = next(splitter.split(ids, groups=ids))
    trainval_subjects, test_subjects = set(ids[trainval_idx]), set(ids[test_idx])
    trainval_ids = np.asarray(sorted(trainval_subjects))
    split2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)
    train_idx, val_idx = next(split2.split(trainval_ids, groups=trainval_ids))
    train_subjects, val_subjects = set(trainval_ids[train_idx]), set(trainval_ids[val_idx])
    subject_split = {**{s: "train" for s in train_subjects}, **{s: "validation" for s in val_subjects},
                     **{s: "test" for s in test_subjects}}

    rows, sample_waveforms = [], {}
    fs = 360
    left, right = int(0.20 * fs), int(0.36 * fs)
    for record_id in records:
        base = str(DATA_DIR / record_id)
        record = wfdb.rdrecord(base, channels=[0])
        signal = np.asarray(record.p_signal[:, 0], dtype=float)
        fs = int(record.fs)
        left, right = int(0.20 * fs), int(0.36 * fs)
        annotation = wfdb.rdann(base, "atr")
        beat_symbols = [(int(s), sym) for s, sym in zip(annotation.sample, annotation.symbol) if sym in BEAT_LABELS]
        intervals = np.diff([s for s, _ in beat_symbols], prepend=beat_symbols[0][0]) / fs if beat_symbols else []
        meta = record_metadata[record_id]
        previous_rr = None
        for beat_index, (sample, symbol) in enumerate(beat_symbols):
            if sample - left < 0 or sample + right >= len(signal):
                continue
            segment = signal[sample - left: sample + right]
            feats = summarize_segment(segment, fs)
            rr_ms = float(intervals[beat_index] * 1000) if beat_index < len(intervals) and beat_index > 0 else np.nan
            feats["rr_interval_ms"] = rr_ms
            feats["rr_change_ms"] = float(rr_ms - previous_rr) if np.isfinite(rr_ms) and previous_rr is not None else 0.0
            if np.isfinite(rr_ms):
                previous_rr = rr_ms
            row_id = f"{record_id}:{sample}"
            rows.append({"sample_id": row_id, "record_id": record_id, "subject_id": meta["subject_id"],
                         "split": subject_split[meta["subject_id"]], "sex": meta["sex"],
                         "age": meta["age"], "age_band": meta["age_band"],
                         "annotation": symbol, "target": int(symbol != "N"), **feats})
            if subject_split[meta["subject_id"]] == "test" and len(sample_waveforms) < 1200:
                sample_waveforms[row_id] = segment.tolist()
    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    # Fill the first beat's unavailable RR interval from the training median only.
    train_median = frame.loc[frame.split == "train", "rr_interval_ms"].median()
    frame["rr_interval_ms"] = frame["rr_interval_ms"].fillna(train_median)
    frame["rr_change_ms"] = frame["rr_change_ms"].fillna(0.0)
    return frame, sample_waveforms


def metric_table(y: pd.Series, pred: np.ndarray, group: pd.Series, subjects: pd.Series) -> dict:
    frame = MetricFrame(metrics=METRICS, y_true=y, y_pred=pred, sensitive_features=group)
    subject_counts = pd.DataFrame({"group": np.asarray(group).astype(str),
                                   "subject": np.asarray(subjects).astype(str)}).groupby("group")["subject"].nunique()
    result = {}
    for cohort, row in frame.by_group.iterrows():
        result[str(cohort)] = {name: int(value) if name == "n_beats" else float(value)
                               for name, value in row.items()}
        result[str(cohort)]["n_subjects"] = int(subject_counts[str(cohort)])
    y_array = np.asarray(y)
    result["_summary"] = {
        "demographic_parity_difference": float(demographic_parity_difference(y_array, pred, sensitive_features=np.asarray(group))),
        "equalized_odds_difference": float(equalized_odds_difference(y_array, pred, sensitive_features=np.asarray(group))),
        "fpr_gap": float(frame.by_group["fpr"].max() - frame.by_group["fpr"].min()),
    }
    return result


def fit_fpr_optimizer(model, X_val, y_val, sensitive):
    groups = pd.Series(sensitive).dropna().unique()
    if len(groups) < 2:
        return None
    optimizer = ThresholdOptimizer(estimator=model, constraints="false_positive_rate_parity",
                                   objective="balanced_accuracy_score", grid_size=100,
                                   prefit=True, predict_method="predict_proba")
    optimizer.fit(X_val, y_val, sensitive_features=sensitive)
    return optimizer


def normalize_shap(explainer, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        raw = raw[1]
    raw = np.asarray(raw)
    if raw.ndim == 3:
        raw = raw[:, :, 1]
    base = np.asarray(explainer.expected_value)
    base_value = float(base.reshape(-1)[1] if base.size > 1 else base.reshape(-1)[0])
    return raw.astype(float), base_value


def make_wave_svg(samples: list[float], fs: int) -> str:
    values = np.asarray(samples, dtype=float)
    width, height = 760, 190
    x = np.linspace(0, width, len(values))
    peak = max(float(np.max(np.abs(values))), 1e-6)
    y = height / 2 - values / peak * 68
    points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    # The colored regions are approximate teaching windows around the annotated beat.
    ms_to_x = lambda ms: width * (ms + 200) / 560
    bands = [("P", -200, -100, "#dceafa"), ("QRS", -80, 80, "#f8dfd8"),
             ("ST", 80, 180, "#e0f2eb"), ("T", 180, 360, "#f3e8fb")]
    background = "".join(f'<rect x="{ms_to_x(a):.1f}" y="0" width="{ms_to_x(b)-ms_to_x(a):.1f}" height="{height}" fill="{c}"/><text x="{(ms_to_x(a)+ms_to_x(b))/2:.1f}" y="18" text-anchor="middle" font-size="12" fill="#445">{name}</text>' for name, a, b, c in bands)
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Annotated ECG beat waveform">{background}<line x1="0" y1="{height/2}" x2="{width}" y2="{height/2}" stroke="#9aa8b8"/><polyline points="{points}" fill="none" stroke="#173f68" stroke-width="2"/></svg>'


def safe_json(payload: dict) -> str:
    return json.dumps(payload, allow_nan=False).replace("</", "<\\/")


def write_dashboard(path: Path, payload: dict) -> None:
    data = safe_json(payload)
    page = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PulseFair · ECG ethics audit</title>
<style>:root{--ink:#192536;--muted:#5b6b7c;--paper:#f5f7fb;--card:#fff;--blue:#3569ad;--teal:#238875;--orange:#d58436;--line:#dce3ed}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,sans-serif}.wrap{max-width:1120px;margin:auto;padding:28px 20px 55px}.hero{background:#183653;color:white;padding:28px;border-radius:18px}.hero h1{margin:4px 0;font-size:34px}.hero p{max-width:850px;color:#e1ebf5}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.card{background:white;border:1px solid var(--line);border-radius:14px;padding:19px;margin:15px 0}.card h2{font-size:19px;margin:0 0 10px}.muted{color:var(--muted)}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:11px}.bars{display:grid;grid-template-columns:180px 1fr 58px;gap:8px;align-items:center;margin:10px 0}.track{height:11px;background:#e9eef5;border-radius:8px;overflow:hidden}.fill{height:100%;background:var(--blue)}.fill.teal{background:var(--teal)}select{padding:9px;border:1px solid var(--line);border-radius:8px;background:white}.chip{display:inline-block;padding:3px 9px;border-radius:20px;background:#eaf1fb;color:#254f80;font-size:12px}.score{font-size:27px;font-weight:700}.contrib{display:grid;grid-template-columns:175px 1fr 64px;align-items:center;gap:8px;margin:7px 0}.axis{height:10px;background:#eef1f5;border-radius:5px;position:relative}.zero{position:absolute;left:50%;height:100%;width:1px;background:#9aa8b8}.pos,.neg{position:absolute;height:100%;border-radius:5px}.pos{left:50%;background:var(--teal)}.neg{right:50%;background:var(--orange)}svg{width:100%;height:auto;border:1px solid var(--line);border-radius:9px}.tabs{display:flex;gap:8px;margin:10px 0}.tabs button{border:1px solid var(--line);background:#f5f7fb;padding:8px 13px;border-radius:8px;cursor:pointer}.tabs button.active{background:#183653;color:white}footer{font-size:13px;color:var(--muted);margin-top:25px}@media(max-width:700px){.grid{grid-template-columns:1fr}.bars{grid-template-columns:130px 1fr 48px}.contrib{grid-template-columns:130px 1fr 55px}}.kicker{letter-spacing:.12em;text-transform:uppercase;font-size:12px;color:#b9d5f6}.hero h1{font-size:36px}.hero p{font-size:16px}.intro h2{font-size:24px}.method{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:16px}.step{background:#f5f7fb;border:1px solid var(--line);border-radius:10px;padding:13px}.step b{display:block;margin-bottom:4px;color:#254f80}.insight{border-left:4px solid var(--teal)}.gap-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.gap-box{background:#f5f7fb;border-radius:9px;padding:12px}.gap-box b{display:block;margin-bottom:4px}@media(max-width:700px){.method,.gap-grid{grid-template-columns:1fr}}</style>
<div class="wrap"><header class="hero"><div class="kicker">Ethics in AI · Explainability + fairness</div><h1>PulseFair</h1><p>When an ECG model flags a heartbeat, what did it notice—and do its errors affect groups differently?</p></header>
<section class="card intro"><div class="kicker" style="color:#3569ad">The project in three steps</div><div class="method"><div class="step"><b>1 · Real ECG data</b>Annotated heartbeat recordings with age and sex metadata.</div><div class="step"><b>2 · Explain the model</b>A Random Forest uses signal features; SHAP and LIME show which features influence a beat score.</div><div class="step"><b>3 · Audit fairness</b>Compare group errors, then try an FPR-parity adjustment fitted on validation data.</div></div></section>
<section class="card insight"><h2>Key finding</h2><div id="insight"></div></section>
<div id="dataset" class="muted"></div><div class="grid"><section class="card"><h2>Sex-group results · shared threshold</h2><div id="sexbase"></div></section><section class="card"><h2>Sex-group results · FPR-parity adjustment</h2><div id="sexadj"></div></section></div>
<section class="card"><h2>Age-group results</h2><div id="ageaudit"></div><div id="deltas"></div></section>
<section class="card"><h2>Global SHAP importance</h2><p class="muted">Mean absolute contribution across held-out beats. Model influence is not clinical importance.</p><div id="importance"></div></section>
<section class="card"><h2>Inspect a held-out beat</h2><select id="sample"></select><div id="wave"></div><div class="tabs"><button class="active" data-tab="shap">SHAP</button><button data-tab="lime">LIME</button></div><div id="explanation"></div></section>
<section class="card"><h2>COVID-19 case study: pulse oximetry</h2><p>A 2020 hospital-cohort study found occult hypoxemia missed by pulse oximetry occurred nearly three times as often in Black patients as White patients. The FDA lists skin pigmentation among factors that can affect readings and later proposed broader performance testing. This is adjacent optical-sensor evidence; it does not imply the ECG dataset or this model has the same disparity. Race is not a direct skin-tone measurement.</p><a href="https://doi.org/10.1056/NEJMc2029240">Sjoding et al., NEJM 2020</a> · <a href="https://www.fda.gov/medical-devices/products-and-medical-procedures/pulse-oximeters">FDA: Pulse Oximeters</a> · <a href="https://www.fda.gov/regulatory-information/search-fda-guidance-documents/pulse-oximeters-medical-purposes-non-clinical-and-clinical-performance-testing-labeling-and">FDA draft performance guidance (2025)</a></section>
<footer>ECG source: MIT-BIH Arrhythmia Database, Moody &amp; Mark (2005), DOI 10.13026/C2F305. Clinical context: sources linked in the case study below.</footer></div><script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),pc=x=>(100*x).toFixed(1)+'%';function table(obj){let gs=Object.keys(obj).filter(k=>!k.startsWith('_'));return '<table><tr><th>Group</th><th>Subjects</th><th>Beats</th><th>Accuracy</th><th>FPR</th><th>FNR</th><th>TPR</th><th>Flag rate</th></tr>'+gs.map(g=>{let m=obj[g];return `<tr><td>${g}</td><td>${m.n_subjects}</td><td>${m.n_beats}</td><td>${pc(m.accuracy)}</td><td>${pc(m.fpr)}</td><td>${pc(m.fnr)}</td><td>${pc(m.tpr)}</td><td>${pc(m.selection_rate)}</td></tr>`}).join('')+'</table>'}document.getElementById('dataset').textContent=`${D.n_beats.toLocaleString()} annotated beats · ${D.n_records} records · ${D.n_subjects} subjects · train/validation/test split by subject`;const sexAudit=D.audit.sex,ageAudit=D.audit.age;document.getElementById('insight').innerHTML=`<p>The FPR-parity adjustment nearly closed the validation gaps, but the gaps grew on held-out subjects. In this small split, the change did not generalize.</p><div class="gap-grid"><div class="gap-box"><b>Sex-group FPR gap</b>Validation ${pc(sexAudit.validation_base._summary.fpr_gap)} → ${pc(sexAudit.validation_adjusted._summary.fpr_gap)}<br>Held-out ${pc(sexAudit.base._summary.fpr_gap)} → ${pc(sexAudit.adjusted._summary.fpr_gap)}</div><div class="gap-box"><b>Age-group FPR gap</b>Validation ${pc(ageAudit.validation_base._summary.fpr_gap)} → ${pc(ageAudit.validation_adjusted._summary.fpr_gap)}<br>Held-out ${pc(ageAudit.base._summary.fpr_gap)} → ${pc(ageAudit.adjusted._summary.fpr_gap)}</div></div><p class="muted">The held-out cohort contains only a few people per group. Treat these results as a demonstration of why fairness adjustments need new-subject validation, not as evidence that disparity was solved.</p>`;document.getElementById('sexbase').innerHTML=table(D.audit.sex.base);document.getElementById('sexadj').innerHTML=table(D.audit.sex.adjusted);document.getElementById('ageaudit').innerHTML='<h3>Shared probability threshold 0.50</h3>'+table(D.audit.age.base)+'<h3>FPR parity post-processing</h3>'+table(D.audit.age.adjusted);document.getElementById('deltas').innerHTML=Object.entries(D.audit).map(([k,v])=>`<p><b>${k}:</b> validation FPR gap ${pc(v.validation_base._summary.fpr_gap)} → ${pc(v.validation_adjusted._summary.fpr_gap)}; held-out FPR gap ${pc(v.base._summary.fpr_gap)} → ${pc(v.adjusted._summary.fpr_gap)} · unknown metadata omitted ${v.unknown_test_subjects} subjects / ${v.unknown_test_beats} beats</p>`).join('');document.getElementById('importance').innerHTML=D.importance.map(([k,v])=>`<div class="bars"><span>${k.replaceAll('_',' ')}</span><div class="track"><div class="fill" style="width:${Math.max(1,v/D.importance[0][1]*100)}%"></div></div><b>${v.toFixed(4)}</b></div>`).join('');
const sel=document.getElementById('sample');D.examples.forEach((e,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${e.id} · ${e.sex} · ${e.age_band} · ${e.score.toFixed(2)} · ${e.label}`;sel.appendChild(o)});let activeTab='shap';function render(){let e=D.examples[sel.value];document.getElementById('wave').innerHTML=`<p><span class="chip">${e.sex}</span> <span class="chip">${e.age_band}</span> <span class="chip">${e.record_id}</span></p>${e.svg}<p class="score">${(e.score*100).toFixed(1)}% <span class="muted" style="font-size:14px">abnormal-beat model score · ${e.prediction?'flagged':'not flagged'} · reference ${e.label}</span></p>`;let obj=e[activeTab];document.getElementById('explanation').innerHTML=`<p class="muted">${activeTab==='shap'?'TreeSHAP contributions add to the model output from a baseline.':'LIME fits a local interpretable surrogate around this example; contributions are approximate.'}</p>`+Object.entries(obj).map(([k,v])=>{let w=Math.min(49,Math.abs(v)*300);return `<div class="contrib"><span>${k.replaceAll('_',' ')}</span><div class="axis"><i class="zero"></i><i class="${v>=0?'pos':'neg'}" style="width:${w}%"></i></div><b>${v>=0?'+':''}${v.toFixed(4)}</b></div>`}).join('')}sel.addEventListener('change',render);document.querySelectorAll('.tabs button').forEach(b=>b.addEventListener('click',()=>{activeTab=b.dataset.tab;document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('active',x===b));render()}));render();
</script></html>'''.replace("__DATA__", data)
    path.write_text(page, encoding="utf-8")


def main() -> None:
    global DATA_DIR
    parser = argparse.ArgumentParser(description="Train an explainable ECG beat model and fairness audit")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    DATA_DIR = args.data_dir.resolve()
    frame, waveforms = build_dataset()
    args.out.mkdir(parents=True, exist_ok=True)
    X_train, y_train = frame.loc[frame.split == "train", FEATURES], frame.loc[frame.split == "train", "target"]
    X_val, y_val = frame.loc[frame.split == "validation", FEATURES], frame.loc[frame.split == "validation", "target"]
    test = frame.loc[frame.split == "test"].reset_index(drop=True)
    X_test, y_test = test[FEATURES], test["target"]
    model = RandomForestClassifier(n_estimators=180, max_depth=12, min_samples_leaf=8,
                                   class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    base_pred = (probability >= 0.50).astype(int)
    val_probability = model.predict_proba(X_val)[:, 1]
    val_base_pred = (val_probability >= 0.50).astype(int)

    audit_results = {}
    for name in ("sex", "age_band"):
        group = test[name].fillna("Unknown").astype(str)
        known_test = group != "Unknown"
        eval_group = group.loc[known_test]
        test_subjects = test.loc[known_test, "subject_id"]
        base = metric_table(y_test.loc[known_test], base_pred[known_test.to_numpy()], eval_group, test_subjects)
        validation_group = frame.loc[frame.split == "validation", name].fillna("Unknown").astype(str)
        known_val = validation_group != "Unknown"
        val_group = validation_group.loc[known_val]
        val_subjects = frame.loc[frame.split == "validation", "subject_id"].loc[known_val]
        val_base = metric_table(y_val.loc[known_val], val_base_pred[known_val.to_numpy()], val_group, val_subjects)
        optimizer = fit_fpr_optimizer(model, X_val.loc[known_val], y_val.loc[known_val], val_group)
        if optimizer is not None:
            pred = optimizer.predict(X_test.loc[known_test], sensitive_features=eval_group, random_state=42)
            adjusted = metric_table(y_test.loc[known_test], pred, eval_group, test_subjects)
            val_pred = optimizer.predict(X_val.loc[known_val], sensitive_features=val_group, random_state=42)
            val_adjusted = metric_table(y_val.loc[known_val], val_pred, val_group, val_subjects)
        else:
            adjusted = base
            val_adjusted = val_base
        audit_results[name] = {"base": base, "adjusted": adjusted,
                               "validation_base": val_base, "validation_adjusted": val_adjusted,
                               "unknown_test_beats": int((~known_test).sum()),
                               "unknown_test_subjects": int(test.loc[~known_test, "subject_id"].nunique())}

    # SHAP on a bounded sample of test beats keeps the first run manageable.
    explain_count = min(600, len(X_test))
    explain_idx = np.linspace(0, len(X_test) - 1, explain_count, dtype=int)
    X_explain = X_test.iloc[explain_idx]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    base_values = np.asarray(explainer.expected_value)
    base_value = float(base_values.reshape(-1)[1] if base_values.size > 1 else base_values.reshape(-1)[0])
    importance = {feature: float(np.mean(np.abs(shap_values[:, j]))) for j, feature in enumerate(FEATURES)}
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.to_numpy(), feature_names=FEATURES, class_names=["normal", "abnormal"],
        mode="classification", discretize_continuous=True, random_state=42)
    local_positions = list(range(min(6, len(test))))
    local_values, _ = normalize_shap(explainer, X_test.iloc[local_positions])
    examples = []
    for i, pos in enumerate(local_positions):
        row, x = test.iloc[pos], X_test.iloc[[pos]]
        lime_predict = lambda values: model.predict_proba(pd.DataFrame(values, columns=FEATURES))
        raw_lime = lime_explainer.explain_instance(X_test.iloc[pos].to_numpy(), lime_predict,
                                                   num_features=10, num_samples=1000, labels=(1,))
        lime_values = {name: float(value) for name, value in raw_lime.as_list(label=1)}
        sv = {name: float(local_values[i, j]) for j, name in enumerate(FEATURES)}
        record_id = str(row.record_id)
        examples.append({"id": row.sample_id, "record_id": record_id, "sex": str(row.sex),
                         "age_band": str(row.age_band), "score": float(probability[pos]),
                         "prediction": bool(base_pred[pos]), "label": "abnormal" if row.target else "normal",
                         "base": base_value, "shap": sv, "lime": lime_values,
                         "svg": make_wave_svg(waveforms.get(row.sample_id, []), 360)})

    frame.to_csv(args.out / "mitbih_beats.csv", index=False, float_format="%.6g")
    dashboard_audit = {"sex": audit_results["sex"], "age": audit_results["age_band"]}
    payload = {"n_beats": len(frame), "n_records": int(frame.record_id.nunique()),
               "n_subjects": int(frame.subject_id.nunique()), "audit": dashboard_audit,
               "importance": importance, "examples": examples}
    write_dashboard(args.out / "index.html", payload)
    output_lines = ["# PulseFair · real-data results", "",
                    "> Educational demonstration; not a medical device or clinical model.", "",
                    f"- Source: PhysioNet MIT-BIH Arrhythmia Database ({payload['n_records']} records, {payload['n_subjects']} subjects).",
                    f"- Beat windows: {payload['n_beats']:,}; subject-level train/validation/test split to avoid the same person appearing across splits.",
                    "- Task: distinguish normal (`N`) beats from other beat annotations. This is not an atrial-fibrillation diagnosis.",
                    "- Cohorts: sex and broad age bands parsed from available headers. Race and skin pigmentation are not present.",
                    "- Group metrics are beat-weighted; subject counts are shown separately because beats from one person are correlated.",
                    "- FPR parity is calibrated on validation data and evaluated on held-out subjects; Fairlearn may randomize post-processed predictions.", ""]
    for name, title in (("sex", "Sex cohorts"), ("age_band", "Age cohorts")):
        output_lines += [f"## {title}", "", "| Policy | Group | Subjects | Beats | Accuracy | FPR | FNR | TPR | Selection rate |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for policy in ("base", "adjusted"):
            for group_name, m in audit_results[name][policy].items():
                if group_name == "_summary": continue
                output_lines.append(f"| {policy} | {group_name} | {m['n_subjects']} | {m['n_beats']} | {m['accuracy']:.3f} | {m['fpr']:.3f} | {m['fnr']:.3f} | {m['tpr']:.3f} | {m['selection_rate']:.3f} |")
        output_lines.append(f"Unknown cohort metadata, omitted from this audit: {audit_results[name]['unknown_test_subjects']} test subjects / {audit_results[name]['unknown_test_beats']} test beats.\n")
    output_lines += ["## Fairness gaps", "", "| Attribute | Policy | FPR gap | Demographic-parity difference | Equalized-odds difference |", "|---|---|---:|---:|---:|"]
    for name in ("sex", "age_band"):
        policies = (("validation_base", "Validation · shared threshold"),
                    ("validation_adjusted", "Validation · FPR parity"),
                    ("base", "Test · shared threshold"),
                    ("adjusted", "Test · FPR parity"))
        for policy, policy_label in policies:
            summary = audit_results[name][policy]["_summary"]
            output_lines.append(f"| {name} | {policy_label} | {summary['fpr_gap']:.3f} | {summary['demographic_parity_difference']:.3f} | {summary['equalized_odds_difference']:.3f} |")
    output_lines += ["", "## SHAP global feature importance", "", "Mean absolute contribution across a sample of held-out beats; model importance is not medical importance.", "", "| Feature | Mean absolute SHAP |", "|---|---:|"]
    output_lines.extend(f"| {name} | {value:.5f} |" for name, value in importance)
    output_lines += ["", "## Limits", "", "MIT-BIH is a small historical ECG dataset with limited demographic metadata. No skin pigmentation is supplied, so this project does not test the screenshot's specific skin-tone disparity hypothesis. Fixed P/QRS/ST/T windows are rough signal features, not clinically validated delineation. SHAP and LIME explain model behavior, not causality or clinical reasoning. See `docs/ETHICS_NOTES.md` for the FDA pulse-oximeter case study and references.", ""]
    (args.out / "report.md").write_text("\n".join(output_lines), encoding="utf-8")
    print(f"Built {payload['n_beats']:,} annotated beat examples from {payload['n_records']} ECG records")
    print(f"Generated dashboard, report, and beat-level CSV in {args.out.resolve()}")
    print("Synthetic skin-tone cohorts were not invented; MIT-BIH does not contain pigmentation labels.")


if __name__ == "__main__":
    main()
