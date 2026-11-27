#!/usr/bin/env python3
"""Offline retrospective ESPN probability calibration research for NCAA men's basketball.

Source: previously archived 500 completed 2026 games with retrieved ESPN Core predictor
probabilities (publication-as-of-tipoff unverified). Educational evaluation ONLY.
No network, DB writes, betting output, or forward-game predictions.
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

REQUIRED = [
    "game_id", "season", "game_date", "home_win", "neutral_site",
    "home_net", "away_net", "espn_home_probability", "competitive", "tipoff_utc",
    "retrieved_at_utc",
]
SPLITS = {"train": ["2025-11", "2025-12", "2026-01"],
          "validation": ["2026-02"],
          "evaluation": ["2026-03", "2026-04"]}
CANDIDATES = ["ESPN (unchanged)", "ESPN + efficiency", "ESPN + efficiency + venue"]
P_CLIP = 1e-5
PENALTY = 10.0
SELECTION_MIN_BRIER_GAIN = 0.001


def load_and_validate(path):
    data = pd.read_csv(path)
    absent = sorted(set(REQUIRED) - set(data.columns))
    if absent:
        raise ValueError("Missing required column(s): " + ", ".join(absent))
    if data.game_id.isna().any() or data.game_id.astype(str).duplicated().any():
        raise ValueError("game_id missing or duplicated")
    if data[REQUIRED].isna().any().any():
        raise ValueError("Core evaluation columns contain missing values")
    if not data.season.eq(2026).all():
        raise ValueError("Expected archived 2026-season games only")
    if not data.home_win.isin([0, 1]).all() or not data.neutral_site.isin([0, 1]).all():
        raise ValueError("Labels and venue flags must be 0 or 1")
    if not data.espn_home_probability.between(0, 1, inclusive="neither").all():
        raise ValueError("ESPN probabilities must be strictly between 0 and 1")
    if not np.isfinite(data[["home_net", "away_net", "espn_home_probability"]].to_numpy(dtype=float)).all():
        raise ValueError("Rating or probability has infinite or invalid value")
    data["game_date"] = pd.to_datetime(data["game_date"], errors="raise")
    data["tipoff_utc"] = pd.to_datetime(data["tipoff_utc"], utc=True, errors="raise")
    data["retrieved_at_utc"] = pd.to_datetime(data["retrieved_at_utc"], utc=True, errors="raise")
    if data["retrieved_at_utc"].le(data["tipoff_utc"]).any():
        raise ValueError("Input must contain retrospectively retrieved, completed-game data only")
    data["month"] = data.game_date.dt.strftime("%Y-%m")
    if not set(data.month.unique()).issubset(set(sum(SPLITS.values(), []))):
        raise ValueError("Unexpected dates outside November 2025 through April 2026")
    data["competitive"] = data.competitive.astype(bool)
    data["net_difference"] = (data.home_net - data.away_net).astype(float)
    data = data.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return data


def design_matrix(d, candidate):
    diff = d.net_difference.to_numpy(dtype=float) / 10.0
    neutral = d.neutral_site.to_numpy(dtype=float)
    ones = np.ones(len(d), dtype=float)
    if candidate == "ESPN (unchanged)":
        return np.zeros((len(d), 0))
    if candidate == "ESPN + efficiency":
        return np.column_stack([ones, diff])
    if candidate == "ESPN + efficiency + venue":
        return np.column_stack([ones, diff, neutral, neutral * diff])
    raise ValueError(f"Unknown model: {candidate}")


def fit_correction(train, candidate, penalty=PENALTY):
    x = design_matrix(train, candidate)
    if not x.shape[1]:
        return np.zeros(0)
    y = train.home_win.to_numpy(dtype=float)
    base = logit(train.espn_home_probability.to_numpy(dtype=float).clip(P_CLIP, 1 - P_CLIP))
    def objective(beta):
        logits = base + x @ beta
        # sum logistic loss + ridge penalty; ESPN unchanged is the zero vector
        loss = np.logaddexp(0.0, logits).sum() - (y * logits).sum()
        pen = 0.5 * penalty * (beta @ beta)
        grad = x.T @ (expit(logits) - y) + penalty * beta
        return loss + pen, grad
    result = minimize(objective, np.zeros(x.shape[1]), method="L-BFGS-B", jac=True)
    if not result.success:
        raise RuntimeError("Fit did not converge: " + result.message)
    return result.x


def predict(d, candidate, beta):
    original = logit(d.espn_home_probability.to_numpy(dtype=float).clip(P_CLIP, 1 - P_CLIP))
    x = design_matrix(d, candidate)
    if x.shape[1] != len(beta):
        raise ValueError("Correction coefficient mismatch")
    return expit(original + x @ beta).clip(P_CLIP, 1 - P_CLIP)


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if not n:
        return {"games": 0, "correct": None, "accuracy_pct": None, "brier": None, "log_loss": None}
    return {"games": int(n), "correct": int(((p >= 0.5).astype(int) == y).sum()),
            "accuracy_pct": round(float(((p >= 0.5).astype(int) == y).mean() * 100), 3),
            "brier": round(float(np.mean((p - y) ** 2)), 6),
            "log_loss": round(float(np.mean(-(y * np.log(p) + (1-y) * np.log1p(-p)))), 6)}


def paired_bootstrap_brier(y, baseline, other, n_boot=5000, seed=2026):
    """Other minus baseline; negative means adjusted model improves Brier."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    errors = (np.asarray(other) - y)**2 - (np.asarray(baseline) - y)**2
    if not len(errors):
        return [None, None]
    idx = rng.integers(0, len(errors), size=(n_boot, len(errors)))
    return [round(float(v), 6) for v in np.quantile(errors[idx].mean(axis=1), [0.025, 0.975])]


def run_study(data):
    parts = {name: data[data.month.isin(months)].copy() for name, months in SPLITS.items()}
    if min(len(part) for part in parts.values()) < 20:
        raise ValueError("Insufficient games in train/validation/evaluation splits")
    fitted = {name: fit_correction(parts["train"], name) for name in CANDIDATES}
    valid_results = []
    for model in CANDIDATES:
        score = metrics(parts["validation"].home_win, predict(parts["validation"], model, fitted[model]))
        valid_results.append({"model": model, **score})
    base_score = next(x for x in valid_results if x["model"] == CANDIDATES[0])["brier"]
    best = min(valid_results, key=lambda x: x["brier"])
    selected = best["model"] if best["model"] != CANDIDATES[0] and base_score - best["brier"] >= SELECTION_MIN_BRIER_GAIN else CANDIDATES[0]
    test = parts["evaluation"]
    predictions = test[["game_id", "game_date", "month", "neutral_site", "competitive", "home_win"]].copy()
    predictions["espn_probability"] = predict(test, CANDIDATES[0], fitted[CANDIDATES[0]])
    predictions["efficiency_corrected_probability"] = predict(test, CANDIDATES[1], fitted[CANDIDATES[1]])
    predictions["efficiency_venue_corrected_probability"] = predict(test, CANDIDATES[2], fitted[CANDIDATES[2]])
    # These are historical outcomes from completed games only, not future predictions.
    validation = pd.DataFrame(valid_results)
    summary = []
    groups = {"all": np.ones(len(test), dtype=bool), "competitive": test.competitive.to_numpy(),
              "other": ~test.competitive.to_numpy(), "neutral": test.neutral_site.to_numpy() == 1,
              "nonneutral": test.neutral_site.to_numpy() == 0}
    for group, mask in groups.items():
        for model, col in zip(CANDIDATES, ["espn_probability", "efficiency_corrected_probability", "efficiency_venue_corrected_probability"]):
            score = metrics(predictions.loc[mask, "home_win"], predictions.loc[mask, col])
            summary.append({"group": group, "model": model, "validation_selected": model == selected, **score})
    overall = pd.DataFrame(summary)
    by = overall[overall.group == "all"].set_index("model")
    eb = predictions.espn_probability.to_numpy()
    yp = test.home_win.to_numpy()
    bootstrap = {model: paired_bootstrap_brier(yp, eb, predict(test, model, fitted[model]))
                 for model in CANDIDATES[1:]}
    return {"validation": validation, "results": overall, "predictions": predictions,
            "selected": selected, "split_counts": {k: len(v) for k, v in parts.items()},
            "coefficients": {k: [round(float(c), 6) for c in v] for k, v in fitted.items()},
            "bootstrap_brier_diff_95pct": bootstrap,
            "brier_difference_vs_espn": {m: round(float(by.loc[m, "brier"] - by.loc[CANDIDATES[0], "brier"]), 6)
                                          for m in CANDIDATES[1:]},
            "provenance": {"source": "2026 completed historical games; ESPN probabilities retrieved after tipoff",
                           "verified_original_pregame_published_values": False,
                           "season_representative_sampling": False,
                           "2026_independent_untouched_holdout": False,
                           "exposure_to_market_data": False,
                           "writes_to_database": False,
                           "live_predictions_generated": False}}


def render_html(report):
    v = report["validation"].to_html(index=False, border=0, classes="data-table")
    e = report["results"].to_html(index=False, border=0, classes="data-table")
    text = html.escape(report["selected"])
    counts = html.escape(str(report["split_counts"]))
    diffs = html.escape(str(report["bootstrap_brier_diff_95pct"]))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>ESPN Adjusted Research | Historical Pilot</title><style>
body{{background:#f6f8fb;color:#182537;font-family:system-ui,Arial,sans-serif;margin:0;padding:36px}}
main{{max-width:1100px;margin:auto}} h1{{font-size:29px}} h2{{margin-top:32px}}
.notice{{padding:18px;border-left:4px solid #be8741;background:#fff7e9;line-height:1.6}}
.kpi{{background:#fff;border:1px solid #dbe3ed;border-radius:12px;padding:18px}}
small{{color:#4b6078}}table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}
th,td{{border-bottom:1px solid #e3e8ef;text-align:left;padding:10px}}th{{background:#eaf0f7}}
</style></head><body><main><h1>ESPN-Adjusted Probability — Retrospective Research</h1>
<div class="notice"><strong>Source caveats.</strong> ESPN probabilities were retrieved after the games, so their actual pregame publication is unverified. Historical source revisions and backfilled ratings may be present. This 500-game archive is deliberately stratified and is not a season-representative sample. Earlier research has examined the 2026 season, so this is <em>not</em> a pristine holdout. This is not a live forecasting system.</div>
<h2>Fixed chronological split</h2><div class="kpi">November–January training, February validation, March–April evaluation: {counts}. No fitting or selection on evaluation outcomes.</div>
<h2>Validation results (February)</h2>{v}
<h2>Model selected using February only</h2><div class="kpi"><strong>{text}</strong><p>Corrections must improve validation Brier by at least 0.001 to replace ESPN unchanged; otherwise the baseline is preserved.</p></div>
<h2>March–April evaluation (no refit)</h2>{e}
<h2>Paired bootstrap: adjusted Brier minus ESPN Brier</h2><div class="kpi">95% percentile intervals: {diffs}<p>Negative differences indicate smaller probability error. If the interval includes zero, the sample does not establish a clear difference.</p></div>
<h2>Important limitations</h2><p>This evaluates only a small retrospectively retrieved, non-representative dataset. It does not establish a prospective advantage, whether ESPN's original published probabilities were the same, or whether upstream ratings were immutable as of tipoff.</p>
</main></body></html>'''


def save_results(report, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    report["validation"].to_csv(out / "february_validation.csv", index=False)
    report["results"].to_csv(out / "march_april_evaluation.csv", index=False)
    report["predictions"].to_csv(out / "completed_games_evaluation_only.csv", index=False)
    payload = {k: v for k, v in report.items() if not isinstance(v, pd.DataFrame)}
    with (out / "study_summary.json").open("w") as f:
        json.dump(payload, f, indent=2)
    (out / "report.html").write_text(render_html(report), encoding="utf-8")
    (out / "README_RESULTS.md").write_text(
        f"# ESPN-Adjusted Research Pilot\n\nSelected using February validation: **{report['selected']}**.\n\n"
        "No March–April results were used for model fitting or candidate selection.\n\n"
        "**Caveat:** Retrospective ESPN probabilities and reconstructed ratings have no verified pretipoff-as-of provenance. The deliberate 500-game sample is not season-representative; March–April is not an independent untouched holdout because previous research examined 2026.\n\n"
        "Open report.html for tables and limitations.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Read-only retrospective ESPN adjustment evaluation")
    parser.add_argument("--input", default="reference/espn_core_2026_500.csv")
    parser.add_argument("--output-dir", default="espn_adjusted_research_results")
    args = parser.parse_args()
    data = load_and_validate(args.input)
    report = run_study(data)
    save_results(report, args.output_dir)
    print("Research complete. Selected via February:", report["selected"])
    print(report["validation"].to_string(index=False))
    print("\nMarch-April evaluation:\n", report["results"].query("group == 'all'").to_string(index=False))
    print("\nNot verified as historical pregame ESPN publication snapshots.")

if __name__ == "__main__":
    main()
