"""NCAA men's basketball V3: retrospective, leakage-conscious model evaluation.

Reads the existing Supabase PostgreSQL tables in read-only mode.
Never creates future-game selections, modifies saved V1/V2 predictions, or writes to DB.

Split: train 2022-24; choose regularization on 2025; historical check 2026.
2026 has previously been examined, so it is NOT an untouched holdout.
Only future, saved-in-advance 2027 results can establish prospective accuracy.
"""
from __future__ import annotations

import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(os.getenv("V3_OUTPUT_DIR", "v3_results"))
SEASONS = (2022, 2023, 2024, 2025, 2026)
TRAIN_SEASONS = (2022, 2023, 2024)
VALIDATION_SEASON = 2025
REVIEW_SEASON = 2026

# Only fields from a per-game PRE-GAME record are supplied to the model.
# Final scores and completed-game box scores are used solely to label and
# authenticate the historical game, never as model features.
SQL = """
SELECT
    g.season, g.game_id::text AS game_id, g.game_date,
    g.neutral_site,
    g.home_score, g.away_score,
    h.games_before AS h_games_before,
    a.games_before AS a_games_before,
    h.game_date AS h_feature_date,
    a.game_date AS a_feature_date,
    h.rating_snapshot_date AS h_rating_date,
    a.rating_snapshot_date AS a_rating_date,
    h.avg_points_for AS h_ppg,
    a.avg_points_for AS a_ppg,
    h.avg_points_against AS h_allowed,
    a.avg_points_against AS a_allowed,
    h.raw_offense AS h_raw_off,
    a.raw_offense AS a_raw_off,
    h.raw_defense AS h_raw_def,
    a.raw_defense AS a_raw_def,
    h.raw_net AS h_raw_net,
    a.raw_net AS a_raw_net,
    h.avg_possessions AS h_pace,
    a.avg_possessions AS a_pace,
    h.recent_five_margin AS h_recent_margin,
    a.recent_five_margin AS a_recent_margin,
    h.rest_days AS h_rest,
    a.rest_days AS a_rest,
    h.adjusted_net_before AS h_adj_net,
    a.adjusted_net_before AS a_adj_net,
    h.opponent_sos_before AS h_sos,
    a.opponent_sos_before AS a_sos
FROM public.games g
JOIN public.teams ht
  ON ht.team_id = g.home_team_id AND ht.is_d1 = 1
JOIN public.teams at
  ON at.team_id = g.away_team_id AND at.is_d1 = 1
JOIN public.pregame_team_features h
  ON h.game_id = g.game_id AND h.season = g.season
 AND h.team_id = g.home_team_id
JOIN public.pregame_team_features a
  ON a.game_id = g.game_id AND a.season = g.season
 AND a.team_id = g.away_team_id
JOIN public.team_game_stats hs
  ON hs.game_id = g.game_id AND hs.team_id = g.home_team_id
JOIN public.team_game_stats aws
  ON aws.game_id = g.game_id AND aws.team_id = g.away_team_id
WHERE g.season BETWEEN 2022 AND 2026
  AND g.completed = 1
  AND g.neutral_site IN (0, 1)
  AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
  AND g.home_score <> g.away_score
  AND g.home_score >= 0 AND g.away_score >= 0
  AND NOT ((g.home_score = 0 AND g.away_score = 2)
        OR (g.home_score = 2 AND g.away_score = 0))
  AND h.games_before >= 5 AND a.games_before >= 5
  AND h.game_date::date <= g.game_date::date
  AND a.game_date::date <= g.game_date::date
  AND hs.possessions > 0 AND aws.possessions > 0
  AND hs.points = g.home_score AND aws.points = g.away_score
ORDER BY g.season, g.game_date, g.game_id
"""

# Broad denominator for transparent game coverage, including matchups without
# usable pregame features. This is descriptive only; not a model input.
DENOM_SQL = """
SELECT g.season, COUNT(*)::integer AS completed_d1_games
FROM public.games g
JOIN public.teams ht ON ht.team_id=g.home_team_id AND ht.is_d1=1
JOIN public.teams at ON at.team_id=g.away_team_id AND at.is_d1=1
WHERE g.season BETWEEN 2022 AND 2026
  AND g.completed=1 AND g.neutral_site IN (0,1)
  AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
  AND g.home_score <> g.away_score
  AND g.home_score >= 0 AND g.away_score >= 0
  AND NOT ((g.home_score=0 AND g.away_score=2)
        OR (g.home_score=2 AND g.away_score=0))
GROUP BY g.season ORDER BY g.season
"""

BASE_FEATURES = ["ppg_margin", "neutral_site"]
V3_FEATURES = [
    "ppg_margin", "neutral_site", "raw_off_gap", "raw_def_gap",
    "raw_net_gap", "adjusted_net_gap", "sos_gap",
    "mean_pace", "pace_gap", "recent_margin_gap", "rest_gap",
    "experience_gap", "min_prior_games",
]


def fetch() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch via server-side credentials; print no secret values to logs."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("Missing GitHub Actions secret DATABASE_URL")
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("Install requirements_v3.txt first") from exc
    with psycopg.connect(url, sslmode="require", connect_timeout=30,
                         prepare_threshold=None, row_factory=dict_row) as conn:
        conn.read_only = True  # apply before implicit first transaction
        with conn.cursor() as cur:
            cur.execute(SQL)
            games = pd.DataFrame(cur.fetchall())
            cur.execute(DENOM_SQL)
            totals = pd.DataFrame(cur.fetchall())
    if games.empty:
        raise RuntimeError("No qualifying completed games. Check historical pregame dataset.")
    return games, totals


def numeric(frame: pd.DataFrame, names: list[str]) -> None:
    for col in names:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")


def features(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    f = raw.copy()
    if f.duplicated(["season", "game_id"]).any():
        raise RuntimeError("Duplicate game rows: check pregame/box-score table uniqueness")
    numeric(f, [
        "season", "neutral_site", "home_score", "away_score",
        "h_games_before", "a_games_before", "h_ppg", "a_ppg",
        "h_allowed", "a_allowed", "h_raw_off", "a_raw_off",
        "h_raw_def", "a_raw_def", "h_raw_net", "a_raw_net",
        "h_pace", "a_pace", "h_recent_margin", "a_recent_margin",
        "h_rest", "a_rest", "h_adj_net", "a_adj_net", "h_sos", "a_sos",
    ])

    # A rating snapshot dated on/after the game date cannot be treated as known
    # ahead of the game. BOTH adjusted rating and SOS are invalidated together.
    game_dates = pd.to_datetime(f["game_date"], utc=True, errors="coerce").dt.date
    invalid_snapshot_rows = {}
    for prefix in ("h", "a"):
        rating_dates = pd.to_datetime(f[f"{prefix}_rating_date"],
                                      utc=True, errors="coerce").dt.date
        valid = rating_dates.notna() & game_dates.notna() & (rating_dates < game_dates)
        present = f[[f"{prefix}_adj_net", f"{prefix}_sos"]].notna().any(axis=1)
        invalid_snapshot_rows[prefix] = int((present & ~valid).sum())
        f.loc[~valid, [f"{prefix}_adj_net", f"{prefix}_sos"]] = np.nan

    f["ppg_margin"] = ((f["h_ppg"] + f["a_allowed"])
                        - (f["a_ppg"] + f["h_allowed"])) / 2.0
    f["raw_off_gap"] = f["h_raw_off"] - f["a_raw_off"]
    f["raw_def_gap"] = f["a_raw_def"] - f["h_raw_def"]  # lower defense is better
    f["raw_net_gap"] = f["h_raw_net"] - f["a_raw_net"]
    f["adjusted_net_gap"] = f["h_adj_net"] - f["a_adj_net"]
    f["sos_gap"] = f["h_sos"] - f["a_sos"]
    f["mean_pace"] = (f["h_pace"] + f["a_pace"]) / 2.0
    f["pace_gap"] = f["h_pace"] - f["a_pace"]
    f["recent_margin_gap"] = f["h_recent_margin"] - f["a_recent_margin"]
    f["rest_gap"] = f["h_rest"] - f["a_rest"]
    f["experience_gap"] = f["h_games_before"] - f["a_games_before"]
    f["min_prior_games"] = f[["h_games_before", "a_games_before"]].min(axis=1)
    f["home_win"] = (f["home_score"] > f["away_score"]).astype(int)
    f.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(f)
    # The baseline margin is required for every comparison, to ensure equal samples.
    f = f.dropna(subset=["game_date", "ppg_margin", "neutral_site", "home_win"])
    f = f[f["neutral_site"].isin([0, 1]) & f["season"].isin(SEASONS)].copy()
    audit = {
        "rows_before_baseline_filter": before,
        "rows_after_baseline_filter": len(f),
        "invalid_prior_rating_rows_home": invalid_snapshot_rows["h"],
        "invalid_prior_rating_rows_away": invalid_snapshot_rows["a"],
        "adjusted_net_pair_available_pct": round(100 * f["adjusted_net_gap"].notna().mean(), 2),
        "sos_pair_available_pct": round(100 * f["sos_gap"].notna().mean(), 2),
    }
    return f, audit


def pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                 keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=c, max_iter=4000,
                                        solver="lbfgs", random_state=42)),
    ])


def fit(frame: pd.DataFrame, fields: list[str], c: float) -> Pipeline:
    train = frame[frame["season"].isin(TRAIN_SEASONS)]
    if len(train) < 200 or train["home_win"].nunique() < 2:
        raise RuntimeError("Insufficient training data from 2022–2024")
    model = pipeline(c)
    model.fit(train[fields], train["home_win"])
    return model


def evaluate(model: Pipeline, frame: pd.DataFrame, fields: list[str],
             label: str, season: int) -> dict:
    test = frame.loc[frame["season"] == season]
    if test.empty:
        raise RuntimeError(f"No usable games for season {season}")
    actual = test["home_win"].to_numpy().astype(int)
    prob = np.clip(model.predict_proba(test[fields])[:, 1], 1e-6, 1 - 1e-6)
    pred = (prob >= 0.5).astype(int)
    return {
        "season": int(season), "model": label, "games": int(len(test)),
        "winner_accuracy_pct": round(100 * float(accuracy_score(actual, pred)), 2),
        "always_home_pct": round(100 * float(actual.mean()), 2),
        "log_loss": round(float(log_loss(actual, prob, labels=[0, 1])), 5),
        "brier_score": round(float(brier_score_loss(actual, prob)), 5),
        "roc_auc": (round(float(roc_auc_score(actual, prob)), 5)
                    if len(np.unique(actual)) > 1 else None),
        "home_wins": int(actual.sum()),
        "away_wins": int((1 - actual).sum()),
        "correct": int((pred == actual).sum()),
    }


def reliability(model: Pipeline, frame: pd.DataFrame, season: int,
                fields: list[str]) -> list[dict]:
    part = frame[frame["season"] == season]
    actual = part["home_win"].to_numpy(dtype=int)
    p = model.predict_proba(part[fields])[:, 1]
    # Accuracy by *probability bin*, not a filtered set of recommended games.
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.000001)]
    rows = []
    for lo, hi in bins:
        mask = (p >= lo) & (p < hi)
        rows.append({
            "p_home_range": f"{lo:.0%}–{min(hi, 1):.0%}",
            "games": int(mask.sum()),
            "mean_predicted_home_win_pct": (round(100 * float(p[mask].mean()), 2)
                                             if mask.any() else None),
            "observed_home_win_pct": (round(100 * float(actual[mask].mean()), 2)
                                      if mask.any() else None),
        })
    return rows


def report_html(metrics: pd.DataFrame, audit: dict,
                coverage: pd.DataFrame, calibrated: pd.DataFrame,
                selected: str) -> str:
    table_kwargs = dict(index=False, border=0, classes="report-table",
                        na_rep="—", escape=True)
    def tab(df: pd.DataFrame) -> str:
        return df.to_html(**table_kwargs)
    safe = html.escape
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>NCAA MBB V3 Historical Research</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:system-ui,Arial;max-width:1100px;margin:36px auto;padding:0 20px;line-height:1.5;color:#17212b}}
h1,h2{{line-height:1.2}}.report-table{{border-collapse:collapse;width:100%;margin:14px 0 32px}}
th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#edf2f7}}
.notice{{padding:14px;border-left:4px solid #52627a;background:#f2f6f8}}
.small{{color:#506072}}code{{background:#eee;padding:2px 4px}}</style></head><body>
<h1>NCAA MBB — V3 Historical Research</h1>
<p class="small">Created {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
<div class="notice"><b>Retrospective research only.</b> Results are not future-game picks, wagering advice,
 or evidence of 90% predictive accuracy. Models trained only on 2022–24 and selected using 2025.
 The 2026 season has been previously examined in this project and is <b>not</b> an untouched holdout.</div>
<h2>Model selection</h2><p>Selected advanced V3 configuration from 2025 validation: <b>{safe(selected)}</b>
 (lowest validation log-loss). All model comparisons below use exactly the same eligible games per season.</p>
<h2>Accuracy and probabilistic quality</h2>{tab(metrics)}
<p>Winner accuracy = percentage correctly classified at probability 0.5; log-loss and Brier score
 evaluate probability quality (lower is better). An always-home benchmark appears beside each model.</p>
<h2>Coverage of completed D1 games</h2>{tab(coverage)}
<h2>2026 predicted probability calibration</h2>{tab(calibrated)}
<h2>Chronology and feature audit</h2><pre>{safe(json.dumps(audit, indent=2))}</pre>
<h2>What this does NOT establish</h2><p>Earlier snapshot dates are checked and invalid adjusted ratings
 are discarded. However, an old date alone cannot prove that all upstream rating inputs were
 constructed without future data. This script does not validate each upstream computation. 
 Real-world accuracy requires forecasts saved before games start and evaluated only after they finish.</p>
</body></html>"""


def research(raw: pd.DataFrame, totals: pd.DataFrame) -> dict:
    f, audit = features(raw)
    seen = f.groupby("season").size().rename("eligible_pregame_games").reset_index()
    coverage = totals.copy()
    numeric(coverage, ["season", "completed_d1_games"])
    coverage = coverage.merge(seen, on="season", how="left")
    coverage["eligible_pregame_games"] = coverage["eligible_pregame_games"].fillna(0).astype(int)
    coverage["coverage_pct"] = (100 * coverage["eligible_pregame_games"]
                                / coverage["completed_d1_games"].replace(0, np.nan)).round(2)
    baseline = fit(f, BASE_FEATURES, 1.0)
    variants = {f"Advanced V3 C={c:g}": fit(f, V3_FEATURES, c)
                for c in (0.1, 1.0, 10.0)}
    validation = [evaluate(model, f, V3_FEATURES, name, VALIDATION_SEASON)
                  for name, model in variants.items()]
    # Choose the model WITHOUT inspecting 2026 results.
    winner = min(validation, key=lambda row: row["log_loss"])
    selected = winner["model"]
    advanced = variants[selected]

    records = []
    for season in (VALIDATION_SEASON, REVIEW_SEASON):
        records.append(evaluate(baseline, f, BASE_FEATURES, "PPG baseline", season))
        records.append(evaluate(advanced, f, V3_FEATURES, selected, season))
    metrics = pd.DataFrame(records)
    calibration = pd.DataFrame(reliability(advanced, f, REVIEW_SEASON, V3_FEATURES))

    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT / "model_comparison.csv", index=False)
    coverage.to_csv(OUT / "sample_coverage.csv", index=False)
    calibration.to_csv(OUT / "calibration_2026.csv", index=False)
    (OUT / "feature_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    summary = {
        "research_only": True,
        "training_seasons": list(TRAIN_SEASONS),
        "validation_season": VALIDATION_SEASON,
        "previously_examined_comparison_season": REVIEW_SEASON,
        "selected_v3_variant": selected,
        "selected_using": "lowest 2025 validation log loss",
        "selection_candidates_2025": validation,
        "comparison": records,
        "audit": audit,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "report.html").write_text(report_html(metrics, audit, coverage,
                                                  calibration, selected), encoding="utf-8")
    print("\nV3 RESEARCH RESULTS — retrospective, no future-game projections")
    print(metrics.to_string(index=False))
    print("\nCoverage:\n" + coverage.to_string(index=False))
    print("\nCreated: " + str((OUT / "report.html").resolve()))
    return summary


def self_test() -> None:
    """Synthetic execution test: no database connection and no real game metrics."""
    rng = np.random.default_rng(17)
    rows = []
    for season in SEASONS:
        for i in range(240):
            gap = rng.normal(0, 15)
            p_home = 1 / (1 + np.exp(-(0.5 + gap / 12)))
            win = int(rng.random() < p_home)
            gd = f"{season-1}-12-15"
            row = {
                "season": season, "game_id": f"{season}-{i}",
                "game_date": gd, "neutral_site": i % 5 == 0,
                "home_score": 75 if win else 65,
                "away_score": 65 if win else 75,
                "h_games_before": 10, "a_games_before": 11,
                "h_feature_date": gd, "a_feature_date": gd,
                "h_rating_date": f"{season-1}-12-14",
                "a_rating_date": f"{season-1}-12-14",
                "h_ppg": 75 + gap / 2, "a_ppg": 70 - gap / 2,
                "h_allowed": 69, "a_allowed": 73,
                "h_raw_off": 104 + gap / 2, "a_raw_off": 103 - gap / 2,
                "h_raw_def": 96, "a_raw_def": 98,
                "h_raw_net": 8 + gap / 2, "a_raw_net": 5 - gap / 2,
                "h_pace": 68, "a_pace": 71,
                "h_recent_margin": gap / 2, "a_recent_margin": -gap / 2,
                "h_rest": 2, "a_rest": 1,
                "h_adj_net": 8 + gap / 2, "a_adj_net": 5 - gap / 2,
                "h_sos": 0.5, "a_sos": -0.5,
            }
            rows.append(row)
    raw = pd.DataFrame(rows)
    total = pd.DataFrame([{"season": y, "completed_d1_games": 240} for y in SEASONS])
    result = research(raw, total)
    assert len(result["comparison"]) == 4
    assert all(0 <= x["winner_accuracy_pct"] <= 100 for x in result["comparison"])
    assert Path(OUT / "report.html").exists()
    print("Synthetic self-test PASSED. Numbers above are simulated, not NCAA results.")


if __name__ == "__main__":
    try:
        if "--self-test" in sys.argv:
            self_test()
        else:
            frames = fetch()
            research(*frames)
    except Exception as exc:
        print(f"V3 research failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
