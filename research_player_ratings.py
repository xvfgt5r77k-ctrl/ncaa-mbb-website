"""Read-only NCAA men's basketball player-data and opponent-ratings research.

Historical replay = reconstructed from historical box scores; NEVER verified
pretipoff availability or timestamped pregame predictions. Does not make live
picks, connect to betting, or change Supabase tables.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "player-ratings-research-1.0"
SEASONS = (2022, 2023, 2024, 2025, 2026)
FEATURES = {
    "A_adjusted_efficiency": ["net_gap", "home_court", "net_home_interaction"],
    "B_early_season_and_opponent": ["net_gap", "home_court", "net_home_interaction",
        "elo_gap", "opponent_elo_gap", "rest_gap", "games_min_log", "games_gap_log",
        "uncertainty_scaled_net", "net_early_interaction"],
    "C_plus_prior_rotation": ["net_gap", "home_court", "net_home_interaction",
        "elo_gap", "opponent_elo_gap", "rest_gap", "games_min_log", "games_gap_log",
        "uncertainty_scaled_net", "net_early_interaction", "rotation_points_gap",
        "rotation_top3_gap", "rotation_continuity_gap", "player_rows_gap",
        "player_expected_minutes_gap", "both_player_sufficient"],
}
TEAM_COLS = ["game_day", "tipoff_utc", "feature_version", "quality_flags",
             "games_before", "elo_rating", "opponent_elo_mean", "rest_days",
             "adjusted_offense_per100", "adjusted_defense_per100", "adjusted_net_per100",
             "player_coverage", "expected_rotation_points_per40",
             "rotation_top3_minutes_share", "rotation_continuity"]


def fetch_dataframe(url: str) -> pd.DataFrame:
    """Read-only, repeatable snapshot of completed D1 games + stored replay features."""
    import psycopg
    from psycopg.rows import dict_row
    cols = ",\n        ".join(
        f"h.{c} AS home_{c}, a.{c} AS away_{c}" for c in TEAM_COLS
    )
    query = f"""
        WITH player_agg AS (
            SELECT season, game_id, team_id,
                   COUNT(*)::int AS player_count,
                   SUM(previous_five_expected_minutes) AS expected_minutes,
                   COUNT(*) FILTER (WHERE prior_box_score_only IS FALSE) AS non_box_source
            FROM public.mbb_v2_player_pregame
            WHERE snapshot_kind='historical_replay' AND feature_version='v2.0'
              AND season BETWEEN 2022 AND 2026
            GROUP BY season, game_id, team_id
        )
        SELECT g.game_id::text AS game_id, g.season, g.game_date::text AS game_date,
               g.start_time_utc::text AS start_time_utc, g.neutral_site,
               g.home_score, g.away_score,
               {cols},
               COALESCE(hp.player_count, 0) AS home_player_count,
               COALESCE(ap.player_count, 0) AS away_player_count,
               hp.expected_minutes AS home_player_expected_minutes,
               ap.expected_minutes AS away_player_expected_minutes,
               COALESCE(hp.non_box_source, 0) AS home_non_box_source,
               COALESCE(ap.non_box_source, 0) AS away_non_box_source
        FROM public.games g
        JOIN public.teams ht ON ht.team_id=g.home_team_id AND ht.is_d1=1
        JOIN public.teams at ON at.team_id=g.away_team_id AND at.is_d1=1
        JOIN public.mbb_v2_team_pregame h
          ON h.game_id=g.game_id::text AND h.team_id=g.home_team_id::text
         AND h.season=g.season AND h.snapshot_kind='historical_replay'
         AND h.feature_version='v2.0'
        JOIN public.mbb_v2_team_pregame a
          ON a.game_id=g.game_id::text AND a.team_id=g.away_team_id::text
         AND a.season=g.season AND a.snapshot_kind='historical_replay'
         AND a.feature_version='v2.0'
        LEFT JOIN player_agg hp
          ON hp.game_id=h.game_id AND hp.team_id=h.team_id AND hp.season=h.season
        LEFT JOIN player_agg ap
          ON ap.game_id=a.game_id AND ap.team_id=a.team_id AND ap.season=a.season
        WHERE g.completed=1 AND g.season BETWEEN 2022 AND 2026
        ORDER BY g.season, g.game_date, g.game_id
    """
    with psycopg.connect(url, sslmode="require", connect_timeout=30,
                         prepare_threshold=None, row_factory=dict_row) as conn:
        conn.read_only = True
        conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        with conn.cursor() as cur:
            cur.execute(query)
            return pd.DataFrame(cur.fetchall())


def _n(df: pd.DataFrame, key: str) -> pd.Series:
    return pd.to_numeric(df[key], errors="coerce")


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate replay structure and derive only prior-game research features."""
    required = {"game_id", "season", "game_date", "start_time_utc", "neutral_site",
                "home_score", "away_score"}
    for side in ("home", "away"):
        required.update(f"{side}_{x}" for x in TEAM_COLS)
        required.update({f"{side}_player_count", f"{side}_player_expected_minutes", f"{side}_non_box_source"})
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Database query missing required columns: {missing}")
    if raw.empty:
        raise ValueError("No historical games; confirm V2 2022-2026 backfill")
    df = raw.copy()
    df["game_id"] = df["game_id"].astype(str)
    if df["game_id"].duplicated().any():
        raise ValueError("Duplicate joined game_id; check features and season mapping")
    df["season"] = _n(df,"season")
    df["home_score"] = _n(df,"home_score")
    df["away_score"] = _n(df,"away_score")
    df["neutral_site"] = _n(df,"neutral_site")
    tipoff = pd.to_datetime(df["start_time_utc"], utc=True, errors="coerce")
    game_day = pd.to_datetime(df["game_date"], errors="coerce", utc=True)
    valid_score = df.home_score.ge(0) & df.away_score.ge(0) & df.home_score.ne(df.away_score)
    administrative = ((df.home_score.eq(0)&df.away_score.eq(2)) |
                      (df.home_score.eq(2)&df.away_score.eq(0)))
    valid = valid_score & ~administrative & df.neutral_site.isin([0,1]) & tipoff.notna()
    reasons = {"invalid_score_venue_or_tipoff": ~valid}
    for side in ("home", "away"):
        prefix = side + "_"
        for c in ["games_before", "elo_rating", "opponent_elo_mean", "rest_days",
                  "adjusted_offense_per100", "adjusted_defense_per100",
                  "adjusted_net_per100", "expected_rotation_points_per40",
                  "rotation_top3_minutes_share", "rotation_continuity",
                  "player_count", "player_expected_minutes", "non_box_source"]:
            df[prefix+c] = _n(df, prefix+c)
        fday = pd.to_datetime(df[prefix+"game_day"], errors="coerce", utc=True)
        ftip = pd.to_datetime(df[prefix+"tipoff_utc"], errors="coerce", utc=True)
        provenance = df[prefix+"quality_flags"].apply(
            lambda x: isinstance(x, (list,tuple)) and "reconstructed_not_saved_pretipoff" in x)
        net_delta = (df[prefix+"adjusted_offense_per100"] -
                     df[prefix+"adjusted_defense_per100"] -
                     df[prefix+"adjusted_net_per100"]).abs()
        reasons[prefix+"bad_provenance_or_date"] = (~provenance |
            ~df[prefix+"feature_version"].eq("v2.0") |
            ~fday.dt.normalize().eq(game_day.dt.normalize()) |
            ~ftip.eq(tipoff))
        reasons[prefix+"bad_rating"] = (
            ~df[prefix+"games_before"].ge(0) |
            df[prefix+"elo_rating"].isna() |
            ~net_delta.le(0.1))
    reasons["under_five_prior_games"] = (df.home_games_before.lt(5) | df.away_games_before.lt(5))
    excluded = pd.concat(reasons.values(), axis=1).any(axis=1)
    audit = {k:int(v.sum()) for k,v in reasons.items()}
    audit["raw_games"] = int(len(df))
    audit["excluded_games_unique"] = int(excluded.sum())
    df = df.loc[~excluded].copy()
    audit["eligible_games"] = int(len(df))
    if df.empty:
        raise ValueError("No eligible games after historical integrity checks")
    df["home_win"] = df.home_score.gt(df.away_score).astype(int)
    df["home_court"] = 1.0 - df.neutral_site
    df["net_gap"] = df.home_adjusted_net_per100 - df.away_adjusted_net_per100
    df["net_home_interaction"] = df.net_gap * df.home_court
    df["elo_gap"] = df.home_elo_rating - df.away_elo_rating
    df["opponent_elo_gap"] = df.home_opponent_elo_mean - df.away_opponent_elo_mean
    df["rest_gap"] = df.home_rest_days - df.away_rest_days
    df["games_min"] = df[["home_games_before", "away_games_before"]].min(axis=1)
    df["games_min_log"] = np.log1p(df.games_min)
    df["games_gap_log"] = np.log1p(df.home_games_before) - np.log1p(df.away_games_before)
    # Confidence adjustment is researched separately from the original V2 8-game prior.
    # It must NOT be interpreted as a proven better rating.
    df["uncertainty_scaled_net"] = df.net_gap * df.games_min / (df.games_min + 10)
    df["net_early_interaction"] = df.net_gap * (df.games_min < 10).astype(float)
    for col, output in [
        ("expected_rotation_points_per40", "rotation_points_gap"),
        ("rotation_top3_minutes_share", "rotation_top3_gap"),
        ("rotation_continuity", "rotation_continuity_gap"),
        ("player_count", "player_rows_gap"),
        ("player_expected_minutes", "player_expected_minutes_gap")]:
        df[output] = df["home_"+col] - df["away_"+col]
    good = (df.home_player_coverage.eq("sufficient") &
            df.away_player_coverage.eq("sufficient") &
            df.home_player_count.gt(0) & df.away_player_count.gt(0))
    df["both_player_sufficient"] = good.astype(float)
    for col in ("rotation_points_gap", "rotation_top3_gap", "rotation_continuity_gap",
                "player_rows_gap", "player_expected_minutes_gap"):
        df.loc[~good,col] = np.nan
    df["competitive"] = df.net_gap.abs().le(5)
    df["early_season"] = df.games_min.lt(10)
    df["player_feature_ready"] = good
    df["player_missing_or_partial"] = ~good
    # Absence of a player row is NOT evidence that a specific person was injured/out.
    df["unverified_availability_flag"] = True
    quality = df.groupby("season",sort=True).agg(
        games=("game_id","size"),
        player_coverage_games=("player_feature_ready","sum"),
        partial_or_missing_player=("player_missing_or_partial","sum"),
        early_season_games=("early_season","sum"),
        neutral_games=("neutral_site","sum"),
    ).reset_index()
    quality["player_coverage_pct"] = (100 * quality.player_coverage_games / quality.games).round(2)
    audit["historical_player_availability_status_verified"] = False
    audit["non_box_score_status_records"] = int(df.home_non_box_source.sum() + df.away_non_box_source.sum())
    audit["timing_limit"] = "Historical replay source-publication timestamps not independently verified"
    return df, quality, audit


def classifier(c: float = .5):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(),
                         LogisticRegression(C=c, max_iter=1200, random_state=17))


def metrics(y: pd.Series | np.ndarray, p: np.ndarray) -> dict:
    yy = np.asarray(y,dtype=int)
    pp = np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    if len(yy)==0:
        return {"n":0,"accuracy_pct":None,"brier":None,"log_loss":None}
    return {"n":int(len(yy)),"correct":int(np.sum((pp>=.5)==yy)),
            "accuracy_pct":round(100*float(np.mean((pp>=.5)==yy)),3),
            "brier":round(float(brier_score_loss(yy,pp)),6),
            "log_loss":round(float(log_loss(yy,pp,labels=[0,1])),6)}


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,dict]:
    train=df[df.season.isin([2022,2023,2024])].copy()
    val=df[df.season.eq(2025)].copy()
    test=df[df.season.eq(2026)].copy()
    for name, split in (("training",train),("validation",val),("retrospective",test)):
        if len(split)<40 or split.home_win.nunique()<2:
            raise ValueError(f"Insufficient {name} historical data: {len(split)} games")
    results=[]
    for name,features in FEATURES.items():
        model=classifier()
        model.fit(train[features], train.home_win)
        pv=model.predict_proba(val[features])[:,1]
        results.append({"model":name,"split":"validation_2025",**metrics(val.home_win,pv)})
    validation=pd.DataFrame(results)
    winner=validation.sort_values(["brier","model"],ascending=[True,True]).iloc[0]["model"]
    tests=[]
    subgroups=[]
    for name,features in FEATURES.items():
        # Configurations fixed in advance, 2026 never used for choosing a model.
        model=classifier()
        train_through_2025=pd.concat([train,val],ignore_index=True)
        model.fit(train_through_2025[features],train_through_2025.home_win)
        probs=model.predict_proba(test[features])[:,1]
        tests.append({"model":name,"split":"retrospective_2026",**metrics(test.home_win,probs),
                      "selected_on_2025":bool(name==winner)})
        for label,mask in (
            ("competitive",test.competitive), ("other",~test.competitive),
            ("neutral",test.neutral_site.eq(1)), ("home_court",test.neutral_site.eq(0)),
            ("early_season",test.early_season), ("established",~test.early_season),
            ("both_player_covered",test.player_feature_ready),
            ("partial_or_missing_player",~test.player_feature_ready)):
            subgroups.append({"model":name,"group":label,**metrics(test.loc[mask,"home_win"],probs[mask.to_numpy()])})
    test_metrics=pd.DataFrame(tests)
    summary={"version":VERSION,"training_games":int(len(train)),
             "validation_games":int(len(val)),"retrospective_2026_games":int(len(test)),
             "chosen_from_2025_validation":winner,
             "selection_metric":"2025 Brier score; fixed before reviewing 2026",
             "2026_is_not_untouched_holdout":True,
             "live_predictions_created":False,
             "availability_verified_from_box_scores":False,
             "database_writes":False}
    return validation,test_metrics,pd.DataFrame(subgroups),summary


def run(raw: pd.DataFrame, output_dir: str):
    target=Path(output_dir)
    target.mkdir(parents=True,exist_ok=True)
    df,quality,audit=prepare(raw)
    val,test,subgroups,summary=evaluate(df)
    quality.to_csv(target/"player_and_rating_quality_by_season.csv",index=False)
    val.to_csv(target/"validation_2025.csv",index=False)
    test.to_csv(target/"retrospective_2026.csv",index=False)
    subgroups.to_csv(target/"subgroups_2026.csv",index=False)
    (target/"audit.json").write_text(json.dumps(audit,indent=2,default=str),encoding="utf-8")
    (target/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (target/"READ_ME_FIRST.txt").write_text(
        "NCAA MBB Steps 2 & 3 historical research audit\n"
        "A = V2 adjusted efficiency + venue.\n"
        "B = A + Elo, opponent Elo, early-season uncertainty, rest.\n"
        "C = B + prior-box-score rotation features (NOT verified player injuries).\n"
        "2022-2024 training, 2025 validation selection, 2026 descriptive review.\n"
        "2026 has been used in previous studies; NOT an untouched holdout.\n"
        "Historical box scores were imported later; pretipoff data-publication time NOT verified.\n"
        "No existing forecasts or database tables were modified.\n", encoding="utf-8"
    )
    print("Historical research finished; outputs:",target)
    print("Selected by 2025 validation:",summary["chosen_from_2025_validation"])
    print(test.to_string(index=False))


def main():
    parser=argparse.ArgumentParser(description="Read-only historical player coverage and rating improvements")
    parser.add_argument("--output-dir",default="player_rating_research_report")
    args=parser.parse_args()
    url=os.getenv("DATABASE_URL")
    if not url:
        parser.error("Missing DATABASE_URL GitHub Actions secret")
    raw=fetch_dataframe(url)
    run(raw,args.output_dir)

if __name__=="__main__":
    main()
