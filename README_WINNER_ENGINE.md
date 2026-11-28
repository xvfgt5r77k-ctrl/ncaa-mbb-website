[Uploading README_WINNER_ENGINE.md…]()
# NCAA Men's Basketball — Winner-Only Predictor V1

**Target:** correctly choose the winner of **at least 90% of all eligible NCAA Division I games**. This is a research objective, **not an achieved or guaranteed rate**. No close games are omitted because they are difficult. The pipeline reports actual classification accuracy as its primary score.

## How it works

1. Reads your existing `public.games`, `public.teams`, and frozen `public.mbb_v2_team_pregame` snapshots from Supabase.
2. Trains directly on home/away **game winners**, not offensive-efficiency error. Only previous-game features are used: Elo gap, adjusted-net gap, recent average scoring-margin gap, venue, rest, experience, and optional shooting profiles. No odds, wagering, real-time player status, or game-in-progress inputs.
3. Trains on 2022–24, chooses a candidate on **2025 winner accuracy**, then evaluates on **2026**. The champion is refit using 2022–26 for **upcoming 2026–27 games (season=2027)**. No information from the game outcome is used to construct its own pregame input.
4. Every upcoming game with fresh, valid previous-day features is scored within six hours of tipoff; the pipeline does not hide low-confidence predictions. Forecasts are **frozen once** in a private SQL table before tipoff. No result is backfilled as a forecast.
5. `grade_winners.py` scores saved forecasts after games finish and reports the real pregame win rate. Gaps in game or feature coverage are separately audited. Existing 2027 frozen V2 prospective rows are intentionally ignored, because they may be weeks old.

**Limits:** Historical `historical_replay` snapshots were reconstructed afterward from earlier box scores. That is not independent proof of historic publication times. 2026 has been used in previous research and is not pristine. A true prospective performance estimate starts with newly saved forecasts after deployment. The pipeline **cannot guarantee 90%**; it will make it measurable.

## Install: upload these 8 files to the existing `ncaa-mbb-website` GitHub repo

```text
ncaa-mbb-website/
  winner_engine.py
  grade_winners.py
  requirements_winner_engine.txt
  README_WINNER_ENGINE.md
  sql/
    01_create_winner_forecasts.sql
    02_accuracy_check.sql
  tests/
    test_winner_engine.py
  .github/
    workflows/
      mbb-winner-engine.yml
```

Don't remove, rename, or replace the existing ESPN importer, V2 feature store, prediction engine, or Streamlit files. Keep your existing `DATABASE_URL` **GitHub Actions secret**. It needs server-side SELECT on games, teams, and V2 features, plus INSERT and SELECT on the new private table.

### One-time database setup

In Supabase **SQL Editor**, run `sql/01_create_winner_forecasts.sql`. It creates a new table and leaves all existing data unchanged. The SQL table is not publicly readable by anonymous browser users.

### Launch

1. Commit the eight files to the exact locations above.
2. Confirm `NCAA MBB Feature Store V2 Refresh` is green at least once and `public.mbb_v2_team_pregame` includes seasons 2022–2026. The predictor recomputes 2027 snapshots fresh each run rather than reusing stale frozen 2027 prospective rows. Historical backfill through 2026 should already exist in your project.
3. GitHub → **Actions** → **NCAA MBB Winner-Only Predictor** → **Run workflow**. This runs twelve tests, trains, validates, creates future predictions, saves any newly eligible games before tipoff, grades saved games, and uploads artifacts.
4. Download the **ncaa-mbb-winner-only-predictions** artifact and inspect:
   - `run_audit.json` — coverage, selected model, winner accuracy for historical 2026
   - `candidate_validation_2025.csv` — controlled candidate selection
   - `winner_accuracy_2026.csv` — all games and competitive/noncompetitive accuracy
   - `next_game_winners.csv` — upcoming game IDs, teams, home-win probability, predicted team ID (no wagering information)
   - `prospective_accuracy.json` — live, saved, pre-tipoff accuracy once games finish
   - `graded_pretip_forecasts.csv` — game-by-game result tracking

It runs automatically **every three hours UTC**, using the latest imported completed 2027 games. It produces frozen predictions during the six-hour window before tipoff. Scheduled GitHub Actions jobs can be delayed; this system deliberately skips games if it cannot save valid information at least ten minutes before tipoff.

### What to do when the reports show a problem

- **No V2 table:** Run `sql/01_create_feature_tables.sql` and the V2 refresh workflow as in your existing project. Do not create the tables from scratch twice.
- **No historical data:** Run the V2 historical backfill for 2022–2026. The predictor will not invent old game results.
- **No upcoming forecasts:** Verify that ESPN importer loaded upcoming 2027 schedule games and that there are at least ten minutes remaining before tipoff. Recomputed features are held in memory, so no saved 2027 prospective rows are required.
- **Missing forecast table:** Run `sql/01_create_winner_forecasts.sql` once.
- **Insufficient privileges:** Fix server-side `DATABASE_URL` database role permissions. Do not paste a production database URL into public GitHub files.

### Important quality rules

A forecast is counted only when it was saved before tipoff. Model selection never uses the later 2026 results. Historical evaluation excludes only invalid score records and structurally inconsistent feature dates; it does not exclude difficult matchups. Future coverage is explicitly reported. No live game score, betting market, or postgame player statistics appear in predictor features.
