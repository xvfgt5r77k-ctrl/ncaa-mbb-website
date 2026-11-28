[README_OPPONENT_V3.md](https://github.com/user-attachments/files/35537572/README_OPPONENT_V3.md)
# NCAA MBB — Opponent-Adjusted Ratings V3 (read-only research)

This is an **experimental alternative** to the existing `db_upgrade_v2.py` opponent adjustment. It **does not** change Supabase tables, production models, dashboards, rankings, or forecasts. It doesn't collect or infer injury information.

## Why test V3?

V2 adjusts each game's offensive/defensive efficiency using the opponent's **rating at the time of that game**. The adjustment attached to that game isn't subsequently reevaluated as the opponent's season unfolds. V3 fits team offense and defense from *all games known before the snapshot date*, so earlier performance is reweighted based on opponents' performance known **by that date**.

V3 uses time-decayed, ridge-regularized two-way offensive/defensive efficiency:

```
Observed offensive points per 100 possessions
    = league_average + own_offense_strength - opponent_defense_strength
      + home_venue_adjustment + error
```

For the pilot, hyperparameters are held fixed in advance:

- Historical game weight halves every **45 days**.
- Offense and defense use **8-game equivalent regularization** per team effect.
- League average shrinks toward **100 points/100 possessions** with a 25-observation equivalent prior.
- Venue adjustment for nonneutral games is **+1 point per 100 possessions** to the home offense and **-1** to the away offense. This is a starting assumption, **not an empirically established home advantage**.
- Opponent strength-of-schedule is the recency-weighted mean of opponents' *current as-of-date adjusted net ratings*.

The model is recomputed once per game date **before applying all results from that day**, so there is no same-day or future-game outcome leakage. Past outcomes are used to reassess opponent quality, but results after a given matchup are **never** used to create that matchup's historical pregame snapshot.

## Upload to your existing `ncaa-mbb-website` repo

Upload the five files in this ZIP to these exact paths:

```
ncaa-mbb-website/
  research_opponent_adjustment_v3.py
  requirements_opponent_v3.txt
  README_OPPONENT_V3.md
  tests/
    test_opponent_adjustment_v3.py
  .github/
    workflows/
      mbb-v3-opponent-adjustment-research.yml
```

**Prerequisites already used by your V2 feature store:** `db_upgrade_v2.py`, `requirements_db_upgrade.txt`, the Supabase `DATABASE_URL` GitHub Actions secret, plus historical ESPN `games` and `team_game_stats` rows. To compare directly with V2, also have `mbb_v2_team_pregame` historical_replay rows for the years being evaluated. You do **not** need any new SQL.

1. Add the files to their exact paths and commit to GitHub.
2. Open **Actions → NCAA MBB V3 Opponent Adjustment Research → Run workflow**.
3. Wait for the test and research steps to finish. Open the run's **Artifacts** section and download **`ncaa-mbb-v3-opponent-adjustment-research`**.
4. Review `summary.txt`, `summary.json`, `v3_vs_v2_matched_efficiency.csv`, and `v3_historical_team_snapshots.csv`.

The default run covers ESPN season identifiers **2022–2026** (academic years 2021–22 through 2025–26). It does not overwrite or insert 2027 prospective snapshots. GitHub uses a read-only SQL transaction for research queries.

## What success looks like

The report compares *both teams' observed offensive points per 100 possessions* to V2 and V3 expectations on **exactly the same eligible game IDs**, with identical initial venue correction. It includes per-season MAE and RMSE (lower is better), 95% paired bootstrap intervals for MAE differences, and early-season/established-team/venue splits.

**These metrics are NOT winner accuracy or verified pregame forecasting accuracy.** Historical ESPN exports may reflect corrections not published before the game. Season 2026 was already inspected in previous research, so it isn't a new untouched holdout. Do not claim an improvement in live prediction performance based on this report alone.

A useful next study would select parameters using **2025 validation only**, lock them, check the unchanged **2026 retrospective** sample, then collect **2027 saved-before-tipoff prospective** evidence. Do not tune on 2026 and then present 2026 as unseen.

## Data safeguards

- D1 vs D1 games only; invalid final scores, 0–2 administrative results, missing boxes, score mismatches, and paired possession estimates differing by more than six are excluded from efficiency analysis.
- No inference from absent player rows, news, or injury data.
- Same-day fixtures are snapshotted simultaneously before any of that day's results.
- Every season restarts at a neutral prior. Ratings are computed **only** from past games in that season.
- Missing or absent V2 backfills result in missing comparisons rather than fabricated numbers.
- Output ZIP files are research artifacts from GitHub; the database is never modified.

## Note on implementation

This add-on **imports helpers from `db_upgrade_v2.py`** (`d1_games_sorted`, `choose_kind`, `get_possessions`, `valid_score`, aliases), but does not change that file. The tests exercise chronology, source quality, venue behavior, and exact-match comparison. It uses NumPy and SciPy for sparse ridge least squares; these are installed by the new requirements file.
