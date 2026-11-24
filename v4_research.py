"""NCAA MBB V4: retrospective research on previous calendar-day game information.

Runs entirely READ-ONLY against existing Supabase tables. Does not generate
upcoming game forecasts or change V1/V2/V3 database records.

Training: seasons 2022-24. Model selection: 2025. Reviewed comparison: 2026.
2026 is *not* an untouched test set in this project.

Requires v3_research.py alongside this file to reproduce the exact V3 baseline
and pregame feature definitions and evaluation eligibility.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import v3_research as v3
except ImportError as exc:
    raise SystemExit('Missing v3_research.py in repository root; keep V3 alongside V4') from exc

OUTPUT = Path(os.environ.get('V4_OUTPUT_DIR', 'v4_results'))
TRAIN = (2022, 2023, 2024)
VALIDATE = 2025
REVIEW = 2026
SEED = 42

SCORE_SQL = """
SELECT g.season, g.game_id::text AS game_id, g.game_date,
       g.home_team_id::text AS home_team_id,
       g.away_team_id::text AS away_team_id,
       g.neutral_site, g.home_score, g.away_score
FROM public.games g
JOIN public.teams ht ON ht.team_id=g.home_team_id AND ht.is_d1=1
JOIN public.teams at ON at.team_id=g.away_team_id AND at.is_d1=1
WHERE g.season BETWEEN 2022 AND 2026
  AND g.completed=1 AND g.neutral_site IN (0,1)
  AND g.game_date IS NOT NULL
  AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
  AND g.home_score <> g.away_score
  AND g.home_score >= 0 AND g.away_score >= 0
  AND NOT ((g.home_score=0 AND g.away_score=2)
        OR (g.home_score=2 AND g.away_score=0))
ORDER BY g.season, g.game_date, g.game_id
"""

STRICT_FIELDS = [
    'strict_ppg_margin', 'elo_gap', 'elo_home_probability',
    'strict_margin_gap', 'recent5_margin_gap', 'recent3_margin_gap',
    'recent5_win_gap', 'past_opponent_elo_gap', 'rest_gap',
    'games_gap', 'min_prior_games_strict', 'season_progress',
    'neutral_site', 'combined_scoring_rate',
]
HYBRID_FIELDS = v3.V3_FEATURES + [x for x in STRICT_FIELDS if x not in v3.V3_FEATURES]


@dataclass
class TeamPast:
    rating: float = 1500.0
    scored: list[float] = field(default_factory=list)
    conceded: list[float] = field(default_factory=list)
    margins: list[float] = field(default_factory=list)
    wins: list[int] = field(default_factory=list)
    opponent_elo: list[float] = field(default_factory=list)
    last_game_date: date | None = None

    @property
    def n(self) -> int:
        return len(self.scored)


def fetch() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read data only; keep secrets out of logs."""
    url = os.environ.get('DATABASE_URL')
    if not url:
        raise RuntimeError('GitHub secret DATABASE_URL is missing')
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(url, sslmode='require', connect_timeout=30,
                         prepare_threshold=None, row_factory=dict_row) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(v3.SQL)
            raw = pd.DataFrame(cur.fetchall())
            cur.execute(v3.DENOM_SQL)
            totals = pd.DataFrame(cur.fetchall())
            cur.execute(SCORE_SQL)
            scores = pd.DataFrame(cur.fetchall())
    if raw.empty or scores.empty:
        raise RuntimeError('No historical games returned. Check database permissions and imports.')
    return raw, totals, scores


def safe_mean(seq, last: int | None = None) -> float:
    values = seq[-last:] if last is not None else seq
    return float(np.mean(values)) if len(values) else np.nan


def state_features(state: TeamPast, day: date) -> dict:
    rest = (day - state.last_game_date).days - 1 if state.last_game_date else np.nan
    return {
        'rating': state.rating,
        'n': state.n,
        'ppg': safe_mean(state.scored),
        'allowed': safe_mean(state.conceded),
        'margin': safe_mean(state.margins),
        'recent5_margin': safe_mean(state.margins, 5),
        'recent3_margin': safe_mean(state.margins, 3),
        'recent5_win': safe_mean(state.wins, 5),
        'past_opponent_elo': safe_mean(state.opponent_elo),
        'rest': float(min(14, max(0, rest))) if np.isfinite(rest) else np.nan,
    }


def chronological_features(scores: pd.DataFrame) -> pd.DataFrame:
    """Replay ALL historical games in order, never using a same-day result.

    Team ratings and running averages are frozen at the start of each calendar
    day; all games that day are processed before applying changes. Seasons reset
    to 1500 Elo. Only D1 games with valid recorded scores enter the history.
    """
    required = ['season', 'game_id', 'game_date', 'home_team_id',
                'away_team_id', 'home_score', 'away_score', 'neutral_site']
    if not set(required).issubset(scores.columns):
        raise RuntimeError('Missing columns in historical scoreboard query')
    frame = scores.copy()
    if frame.duplicated(['season', 'game_id']).any():
        raise RuntimeError('Duplicate scoreboard game_id found')
    for col in ['season', 'home_score', 'away_score', 'neutral_site']:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame['day'] = pd.to_datetime(frame['game_date'], utc=True, errors='coerce').dt.date
    frame = frame.dropna(subset=['day', 'season', 'home_score', 'away_score',
                                 'home_team_id', 'away_team_id']).copy()
    frame['home_team_id'] = frame['home_team_id'].astype(str)
    frame['away_team_id'] = frame['away_team_id'].astype(str)
    frame = frame.sort_values(['season', 'day', 'game_id'], kind='stable')
    rows = []
    for season, season_data in frame.groupby('season', sort=True):
        teams: dict[str, TeamPast] = defaultdict(TeamPast)
        start_day = season_data['day'].min()
        for day, one_day in season_data.groupby('day', sort=True):
            updates = []
            for game in one_day.itertuples(index=False):
                home = teams[game.home_team_id]
                away = teams[game.away_team_id]
                h, a = state_features(home, day), state_features(away, day)
                h_adv = 0.0 if int(game.neutral_site) else 60.0
                p_h = 1.0 / (1.0 + 10.0 ** ((a['rating'] - h['rating'] - h_adv) / 400.0))
                pp_margin = ((h['ppg'] + a['allowed']) - (a['ppg'] + h['allowed'])) / 2
                rows.append({
                    'season': int(season), 'game_id': str(game.game_id),
                    'strict_ppg_margin': pp_margin,
                    'elo_gap': h['rating'] - a['rating'],
                    'elo_home_probability': p_h,
                    'strict_margin_gap': h['margin'] - a['margin'],
                    'recent5_margin_gap': h['recent5_margin'] - a['recent5_margin'],
                    'recent3_margin_gap': h['recent3_margin'] - a['recent3_margin'],
                    'recent5_win_gap': h['recent5_win'] - a['recent5_win'],
                    'past_opponent_elo_gap': h['past_opponent_elo'] - a['past_opponent_elo'],
                    'rest_gap': h['rest'] - a['rest'],
                    'games_gap': h['n'] - a['n'],
                    'min_prior_games_strict': min(h['n'], a['n']),
                    'season_progress': (day - start_day).days,
                    'neutral_site': int(game.neutral_site),
                    'combined_scoring_rate': safe_mean([h['ppg'], a['ppg']])
                        if np.isfinite(h['ppg']) and np.isfinite(a['ppg']) else np.nan,
                    'h_n_strict': h['n'], 'a_n_strict': a['n'],
                })
                # Freeze all values until the NEXT DAY. Compute updates from
                # both teams' ratings BEFORE this game (not a prior same-day game).
                margin = abs(float(game.home_score) - float(game.away_score))
                weight = min(1.6, 0.75 + 0.27 * np.log1p(margin))
                delta = 22.0 * weight * (float(game.home_score > game.away_score) - p_h)
                updates.append((game.home_team_id, game.away_team_id,
                                float(game.home_score), float(game.away_score),
                                float(h['rating']), float(a['rating']), float(delta)))
            # Apply end-of-day updates; never used in predictions dated today.
            for h_id, a_id, h_score, a_score, h_rating, a_rating, delta in updates:
                home, away = teams[h_id], teams[a_id]
                home.rating += delta
                away.rating -= delta
                home.scored.append(h_score)
                home.conceded.append(a_score)
                home.margins.append(h_score - a_score)
                home.wins.append(int(h_score > a_score))
                home.opponent_elo.append(a_rating)
                home.last_game_date = day
                away.scored.append(a_score)
                away.conceded.append(h_score)
                away.margins.append(a_score - h_score)
                away.wins.append(int(a_score > h_score))
                away.opponent_elo.append(h_rating)
                away.last_game_date = day
    return pd.DataFrame(rows)


def make_frame(raw: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame, audit = v3.features(raw)
    historical = chronological_features(scores)
    frame = frame.merge(historical, on=['season', 'game_id'], how='left',
                        validate='one_to_one', suffixes=('', '_strict'))
    before = len(frame)
    # Require at least five previously dated, successfully replayed games per
    # team so strict and V3 comparison samples are identical and defensible.
    frame = frame.loc[(frame['h_n_strict'] >= 5) & (frame['a_n_strict'] >= 5)].copy()
    frame['neutral_site'] = pd.to_numeric(frame['neutral_site'], errors='coerce')
    if len(frame) < 100:
        raise RuntimeError('Too few eligible games after chronology check')
    if not set(TRAIN + (VALIDATE, REVIEW)).issubset(set(frame['season'].unique())):
        raise RuntimeError('Missing required seasons after filtering')
    for col in STRICT_FIELDS + HYBRID_FIELDS:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame = frame.replace([np.inf, -np.inf], np.nan)
    audit.update({
        'v3_qualified_before_strict_filter': before,
        'matched_chronological_games': int(frame.shape[0]),
        'excluded_insufficient_prior_day_games': int(before - len(frame)),
        'strict_fields': STRICT_FIELDS,
        'hybrid_uses_v3_upstream_fields_not_fully_audited': True,
        'strict_data_cutoff': 'previous calendar day only; no same-day outcomes',
        'eligibility_uses_historical_game_labels_and_v3_quality_checks': True,
    })
    return frame, audit


def logistic(c: float) -> Pipeline:
    return Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
        ('scale', StandardScaler()),
        ('model', LogisticRegression(C=c, max_iter=3000, solver='lbfgs', random_state=SEED)),
    ])


def boosted(depth: int, iters: int, leaf: int, rate: float) -> Pipeline:
    return Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
        ('model', HistGradientBoostingClassifier(max_depth=depth, max_iter=iters,
                 min_samples_leaf=leaf, learning_rate=rate, l2_regularization=8.0,
                 early_stopping=False, random_state=SEED)),
    ])


def trees() -> Pipeline:
    return Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
        ('model', ExtraTreesClassifier(n_estimators=300, max_features=0.8, max_depth=10,
                  min_samples_leaf=24, n_jobs=-1, random_state=SEED)),
    ])


def quality(y: np.ndarray, probs: np.ndarray, label: str, season: int) -> dict:
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    correct = pred == y
    return {
        'season': season, 'model': label, 'games': len(y),
        'correct': int(correct.sum()),
        'winner_accuracy_pct': round(100 * float(accuracy_score(y, pred)), 2),
        'always_home_pct': round(100 * float(y.mean()), 2),
        'log_loss': round(float(log_loss(y, p, labels=[0, 1])), 5),
        'brier_score': round(float(brier_score_loss(y, p)), 5),
        'roc_auc': round(float(roc_auc_score(y, p)), 5) if len(np.unique(y)) == 2 else None,
    }


def model_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    train = frame[frame.season.isin(TRAIN)]
    val = frame[frame.season == VALIDATE]
    review = frame[frame.season == REVIEW]
    if len(train) < 500 or len(val) < 200 or len(review) < 200:
        raise RuntimeError('Not enough train/validation/review games')
    y_train = train['home_win'].astype(int).to_numpy()
    y_val = val['home_win'].astype(int).to_numpy()
    y_review = review['home_win'].astype(int).to_numpy()
    models: dict[str, dict] = {}

    def train_one(name: str, fields: list[str], obj: Pipeline, group: str) -> None:
        obj.fit(train[fields], y_train)
        models[name] = {
            'group': group,
            'val': obj.predict_proba(val[fields])[:, 1],
            'review': obj.predict_proba(review[fields])[:, 1],
        }

    train_one('PPG baseline', v3.BASE_FEATURES, v3.pipeline(1.0), 'reference')
    train_one('V3 C=10 reference', v3.V3_FEATURES, v3.pipeline(10.0), 'reference')
    for c in (0.3, 3.0):
        train_one(f'V4 strict logistic C={c}', STRICT_FIELDS, logistic(c), 'strict')
    train_one('V4 strict boost A', STRICT_FIELDS, boosted(3, 220, 100, 0.045), 'strict')
    train_one('V4 strict boost B', STRICT_FIELDS, boosted(3, 320, 80, 0.035), 'strict')
    train_one('V4 strict extra trees', STRICT_FIELDS, trees(), 'strict')
    for c in (0.3, 3.0):
        train_one(f'V4 hybrid logistic C={c}', HYBRID_FIELDS, logistic(c), 'hybrid')
    train_one('V4 hybrid boost A', HYBRID_FIELDS, boosted(3, 220, 100, 0.045), 'hybrid')
    train_one('V4 hybrid boost B', HYBRID_FIELDS, boosted(3, 320, 80, 0.035), 'hybrid')
    # Blends are fixed before validation. No 2026 information enters selection.
    for group in ('strict', 'hybrid'):
        boosts = f'V4 {group} boost A'
        logistic_name = f'V4 {group} logistic C=3.0'
        for weight in (0.4, 0.65):
            name = f'V4 {group} blend {weight:.2f} boost'
            models[name] = {
                'group': group,
                'val': weight * models[boosts]['val'] + (1 - weight) * models[logistic_name]['val'],
                'review': weight * models[boosts]['review'] + (1 - weight) * models[logistic_name]['review'],
            }

    leaderboard = pd.DataFrame([
        dict(group=data['group'], **quality(y_val, data['val'], name, VALIDATE))
        for name, data in models.items()
    ]).sort_values(['winner_accuracy_pct', 'log_loss'], ascending=[False, True])

    # Highest winner accuracy on 2025; probability log-loss breaks ties.
    # No 2026 metric is involved in this selection.
    selected = {}
    for group in ('strict', 'hybrid'):
        subset = leaderboard[leaderboard['group'] == group]
        selected[group] = str(subset.iloc[0]['model'])

    comparisons = ['PPG baseline', 'V3 C=10 reference', selected['strict'], selected['hybrid']]
    # In unusual cases a selected name could overlap (not currently possible).
    metrics = []
    for season, y, key in ((VALIDATE, y_val, 'val'), (REVIEW, y_review, 'review')):
        for name in comparisons:
            metrics.append(quality(y, models[name][key], name, season))
    metrics_df = pd.DataFrame(metrics)

    # Paired game-level bootstrap confidence intervals on 2026 ACCURACY
    # DIFFERENCES. This is uncertainty about this historical sample, not a
    # prospective accuracy guarantee; 2026 previously reviewed in this project.
    baseline_correct = ((models['V3 C=10 reference']['review'] >= 0.5) == y_review).astype(int)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(y_review), size=(1500, len(y_review)))
    pair_rows = []
    for group in ('strict', 'hybrid'):
        name = selected[group]
        cmp_correct = ((models[name]['review'] >= 0.5) == y_review).astype(int)
        differences = cmp_correct - baseline_correct
        sampled = differences[indices].mean(axis=1) * 100
        pair_rows.append({
            'comparison': f'{name} minus V3',
            '2026_accuracy_difference_pp': round(float(differences.mean()) * 100, 3),
            'paired_95pct_bootstrap_lo_pp': round(float(np.percentile(sampled, 2.5)), 3),
            'paired_95pct_bootstrap_hi_pp': round(float(np.percentile(sampled, 97.5)), 3),
            'V4_correct_V3_wrong': int(((cmp_correct == 1) & (baseline_correct == 0)).sum()),
            'V3_correct_V4_wrong': int(((cmp_correct == 0) & (baseline_correct == 1)).sum()),
        })
    paired = pd.DataFrame(pair_rows)

    historic = review[['season', 'game_id', 'home_win']].copy()
    historic['v3_home_win_prob'] = models['V3 C=10 reference']['review']
    historic['v4_strict_home_win_prob'] = models[selected['strict']]['review']
    historic['v4_hybrid_home_win_prob'] = models[selected['hybrid']]['review']

    # Show honest reliability by bins; not instructions for choosing future games.
    reliability = []
    for label in ('v3_home_win_prob', 'v4_strict_home_win_prob', 'v4_hybrid_home_win_prob'):
        for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.00001)):
            mask = (historic[label] >= lo) & (historic[label] < hi)
            reliability.append({
                'model': label, 'predicted_home_win_probability_bin': f'{lo:.0%}–{min(hi,1):.0%}',
                'games': int(mask.sum()),
                'mean_predicted_home_win_pct': (round(float(historic.loc[mask, label].mean())*100, 2)
                                                 if mask.any() else None),
                'observed_home_win_pct': (round(float(historic.loc[mask, 'home_win'].mean())*100, 2)
                                          if mask.any() else None),
            })
    return metrics_df, leaderboard, paired, {'selected': selected, 'historic': historic,
                                             'reliability': pd.DataFrame(reliability)}


def html_report(metrics: pd.DataFrame, leaderboard: pd.DataFrame, paired: pd.DataFrame,
                coverage: pd.DataFrame, audit: dict, picked: dict, reliability: pd.DataFrame) -> str:
    def table(df):
        return df.to_html(index=False, border=0, escape=True, classes='result')
    return f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NCAA MBB V4 Research — Historical Comparison</title>
<style>body{{font-family:system-ui,Arial,sans-serif;max-width:1180px;margin:34px auto;padding:0 24px;color:#162332;line-height:1.5}}
h1,h2{{line-height:1.2}} .note{{border-left:4px solid #526884;background:#eef3f8;padding:14px 18px}}
.result{{border-collapse:collapse;width:100%;font-size:.92em;margin:12px 0 30px;overflow:auto;display:block}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #ddd;white-space:nowrap}}th{{background:#edf2f7}}
small{{color:#596778}} code,pre{{background:#f4f4f4;padding:3px}}</style></head><body>
<h1>NCAA Men's Basketball — V4 Historical Research</h1>
<small>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</small>
<div class="note"><b>Historical evaluation only.</b> No guaranteed accuracy improvement or 90% outcome.
Training: 2022–2024; candidate selection: 2025; reviewed historical comparison: 2026.
2026 has been examined previously and is <b>not</b> a pristine holdout.
V4 strict reconstructs its features using only games from previous calendar days.
V4 hybrid also uses original V3 features, whose upstream derivation has not been fully audited.</div>
<h2>Selection</h2><p>Chosen by <b>highest 2025 winner accuracy</b>, with lower log-loss
breaking ties. Strict: <b>{html.escape(picked['strict'])}</b>.
Hybrid: <b>{html.escape(picked['hybrid'])}</b>.</p>
<h2>2025 and 2026 comparisons — SAME GAMES</h2>{table(metrics)}
<h2>2026 paired accuracy difference versus V3</h2>{table(paired)}
<p>Bootstrap intervals quantify the uncertainty in the historical paired game-level
accuracy difference, not accuracy on future games. Results containing zero do not show a
clear improvement on this reviewed season.</p>
<h2>2025 model selection candidates</h2>{table(leaderboard)}
<h2>Coverage and filtering</h2>{table(coverage)}
<h2>2026 probability calibration</h2>{table(reliability)}
<h2>Chronology and feature audit</h2><pre>{html.escape(json.dumps(audit, indent=2, default=str))}</pre>
<h2>Interpretation</h2><p>Higher historical accuracy is not proof of improved future forecasting.
Data availability and reschedules can affect how much information was actually available on a past date.
The hybrid model reuses V3 upstream efficiency/strength ratings whose calculations have not been independently verified.
For prospective proof, record forecasts before a game and evaluate after its result without model changes.</p>
</body></html>'''


def run(raw: pd.DataFrame, totals: pd.DataFrame, scores: pd.DataFrame) -> dict:
    frame, audit = make_frame(raw, scores)
    metrics, leaderboard, paired, meta = model_predictions(frame)
    available = frame.groupby('season').size().rename('eligible_games_v4').reset_index()
    coverage = totals.copy()
    coverage['season'] = pd.to_numeric(coverage['season'], errors='coerce')
    coverage['completed_d1_games'] = pd.to_numeric(coverage['completed_d1_games'], errors='coerce')
    coverage = coverage.merge(available, on='season', how='left')
    coverage['eligible_games_v4'] = coverage['eligible_games_v4'].fillna(0).astype(int)
    coverage['coverage_pct'] = (100*coverage['eligible_games_v4']
                                / coverage['completed_d1_games']).round(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT/'model_comparison.csv', index=False)
    leaderboard.to_csv(OUTPUT/'selection_2025.csv', index=False)
    paired.to_csv(OUTPUT/'paired_differences_2026.csv', index=False)
    coverage.to_csv(OUTPUT/'coverage.csv', index=False)
    meta['historic'].to_csv(OUTPUT/'historical_2026_predictions.csv', index=False)
    meta['reliability'].to_csv(OUTPUT/'calibration_2026.csv', index=False)
    (OUTPUT/'chronology_audit.json').write_text(json.dumps(audit, indent=2, default=str), encoding='utf-8')
    (OUTPUT/'report.html').write_text(html_report(metrics, leaderboard, paired,
                 coverage, audit, meta['selected'], meta['reliability']), encoding='utf-8')
    summary = {'selected': meta['selected'], 'comparison': metrics.to_dict('records'),
               'paired_differences_2026': paired.to_dict('records'), 'audit': audit,
               'historical_only': True, 'review_season_untouched': False}
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print('\nV4 HISTORICAL RESEARCH — no future game selections')
    print(metrics.to_string(index=False))
    print('\nV4 - V3 paired differences:\n' + paired.to_string(index=False))
    print('\nGenerated: ' + str((OUTPUT/'report.html').resolve()))
    return summary


def self_test() -> None:
    rng = np.random.default_rng(11)
    # Generate ~22 games per team/season, with dates and pregame team fields.
    # All values are synthetic; no NCAA data is used here.
    rows, sc_rows = [], []
    for season in (2022, 2023, 2024, 2025, 2026):
        nteams = 24
        strengths = rng.normal(0, 10, nteams)
        for day_number in range(1, 101):
            day = date(season-1, 11, 1) + pd.Timedelta(days=day_number)
            h = (day_number*3) % nteams
            a = (day_number*3+7) % nteams
            for k in range(4):
                hh, aa = (h+k*2)%nteams, (a+k*2)%nteams
                if hh == aa:
                    continue
                diff = strengths[hh] - strengths[aa] + (0 if day_number % 5 == 0 else 3)
                margin = diff + rng.normal(0, 12)
                home_score = int(round(72 + margin/2))
                away_score = int(round(72 - margin/2))
                if home_score == away_score:
                    home_score += 1
                gid = f'{season}-{day_number}-{k}'
                sc_rows.append(dict(season=season, game_id=gid, game_date=str(day),
                    home_team_id=str(hh), away_team_id=str(aa), neutral_site=int(day_number%5==0),
                    home_score=home_score, away_score=away_score))
                rows.append(dict(season=season, game_id=gid, game_date=str(day),
                    neutral_site=int(day_number%5==0), home_score=home_score, away_score=away_score,
                    h_games_before=10, a_games_before=10,
                    h_rating_date=str(day-pd.Timedelta(days=1)),
                    a_rating_date=str(day-pd.Timedelta(days=1)),
                    h_ppg=75+strengths[hh]/4, a_ppg=75+strengths[aa]/4,
                    h_allowed=74-strengths[hh]/4, a_allowed=74-strengths[aa]/4,
                    h_raw_off=105+strengths[hh]/3, a_raw_off=105+strengths[aa]/3,
                    h_raw_def=100-strengths[hh]/3, a_raw_def=100-strengths[aa]/3,
                    h_raw_net=strengths[hh]/2, a_raw_net=strengths[aa]/2,
                    h_pace=69, a_pace=69,
                    h_recent_margin=strengths[hh]/2, a_recent_margin=strengths[aa]/2,
                    h_rest=2, a_rest=2, h_adj_net=strengths[hh], a_adj_net=strengths[aa],
                    h_sos=0.0, a_sos=0.0))
    raw = pd.DataFrame(rows)
    scores = pd.DataFrame(sc_rows)
    baseline = chronological_features(scores)
    assert len(baseline) == len(scores)
    # Regression test: changing any result on a day cannot alter same-day snapshots.
    changed = scores.copy()
    d = changed.loc[0, 'game_date']
    changed.loc[changed['game_date'] == d, 'home_score'] += 50
    past = chronological_features(changed)
    same = baseline.merge(past, on=['season', 'game_id'], suffixes=('_old', '_new'))
    same_day_ids = set(scores.loc[scores['game_date'] == d, 'game_id'])
    subset = same[same['game_id'].isin(same_day_ids)]
    for key in STRICT_FIELDS:
        if key != 'neutral_site':
            assert np.allclose(subset[f'{key}_old'], subset[f'{key}_new'], equal_nan=True), key
    totals = pd.DataFrame([{'season': y, 'completed_d1_games': 400} for y in (2022,2023,2024,2025,2026)])
    run(raw, totals, scores)
    assert (OUTPUT/'report.html').exists()
    print('Synthetic V4 self-test PASSED. Accuracy above is SIMULATED.')


if __name__ == '__main__':
    try:
        if '--self-test' in sys.argv:
            self_test()
        else:
            run(*fetch())
    except Exception as exc:
        print(f'V4 failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
