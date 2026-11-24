"""NCAA men's basketball V5: four-factor, historical-only research.

This file uses the same eligible games as V4, and adds opponent-adjusted
past efficiency, lagged 3-point statistics, early-season shrinkage/uncertainty,
and venue-specific history. No game-day or future game box score is a feature.

READ-ONLY DB. No upcoming selections. 2026 is a previously reviewed season,
not an untouched holdout or proof of future winner-prediction accuracy.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import v3_research as v3
import v4_research as v4

OUT = Path(os.environ.get('V5_OUTPUT_DIR', 'v5_results'))
SEED = 42
# Deliberately fixed before evaluating history, not optimized on 2026.
SHRINK_GAMES = 10.0
SHRINK_VENUE = 8.0
SHRINK_SHOTS = 120.0
SHRINK_SHOT_RATE = 0.34  # a fixed, disclosed prior, not fitted using future data
CARRYOVER_ELO = 0.20     # conservative previous-season carryover

# Actual shooting and possession inputs exist in public.team_game_stats.
# We LEFT JOIN so missing box scores do not delete games from a comparison.
BOX_SQL = """
SELECT g.season, g.game_id::text AS game_id,
       hs.points AS h_box_points, aws.points AS a_box_points,
       hs.possessions AS h_possessions, aws.possessions AS a_possessions,
       hs.three_m AS h_three_m, hs.three_a AS h_three_a,
       aws.three_m AS a_three_m, aws.three_a AS a_three_a,
       hs.fg_a AS h_fg_a, aws.fg_a AS a_fg_a
FROM public.games g
JOIN public.teams ht ON ht.team_id = g.home_team_id AND ht.is_d1 = 1
JOIN public.teams at ON at.team_id = g.away_team_id AND at.is_d1 = 1
LEFT JOIN public.team_game_stats hs
  ON hs.game_id = g.game_id AND hs.team_id = g.home_team_id
LEFT JOIN public.team_game_stats aws
  ON aws.game_id = g.game_id AND aws.team_id = g.away_team_id
WHERE g.season BETWEEN 2022 AND 2026
  AND g.completed = 1 AND g.neutral_site IN (0,1)
  AND g.game_date IS NOT NULL
  AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
  AND g.home_score <> g.away_score
  AND g.home_score >= 0 AND g.away_score >= 0
  AND NOT ((g.home_score = 0 AND g.away_score = 2)
        OR (g.home_score = 2 AND g.away_score = 0))
ORDER BY g.season, g.game_date, g.game_id
"""

OPP_FIELDS = ['v5_adjusted_net_gap', 'v5_adj_off_gap', 'v5_adj_def_gap',
              'v5_lagged_sos_gap', 'v5_adjusted_ppg_margin']
SHOOT_FIELDS = ['v5_three_pct_gap', 'v5_three_matchup_gap', 'v5_three_attempt_share_gap',
                'v5_three_volatility_sum', 'v5_shooting_games_min']
EARLY_FIELDS = ['v5_past_season_elo_gap', 'v5_elo_shrunk_gap',
                'v5_small_sample_uncertainty', 'v5_reliability_gap',
                'v5_min_box_games']
VENUE_FIELDS = ['v5_venue_context_gap', 'v5_venue_games_min',
                'v5_neutral_context_gap']
ALL_NEW = OPP_FIELDS + SHOOT_FIELDS + EARLY_FIELDS + VENUE_FIELDS
STRICT_FIELDS = list(dict.fromkeys(v4.STRICT_FIELDS + ALL_NEW))
HYBRID_FIELDS = list(dict.fromkeys(v4.HYBRID_FIELDS + ALL_NEW))


@dataclass
class TeamHistory:
    rating: float = 1500.0
    seeded_rating: float = 1500.0
    adj_off: list[float] = field(default_factory=list)
    adj_def: list[float] = field(default_factory=list)
    opp_adjnet: list[float] = field(default_factory=list)
    threes_m: float = 0.0
    threes_a: float = 0.0
    total_fga: float = 0.0
    shot_game_rates: list[float] = field(default_factory=list)
    opponent_threes_m: float = 0.0
    opponent_threes_a: float = 0.0
    adj_net_by_venue: dict[str, list[float]] = field(
        default_factory=lambda: {'home': [], 'away': [], 'neutral': []})
    box_games: int = 0

    def adj(self) -> tuple[float, float]:
        n = len(self.adj_off)
        factor = n / (n + SHRINK_GAMES)
        offense = 100.0 + factor * (float(np.mean(self.adj_off)) - 100.0) if n else 100.0
        defense = 100.0 + factor * (float(np.mean(self.adj_def)) - 100.0) if n else 100.0
        return offense, defense

    def shot(self) -> tuple[float, float, float, float]:
        # Beta-like attempt shrinkage. Missing stats mean prior, not zero made shots.
        pct = (self.threes_m + SHRINK_SHOTS * SHRINK_SHOT_RATE) / (
            self.threes_a + SHRINK_SHOTS)
        opponent_pct = (self.opponent_threes_m + SHRINK_SHOTS * SHRINK_SHOT_RATE) / (
            self.opponent_threes_a + SHRINK_SHOTS)
        share = self.threes_a / self.total_fga if self.total_fga else np.nan
        if len(self.shot_game_rates) >= 3:
            std = float(np.std(self.shot_game_rates, ddof=1))
            std *= len(self.shot_game_rates) / (len(self.shot_game_rates) + SHRINK_GAMES)
        else:
            std = np.nan
        return pct, opponent_pct, share, std

    def venue(self, venue: str) -> tuple[float, int]:
        at_venue = self.adj_net_by_venue[venue]
        total = [x for group in self.adj_net_by_venue.values() for x in group]
        if not at_venue or not total:
            return 0.0, len(at_venue)
        effect = float(np.mean(at_venue)) - float(np.mean(total))
        effect *= len(at_venue) / (len(at_venue) + SHRINK_VENUE)
        return effect, len(at_venue)


def fetch() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    url = os.getenv('DATABASE_URL')
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
            cur.execute(v4.SCORE_SQL)
            scores = pd.DataFrame(cur.fetchall())
            cur.execute(BOX_SQL)
            boxes = pd.DataFrame(cur.fetchall())
    if raw.empty or scores.empty:
        raise RuntimeError('Historical rows unavailable; check read-only database credentials')
    return raw, totals, scores, boxes


def valid_box(row) -> bool:
    """Only coherent double-sided shooting/possession records count."""
    try:
        for key in ('h_box_points', 'a_box_points', 'h_possessions', 'a_possessions',
                    'h_three_m', 'a_three_m', 'h_three_a', 'a_three_a', 'h_fg_a', 'a_fg_a'):
            if not np.isfinite(float(getattr(row, key))):
                return False
        return (float(row.h_box_points) == float(row.home_score)
                and float(row.a_box_points) == float(row.away_score)
                and float(row.h_possessions) > 0 and float(row.a_possessions) > 0
                and 0 <= float(row.h_three_m) <= float(row.h_three_a) <= float(row.h_fg_a)
                and 0 <= float(row.a_three_m) <= float(row.a_three_a) <= float(row.a_fg_a)
                and float(row.h_fg_a) > 0 and float(row.a_fg_a) > 0)
    except (TypeError, ValueError, AttributeError):
        return False


def replay_features(scores: pd.DataFrame, boxes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """For each date, freeze all team features; apply today's games AFTER features.

    Past opponent-adjusted results are calculated using the opponent's rating
    as it existed BEFORE the relevant historical game, never its eventual rating.
    """
    required_scores = {'season', 'game_id', 'game_date', 'home_team_id',
                       'away_team_id', 'home_score', 'away_score', 'neutral_site'}
    if not required_scores.issubset(scores):
        raise ValueError(f'Missing scoreboard columns: {required_scores - set(scores)}')
    if boxes.duplicated(['season', 'game_id']).any():
        raise ValueError('Duplicate game/team box statistics; inspect team_game_stats uniqueness')
    s = scores.copy()
    s['season'] = pd.to_numeric(s['season'], errors='coerce')
    s['day'] = pd.to_datetime(s['game_date'], utc=True, errors='coerce').dt.date
    s = s.dropna(subset=['season', 'day', 'game_id']).copy()
    s['game_id'] = s['game_id'].astype(str)
    s = s.merge(boxes, on=['season', 'game_id'], how='left', validate='one_to_one')
    s = s.sort_values(['season', 'day', 'game_id'], kind='stable')
    snapshots = []
    last_season_ratings: dict[str, float] = {}
    updated_boxes, missing_boxes = 0, 0

    for season, season_games in s.groupby('season', sort=True):
        def new_state(team_id):
            prior = last_season_ratings.get(str(team_id), 1500.0)
            seed = 1500.0 + CARRYOVER_ELO * (prior - 1500.0)
            return TeamHistory(rating=seed, seeded_rating=seed)

        teams = defaultdict(lambda: None)
        def state(team_id: str) -> TeamHistory:
            if team_id not in teams:
                teams[team_id] = new_state(team_id)
            return teams[team_id]

        for day, day_games in season_games.groupby('day', sort=True):
            updates = []
            for g in day_games.itertuples(index=False):
                home, away = state(str(g.home_team_id)), state(str(g.away_team_id))
                h_off, h_def = home.adj()
                a_off, a_def = away.adj()
                h_sh, h_sh_allowed, h_share, h_var = home.shot()
                a_sh, a_sh_allowed, a_share, a_var = away.shot()
                h_hvenue, n_hvenue = home.venue('home')
                a_avenue, n_avenue = away.venue('away')
                h_neutral, n_hneutral = home.venue('neutral')
                a_neutral, n_aneutral = away.venue('neutral')
                is_neutral = int(g.neutral_site) == 1
                h_early = home.box_games / (home.box_games + SHRINK_GAMES)
                a_early = away.box_games / (away.box_games + SHRINK_GAMES)
                # Opponent-adjusted efficiency, reconstructed from prior results.
                h_net, a_net = h_off - h_def, a_off - a_def
                h_sos = float(np.mean(home.opp_adjnet)) if home.opp_adjnet else 0.0
                a_sos = float(np.mean(away.opp_adjnet)) if away.opp_adjnet else 0.0
                snapshots.append({
                    'season': int(season), 'game_id': str(g.game_id),
                    'v5_adj_off_gap': h_off - a_off,
                    'v5_adj_def_gap': a_def - h_def,  # lower defense is better
                    'v5_adjusted_net_gap': h_net - a_net,
                    'v5_lagged_sos_gap': h_sos - a_sos,
                    'v5_adjusted_ppg_margin': 0.5 * ((h_off + a_def) - (a_off + h_def)),
                    'v5_three_pct_gap': h_sh - a_sh,
                    'v5_three_matchup_gap': (h_sh - a_sh_allowed) - (a_sh - h_sh_allowed),
                    'v5_three_attempt_share_gap': h_share - a_share,
                    'v5_three_volatility_sum': h_var + a_var,
                    'v5_shooting_games_min': min(len(home.shot_game_rates),
                                                len(away.shot_game_rates)),
                    'v5_past_season_elo_gap': home.seeded_rating - away.seeded_rating,
                    'v5_elo_shrunk_gap': (home.rating - 1500.) * h_early
                                          - (away.rating - 1500.) * a_early,
                    'v5_small_sample_uncertainty': (1 / np.sqrt(home.box_games + 1)
                                                     + 1 / np.sqrt(away.box_games + 1)),
                    'v5_reliability_gap': h_early - a_early,
                    'v5_min_box_games': min(home.box_games, away.box_games),
                    'v5_venue_context_gap': 0.0 if is_neutral else h_hvenue - a_avenue,
                    'v5_venue_games_min': (min(n_hneutral, n_aneutral) if is_neutral
                                           else min(n_hvenue, n_avenue)),
                    'v5_neutral_context_gap': h_neutral - a_neutral if is_neutral else 0.0,
                    'v5_source_box_available': int(valid_box(g)),
                })
                # Compute all game-dependent transformations from *frozen* states;
                # don't commit until all same-date snapshots are captured.
                p_home = 1 / (1 + 10 ** ((away.rating-home.rating - (0 if is_neutral else 60))/400))
                margin = abs(float(g.home_score) - float(g.away_score))
                k = 22.0 * min(1.6, 0.75 + 0.27 * np.log1p(margin))
                delta = k * (float(g.home_score > g.away_score) - p_home)
                if valid_box(g):
                    poss = (float(g.h_possessions) + float(g.a_possessions)) / 2.0
                    h_eff, a_eff = 100 * float(g.home_score) / poss, 100 * float(g.away_score) / poss
                    # Each per-game adjusted rating is locked using prior-day
                    # opponent performance, not post-game or season-end stats.
                    h_adjusted_off = h_eff + (100.0 - a_def)
                    h_adjusted_def = a_eff + (100.0 - a_off)
                    a_adjusted_off = a_eff + (100.0 - h_def)
                    a_adjusted_def = h_eff + (100.0 - h_off)
                    h_adj_opp_net, a_adj_opp_net = a_net, h_net
                    box_values = (h_adjusted_off, h_adjusted_def,
                                  a_adjusted_off, a_adjusted_def,
                                  h_adj_opp_net, a_adj_opp_net)
                else:
                    box_values = None
                updates.append((str(g.home_team_id), str(g.away_team_id), g,
                                delta, box_values, is_neutral))
            for h_id, a_id, g, delta, box_values, is_neutral in updates:
                home, away = state(h_id), state(a_id)
                home.rating += delta
                away.rating -= delta
                if box_values is None:
                    missing_boxes += 1
                    continue
                updated_boxes += 1
                h_off_g, h_def_g, a_off_g, a_def_g, h_opp_net, a_opp_net = box_values
                home.adj_off.append(h_off_g)
                home.adj_def.append(h_def_g)
                home.opp_adjnet.append(h_opp_net)
                away.adj_off.append(a_off_g)
                away.adj_def.append(a_def_g)
                away.opp_adjnet.append(a_opp_net)
                home.box_games += 1
                away.box_games += 1
                home.threes_m += float(g.h_three_m)
                home.threes_a += float(g.h_three_a)
                home.total_fga += float(g.h_fg_a)
                home.opponent_threes_m += float(g.a_three_m)
                home.opponent_threes_a += float(g.a_three_a)
                away.threes_m += float(g.a_three_m)
                away.threes_a += float(g.a_three_a)
                away.total_fga += float(g.a_fg_a)
                away.opponent_threes_m += float(g.h_three_m)
                away.opponent_threes_a += float(g.h_three_a)
                if g.h_three_a >= 8:
                    home.shot_game_rates.append(float(g.h_three_m) / float(g.h_three_a))
                if g.a_three_a >= 8:
                    away.shot_game_rates.append(float(g.a_three_m) / float(g.a_three_a))
                h_venue = 'neutral' if is_neutral else 'home'
                a_venue = 'neutral' if is_neutral else 'away'
                home.adj_net_by_venue[h_venue].append(h_off_g - h_def_g)
                away.adj_net_by_venue[a_venue].append(a_off_g - a_def_g)
        # Cross-season prior uses ONLY a completed earlier season.
        last_season_ratings = {tid: st.rating for tid, st in teams.items()}
    return pd.DataFrame(snapshots), {
        'scoreboard_games_replayed': len(s),
        'games_with_valid_box_scores': updated_boxes,
        'games_without_valid_box_scores': missing_boxes,
        'box_score_valid_pct': round(100 * updated_boxes / max(1, len(s)), 2),
        'same_day_results_used_as_features': False,
        'opponent_rating_timestamp': 'start of preceding-game day',
        'previous_season_elo_carryover': CARRYOVER_ELO,
        'shooting_prior_attempts': SHRINK_SHOTS,
        'shooting_prior_percentage': SHRINK_SHOT_RATE,
    }


def make_frame(raw: pd.DataFrame, scores: pd.DataFrame, boxes: pd.DataFrame):
    frame, audit = v4.make_frame(raw, scores)
    lagged, lag_audit = replay_features(scores, boxes)
    frame = frame.merge(lagged, on=['season', 'game_id'], how='left',
                        validate='one_to_one')
    if frame[ALL_NEW].isna().all(axis=1).any():
        raise RuntimeError('Some eligible games lack lagged V5 feature records')
    for key in ALL_NEW:
        frame[key] = pd.to_numeric(frame[key], errors='coerce')
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    audit.update(lag_audit)
    audit.update({
        'strict_v5_uses_only_previous_dates_for_new_features': True,
        'v3_and_v4_upstream_hybrid_fields_independently_audited': False,
        'eligible_sample_unchanged_from_v4': True,
        'candidate_selection_year': v4.VALIDATE,
        'review_year_previously_examined': v4.REVIEW,
        'missing_shooting_volatility_pct': round(100 * frame['v5_three_volatility_sum'].isna().mean(), 2),
        'missing_three_attempt_share_pct': round(100 * frame['v5_three_attempt_share_gap'].isna().mean(), 2),
    })
    return frame, audit


def predict_and_evaluate(frame: pd.DataFrame):
    """Reference V4 recomputed on exact same rows; V5 selected only in 2025."""
    old_metrics, old_selection, old_paired, old_meta = v4.model_predictions(frame)
    train = frame[frame.season.isin(v4.TRAIN)]
    val = frame[frame.season == v4.VALIDATE]
    review = frame[frame.season == v4.REVIEW]
    y_train, y_val, y_review = [d.home_win.astype(int).to_numpy() for d in (train, val, review)]
    candidates: dict[str, dict] = {}

    def add(name: str, family: str, fields: list[str], model):
        model.fit(train[fields], y_train)
        candidates[name] = {
            'family': family,
            'val': model.predict_proba(val[fields])[:, 1],
            'review': model.predict_proba(review[fields])[:, 1],
        }

    # First isolate the contribution of EACH new data category (same linear
    # model, fixed C): no feature family is awarded credit by 2026 selection.
    for label, added in [('opponent', OPP_FIELDS), ('three_point', SHOOT_FIELDS),
                         ('early', EARLY_FIELDS), ('venue', VENUE_FIELDS)]:
        add(f'V5 ablation +{label}', 'ablation', list(dict.fromkeys(v4.STRICT_FIELDS + added)),
            v4.logistic(3.0))
    # Controlled comparisons of only lagged inputs (strict) versus the hybrid
    # set retaining existing V3 upstream features (not fully provenance-audited).
    for family, fields in [('strict', STRICT_FIELDS), ('hybrid', HYBRID_FIELDS)]:
        add(f'V5 {family} logistic C=0.3', family, fields, v4.logistic(0.3))
        add(f'V5 {family} logistic C=3.0', family, fields, v4.logistic(3.0))
        add(f'V5 {family} boost A', family, fields, v4.boosted(3, 220, 100, 0.045))
        add(f'V5 {family} boost B', family, fields, v4.boosted(3, 320, 80, 0.035))
        boost = candidates[f'V5 {family} boost A']
        linear = candidates[f'V5 {family} logistic C=3.0']
        candidates[f'V5 {family} blend'] = {
            'family': family,
            'val': 0.6 * boost['val'] + 0.4 * linear['val'],
            'review': 0.6 * boost['review'] + 0.4 * linear['review'],
        }

    leaderboard = pd.DataFrame([
        dict(family=obj['family'], **v4.quality(y_val, obj['val'], name, v4.VALIDATE))
        for name, obj in candidates.items()
    ]).sort_values(['family', 'winner_accuracy_pct', 'log_loss'],
                   ascending=[True, False, True])
    chosen = {}
    for family in ('strict', 'hybrid'):
        options = leaderboard[leaderboard['family'] == family]
        chosen[family] = str(options.iloc[0]['model'])

    old_selected = old_meta['selected']
    # Paired comparisons share exact same 2026 game IDs and outcome labels.
    new_review = review[['season', 'game_id', 'home_win']].copy()
    v4_old = old_meta['historic']
    assert np.array_equal(new_review['game_id'].to_numpy(), v4_old['game_id'].to_numpy())
    new_review['v4_strict_home_win_prob'] = v4_old['v4_strict_home_win_prob'].to_numpy()
    new_review['v4_hybrid_home_win_prob'] = v4_old['v4_hybrid_home_win_prob'].to_numpy()
    new_review['v3_home_win_prob'] = v4_old['v3_home_win_prob'].to_numpy()
    new_review['v5_strict_home_win_prob'] = candidates[chosen['strict']]['review']
    new_review['v5_hybrid_home_win_prob'] = candidates[chosen['hybrid']]['review']
    new_review['v5_game_shooting_sample_min'] = review['v5_shooting_games_min'].to_numpy()
    new_review['v5_game_box_sample_min'] = review['v5_min_box_games'].to_numpy()
    new_review['neutral_site'] = review['neutral_site'].to_numpy()

    metrics = []
    for season, y, step in [(v4.VALIDATE, y_val, 'val'), (v4.REVIEW, y_review, 'review')]:
        for name, data in candidates.items():
            if data['family'] != 'ablation' and name not in chosen.values():
                continue
            metrics.append(v4.quality(y, data[step], name, season))
    all_metrics = pd.concat([old_metrics, pd.DataFrame(metrics)], ignore_index=True)
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(y_review), size=(1000, len(y_review)))
    paired = []
    for family in ('strict', 'hybrid'):
        new_prob = candidates[chosen[family]]['review']
        old_prob = new_review[f'v4_{family}_home_win_prob'].to_numpy()
        new_correct = ((new_prob >= .5) == y_review).astype(int)
        old_correct = ((old_prob >= .5) == y_review).astype(int)
        diff = new_correct - old_correct
        sampled = diff[picks].mean(axis=1) * 100
        paired.append({
            'comparison': f'V5 {family} minus V4 {family}',
            '2026_accuracy_difference_pp': round(float(diff.mean())*100, 3),
            '95pct_bootstrap_low_pp': round(float(np.percentile(sampled, 2.5)), 3),
            '95pct_bootstrap_high_pp': round(float(np.percentile(sampled, 97.5)), 3),
            'V5_right_V4_wrong': int(((new_correct == 1) & (old_correct == 0)).sum()),
            'V4_right_V5_wrong': int(((new_correct == 0) & (old_correct == 1)).sum()),
        })
    paired = pd.DataFrame(paired)

    # Factor-by-factor comparisons are exploratory and do not claim causality.
    ablations = []
    for name, data in candidates.items():
        if data['family'] != 'ablation':
            continue
        for season, y, key in ((v4.VALIDATE, y_val, 'val'), (v4.REVIEW, y_review, 'review')):
            ablations.append(v4.quality(y, data[key], name, season))
    return all_metrics, leaderboard, paired, new_review, chosen, old_selected, pd.DataFrame(ablations)


def html_report(metrics, leaderboard, paired, ablations, coverage, audit, chosen, old_selected):
    def tbl(df):
        return df.to_html(index=False, border=0, escape=True, classes='result')
    return f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NCAA MBB V5 — Four Factor Research</title>
<style>body{{font-family:system-ui,Arial;max-width:1200px;margin:32px auto;padding:0 24px;color:#15222c;line-height:1.5}}
h1,h2{{line-height:1.25}}.note{{border-left:4px solid #51657f;background:#edf3f7;padding:14px 18px}}
.result{{border-collapse:collapse;display:block;overflow:auto;width:100%;margin:12px 0 26px;font-size:.92em}}
th,td{{border-bottom:1px solid #ddd;padding:8px 11px;text-align:left;white-space:nowrap}}
th{{background:#eef2f8}}pre{{white-space:pre-wrap;background:#f7f8fa;padding:14px}}</style></head><body>
<h1>NCAA MBB — V5 Four-Factor Historical Research</h1>
<p>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
<div class="note"><b>Historical research only.</b> No current-game selections or wagering guidance.
All new V5 box-score features are frozen before the calendar day of each historical game.
Train: 2022–2024. Select by 2025 accuracy. Historical review: 2026, which <b>has already been repeatedly inspected</b>
and is <b>not a fresh holdout</b>. V5 hybrid retains unverified upstream V3 rating features.
Improvement is neither guaranteed nor demonstrated until the report is produced on your database.</div>
<h2>Added information</h2><p><b>Opponent:</b> lagged, shrunk points per 100 possessions adjusted for opponents' previously known strength;
<b>3-point:</b> past shooting, attempt share, opponent shooting allowed, volatility and sample size;
<b>early season:</b> conservatively carried prior-season Elo and sample-size uncertainty;
<b>venue:</b> shrunk home, road, and neutral performance compared with each team's prior overall level.</p>
<h2>Selected configurations (2025 only)</h2><p>V4 references: strict <b>{html.escape(old_selected['strict'])}</b>,
hybrid <b>{html.escape(old_selected['hybrid'])}</b>. New V5: strict <b>{html.escape(chosen['strict'])}</b>,
hybrid <b>{html.escape(chosen['hybrid'])}</b>.</p>
<h2>Winner accuracy and probability quality: same games</h2>{tbl(metrics)}
<h2>Paired difference: V5 versus V4 on 2026</h2>{tbl(paired)}
<p>Intervals containing zero indicate no clear advantage on the reviewed 2026 sample.
This does not establish future-game accuracy.</p>
<h2>Individual factor ablations (exploratory)</h2>{tbl(ablations)}
<h2>2025 candidate leaderboard</h2>{tbl(leaderboard)}
<h2>Coverage</h2>{tbl(coverage)}
<h2>Audit and limitations</h2><pre>{html.escape(json.dumps(audit, indent=2, default=str))}</pre>
<p>Important: some earlier V3 fields included in hybrid models have not been independently
proven leakage-free by upstream processing audits; and date-only checks cannot prove
that a historical record existed before the relevant game. Prospective evaluation
requires timestamped pre-game features.</p></body></html>'''


def run(raw, totals, scores, boxes):
    frame, audit = make_frame(raw, scores, boxes)
    metrics, leaderboard, paired, historic, chosen, old_selected, ablations = predict_and_evaluate(frame)
    covered = frame.groupby('season').size().rename('eligible_games').reset_index()
    coverage = totals.copy()
    coverage['season'] = pd.to_numeric(coverage['season'], errors='coerce')
    coverage['completed_d1_games'] = pd.to_numeric(coverage['completed_d1_games'], errors='coerce')
    coverage = coverage.merge(covered, on='season', how='left')
    coverage['eligible_games'] = coverage['eligible_games'].fillna(0).astype(int)
    coverage['coverage_pct'] = (coverage['eligible_games'] / coverage['completed_d1_games']*100).round(2)
    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT/'model_comparison.csv', index=False)
    leaderboard.to_csv(OUT/'selection_2025.csv', index=False)
    paired.to_csv(OUT/'paired_differences_2026.csv', index=False)
    ablations.to_csv(OUT/'factor_ablation.csv', index=False)
    coverage.to_csv(OUT/'coverage.csv', index=False)
    historic.to_csv(OUT/'historical_2026_predictions.csv', index=False)
    (OUT/'chronology_audit.json').write_text(json.dumps(audit, indent=2, default=str), encoding='utf-8')
    (OUT/'report.html').write_text(html_report(metrics, leaderboard, paired, ablations,
                             coverage, audit, chosen, old_selected), encoding='utf-8')
    summary = {'selected_v5': chosen, 'selected_v4': old_selected,
               'comparison': metrics.to_dict('records'),
               'paired_differences_2026': paired.to_dict('records'),
               'audit': audit, 'historical_only': True, 'review_year_untouched': False}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print('\nV5 HISTORICAL RESEARCH — no upcoming game selection')
    print(metrics.to_string(index=False))
    print('\nV5 vs V4 paired:\n' + paired.to_string(index=False))
    print('Report: ' + str((OUT/'report.html').resolve()))
    return summary


def self_test():
    """Synthetic games with multiple games on the same day and season carryover."""
    rng = np.random.default_rng(152)
    raw, scores, boxes = [], [], []
    for season in range(2022, 2027):
        strengths = rng.normal(0, 10, 28)
        for d in range(1, 118):
            day = date(season-1, 11, 1) + pd.Timedelta(days=d)
            for i in range(5):
                h, a = ((d * 3 + 2*i) % 28), ((d * 3 + 2*i + 7) % 28)
                gap = strengths[h] - strengths[a]
                home_score = int(round(72 + (gap + rng.normal(0, 11))/2))
                away_score = int(round(72 - (gap + rng.normal(0, 11))/2))
                if home_score == away_score:
                    home_score += 1
                neutral = int(d % 5 == 0)
                gid = f'{season}-{d}-{i}'
                scores.append(dict(season=season, game_id=gid, game_date=str(day),
                    home_team_id=str(h), away_team_id=str(a), neutral_site=neutral,
                    home_score=home_score, away_score=away_score))
                raw.append(dict(season=season, game_id=gid, game_date=str(day),
                    neutral_site=neutral, home_score=home_score, away_score=away_score,
                    h_games_before=12, a_games_before=12,
                    h_rating_date=str(day-pd.Timedelta(days=1)),
                    a_rating_date=str(day-pd.Timedelta(days=1)),
                    h_ppg=72+strengths[h]/5, a_ppg=72+strengths[a]/5,
                    h_allowed=71-strengths[h]/5, a_allowed=71-strengths[a]/5,
                    h_raw_off=105+strengths[h]/3, a_raw_off=105+strengths[a]/3,
                    h_raw_def=100-strengths[h]/3, a_raw_def=100-strengths[a]/3,
                    h_raw_net=strengths[h]/2, a_raw_net=strengths[a]/2,
                    h_pace=69, a_pace=69,
                    h_recent_margin=strengths[h]/2, a_recent_margin=strengths[a]/2,
                    h_rest=2, a_rest=2, h_adj_net=strengths[h], a_adj_net=strengths[a],
                    h_sos=0, a_sos=0))
                h_fga, a_fga = 60, 60
                h_3a, a_3a = 22, 23
                h_3m = int(rng.binomial(h_3a, .34 + np.clip(gap/500, -.07, .07)))
                a_3m = int(rng.binomial(a_3a, .34 - np.clip(gap/500, -.07, .07)))
                boxes.append(dict(season=season, game_id=gid,
                    h_box_points=home_score, a_box_points=away_score,
                    h_possessions=69, a_possessions=69,
                    h_three_m=h_3m, h_three_a=h_3a, h_fg_a=h_fga,
                    a_three_m=a_3m, a_three_a=a_3a, a_fg_a=a_fga))
    raw, scores, boxes = pd.DataFrame(raw), pd.DataFrame(scores), pd.DataFrame(boxes)
    original, _ = replay_features(scores, boxes)
    mutated = scores.copy()
    day0 = mutated.loc[0, 'game_date']
    changed_mask = mutated.game_date == day0
    mutated.loc[changed_mask, 'home_score'] += 22
    modified, _ = replay_features(mutated, boxes)
    same = original.merge(modified, on=['season', 'game_id'], suffixes=('_original', '_changed'))
    ids = set(scores.loc[scores.game_date == day0, 'game_id'])
    on_date = same[same.game_id.isin(ids)]
    for col in ALL_NEW:
        assert np.allclose(on_date[col+'_original'], on_date[col+'_changed'], equal_nan=True), col
    # Changing same-day 3PT data also cannot affect *that day's* predictions.
    boxes_changed = boxes.copy()
    boxes_changed.loc[boxes_changed.game_id.isin(ids), 'h_three_m'] = 0
    modified_shots, _ = replay_features(scores, boxes_changed)
    same_shots = original.merge(modified_shots, on=['season', 'game_id'], suffixes=('_original', '_changed'))
    on_date_shots = same_shots[same_shots.game_id.isin(ids)]
    for col in ALL_NEW:
        assert np.allclose(on_date_shots[col+'_original'], on_date_shots[col+'_changed'], equal_nan=True), col
    total = pd.DataFrame([{'season': s, 'completed_d1_games': 585} for s in range(2022, 2027)])
    run(raw, total, scores, boxes)
    assert (OUT/'report.html').exists()
    print('V5 SYNTHETIC SELF-TEST PASSED. THESE ACCURACY RESULTS ARE SIMULATED.')


if __name__ == '__main__':
    try:
        if '--self-test' in sys.argv:
            self_test()
        else:
            run(*fetch())
    except Exception as exc:
        print(f'V5 failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
