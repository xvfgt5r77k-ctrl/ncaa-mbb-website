"""V6 NCAA men's basketball retrospective feature research.

NO wager automation or upcoming selections. READ-ONLY PostgreSQL. All new
scoreboard/box-score features replay only information from PREVIOUS dates.
Historical 2026 has been examined repeatedly and is NOT an untouched holdout.

Requires existing v3_research.py, v4_research.py, v5_research.py in repo root.
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

import v4_research as v4
import v5_research as v5

OUT = Path(os.getenv('V6_OUTPUT_DIR', 'v6_results'))
SEED = 61
# Fixed before real historical evaluation. These are smoothing prior strengths,
# not optimized on the repeatedly inspected 2026 season.
PRIOR_GAMES = 8.0
PRIOR_2P_RATE = .49
PRIOR_EFG = .50
PRIOR_TOV_POSS = .17
PRIOR_OREB_RATE = .28
PRIOR_FT_RATE = .28

SHOOTING = ['v6_efg_gap', 'v6_efg_matchup_gap', 'v6_two_pct_gap',
            'v6_two_matchup_gap', 'v6_two_attempt_share_gap']
FOUR_FACTORS = ['v6_tov_gap', 'v6_tov_matchup_gap', 'v6_orb_rate_gap',
                'v6_orb_matchup_gap', 'v6_ft_rate_gap', 'v6_ft_matchup_gap']
OPPONENT = ['v6_elo_opponent_adjusted_eff_gap']
CONTEXT = ['v6_neutral_elo_gap', 'v6_early_elo_gap',
           'v6_early_sample_uncertainty', 'v6_venue_adjusted_gap']
PLAYER = ['v6_reported_out_gap', 'v6_reported_questionable_gap']
ALL = SHOOTING + FOUR_FACTORS + OPPONENT + CONTEXT

# Try alternate column spellings in different ESPN import schemas.
ALIASES = {
    'fg_m': ('fg_m', 'field_goals_made'),
    'fg_a': ('fg_a', 'field_goals_attempted'),
    'three_m': ('three_m', 'three_point_field_goals_made', 'three_pt_made'),
    'three_a': ('three_a', 'three_point_field_goals_attempted', 'three_pt_attempted'),
    'turnovers': ('turnovers', 'turnover', 'tov', 'to_count'),
    'off_rebounds': ('off_rebounds', 'offensive_rebounds', 'orb', 'oreb'),
    'def_rebounds': ('def_rebounds', 'defensive_rebounds', 'drb', 'dreb'),
    'ft_a': ('ft_a', 'fta', 'free_throws_attempted', 'free_throw_attempts'),
    'ft_m': ('ft_m', 'ftm', 'free_throws_made', 'free_throw_makes'),
}


def optional_data():
    """Discover actual schema; absent columns become SQL NULL, never zero.

    Availability is a *disabled-by-default diagnostic*. The current dashboard's
    conference injury_reports are often current-season only, with team names
    instead of stable IDs. They cannot legitimately backfill 2022-26 lineups.
    """
    url = os.getenv('DATABASE_URL')
    if not url:
        raise RuntimeError('GitHub Actions secret DATABASE_URL is missing')
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
    with psycopg.connect(url, sslmode='require', connect_timeout=30,
                         prepare_threshold=None, row_factory=dict_row) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema='public' AND table_name='team_game_stats'
            """)
            existing = {str(r['column_name']) for r in cur.fetchall()}
            mapped = {name: next((c for c in choices if c in existing), None)
                      for name, choices in ALIASES.items()}
            fields = [sql.SQL('g.season'), sql.SQL('g.game_id::text AS game_id')]
            for side, alias in (('home', 'hs'), ('away', 'aws')):
                for friendly, physical in mapped.items():
                    expression = (sql.SQL('NULL::numeric') if physical is None else
                                  sql.SQL('{}.{}').format(sql.Identifier(alias), sql.Identifier(physical)))
                    fields.append(sql.SQL('{} AS {}').format(
                        expression, sql.Identifier(side+'_'+friendly)))
            # Retrieve only completed-score pairs. DO NOT let optional stats
            # change the population of games used in v3/v4/v5 comparisons.
            statement = sql.SQL('''
                SELECT {} FROM public.games g
                LEFT JOIN public.team_game_stats hs
                  ON hs.game_id=g.game_id AND hs.team_id=g.home_team_id
                LEFT JOIN public.team_game_stats aws
                  ON aws.game_id=g.game_id AND aws.team_id=g.away_team_id
                WHERE g.season BETWEEN 2022 AND 2026
                  AND g.completed=1 AND g.home_score IS NOT NULL
                  AND g.away_score IS NOT NULL
            ''').format(sql.SQL(', ').join(fields))
            cur.execute(statement)
            optional = pd.DataFrame(cur.fetchall())
            cur.execute("""
                SELECT table_name, column_name FROM information_schema.columns
                WHERE table_schema='public' AND table_name IN ('injury_reports','games','teams')
            """)
            schema = defaultdict(set)
            for r in cur.fetchall():
                schema[r['table_name']].add(r['column_name'])
            # Requiring IDs + pre-tipoff timestamps prevents ambiguous player
            # matches and postgame reports from leaking into historical data.
            required = {'game_id','team_id','status','reported_at','retrieved_at'}
            availability_ready = required <= schema['injury_reports'] and (
                'source_player_id' in schema['injury_reports'] or
                'player_id' in schema['injury_reports']) and 'start_time_utc' in schema['games']
            availability = pd.DataFrame()
            if availability_ready:
                player_col = ('source_player_id' if 'source_player_id' in schema['injury_reports']
                              else 'player_id')
                # Only a last-known report BEFORE tipoff is valid; even if a
                # provider backdated reported_at, retrieved_at must be earlier.
                q = sql.SQL('''
                    SELECT r.game_id::text game_id, r.team_id::text team_id,
                      r.{}::text player_id, r.status::text status,
                      r.reported_at, r.retrieved_at,
                      NULLIF(g.start_time_utc::text, '')::timestamptz AS tipoff_utc
                    FROM public.injury_reports r
                    JOIN public.games g ON r.game_id=g.game_id
                    WHERE g.season BETWEEN 2022 AND 2026
                      AND r.reported_at IS NOT NULL AND r.retrieved_at IS NOT NULL
                      AND r.reported_at::timestamptz < NULLIF(g.start_time_utc::text,'')::timestamptz
                      AND r.retrieved_at::timestamptz < NULLIF(g.start_time_utc::text,'')::timestamptz
                ''').format(sql.Identifier(player_col))
                cur.execute(q)
                availability = pd.DataFrame(cur.fetchall())
    audit = {
        'team_game_stats_mapped_columns': mapped,
        'optional_stat_groups_complete': {
            'shooting': all(mapped.get(k) for k in ('fg_m','fg_a','three_m','three_a')),
            'turnovers': mapped['turnovers'] is not None,
            'rebounds': mapped['off_rebounds'] is not None and mapped['def_rebounds'] is not None,
            'free_throws': mapped['ft_a'] is not None,
        },
        'player_availability_historical_join_possible': availability_ready,
        'player_availability_rows_pretipoff': len(availability),
        'player_availability_note': ('Only strict game/team/player IDs with both report and retrieval timestamps are eligible.'
            if availability_ready else 'Historical game/team/player IDs or pretipoff timestamps absent; player features disabled.'),
    }
    return optional, availability, audit


def number(value):
    try:
        val = float(value)
        return val if np.isfinite(val) else np.nan
    except (TypeError, ValueError):
        return np.nan


def ratio(num: float, den: float, fallback: float, prior: float) -> float:
    return ((num + prior * fallback)/(den + prior)) if den >= 0 else np.nan


@dataclass
class Team:
    n_shots: int = 0
    n_tov: int = 0
    n_reb: int = 0
    n_ft: int = 0
    fgm: float = 0.
    fga: float = 0.
    three_m: float = 0.
    three_a: float = 0.
    opp_fgm: float = 0.
    opp_fga: float = 0.
    opp_three_m: float = 0.
    opp_three_a: float = 0.
    tov: float = 0.
    poss: float = 0.
    opp_tov: float = 0.
    opp_poss: float = 0.
    orebs: float = 0.
    opp_drebs: float = 0.
    opp_orebs: float = 0.
    drebs: float = 0.
    fta: float = 0.
    ft_fga: float = 0.
    opp_fta: float = 0.
    opp_ft_fga: float = 0.
    # Lagged, one-shot strength-adjusted performance per game.
    adjusted_eff: list[float] = field(default_factory=list)
    venue_nets: dict[str, list[float]] = field(default_factory=lambda: {
        'home': [], 'away': [], 'neutral': []})

    def shooting(self):
        # Made/attempted totals from earlier games only.
        two_m = self.fgm-self.three_m
        two_a = self.fga-self.three_a
        opp_two_m = self.opp_fgm-self.opp_three_m
        opp_two_a = self.opp_fga-self.opp_three_a
        efg = ratio(self.fgm + .5*self.three_m, self.fga, PRIOR_EFG, PRIOR_GAMES*60)
        opp_efg = ratio(self.opp_fgm + .5*self.opp_three_m, self.opp_fga,
                        PRIOR_EFG, PRIOR_GAMES*60)
        two = ratio(two_m, two_a, PRIOR_2P_RATE, PRIOR_GAMES*35)
        opp_two = ratio(opp_two_m, opp_two_a, PRIOR_2P_RATE, PRIOR_GAMES*35)
        share = self.three_a / self.fga if self.fga else np.nan
        return efg, opp_efg, two, opp_two, share

    def four(self):
        tov_rate = ratio(self.tov, self.poss, PRIOR_TOV_POSS, PRIOR_GAMES*70)
        tov_allowed = ratio(self.opp_tov, self.opp_poss, PRIOR_TOV_POSS, PRIOR_GAMES*70)
        orb = ratio(self.orebs, self.orebs+self.opp_drebs, PRIOR_OREB_RATE, PRIOR_GAMES*35)
        orb_allowed = ratio(self.opp_orebs, self.opp_orebs+self.drebs,
                            PRIOR_OREB_RATE, PRIOR_GAMES*35)
        ft_rate = ratio(self.fta, self.ft_fga, PRIOR_FT_RATE, PRIOR_GAMES*60)
        ft_allowed = ratio(self.opp_fta, self.opp_ft_fga, PRIOR_FT_RATE, PRIOR_GAMES*60)
        return tov_rate, tov_allowed, orb, orb_allowed, ft_rate, ft_allowed

    def venue(self, place):
        pool = [x for vals in self.venue_nets.values() for x in vals]
        venue = self.venue_nets[place]
        if not pool or not venue:
            return 0.
        return (float(np.mean(venue))-float(np.mean(pool))) * len(venue)/(len(venue)+8)


def replay(scores: pd.DataFrame, base_boxes: pd.DataFrame, optional: pd.DataFrame):
    """Capture every game before applying ANY outcomes/box scores from its day."""
    if optional.duplicated(['season','game_id']).any():
        raise ValueError('Duplicate optional box rows; verify team_game_stats uniqueness')
    s = scores.copy()
    s['day'] = pd.to_datetime(s['game_date'], utc=True, errors='coerce').dt.date
    s = s.merge(base_boxes, on=['season','game_id'], how='left', validate='one_to_one')
    s = s.merge(optional, on=['season','game_id'], how='left', validate='one_to_one')
    s.sort_values(['season','day','game_id'], kind='stable', inplace=True)
    results = []
    valid_counts = defaultdict(int)
    for season, game_set in s.groupby('season', sort=True):
        teams = defaultdict(Team)
        for day, one_day in game_set.groupby('day', sort=True):
            changes = []
            for game in one_day.itertuples(index=False):
                h, a = teams[str(game.home_team_id)], teams[str(game.away_team_id)]
                h_efg,h_opp_efg,h_2p,h_opp_2p,h_share = h.shooting()
                a_efg,a_opp_efg,a_2p,a_opp_2p,a_share = a.shooting()
                h_tov,h_tov_def,h_orb,h_orb_def,h_ft,h_ft_def = h.four()
                a_tov,a_tov_def,a_orb,a_orb_def,a_ft,a_ft_def = a.four()
                neutral = int(game.neutral_site) == 1
                elo_h = number(getattr(game,'elo_gap',np.nan))
                # Elo gap computed in V4 strictly at prior-day cutoff; here
                # only available if explicitly merged from V4 chronology.
                results.append({
                    'season':int(season),'game_id':str(game.game_id),
                    'v6_efg_gap':h_efg-a_efg,
                    'v6_efg_matchup_gap':(h_efg-a_opp_efg)-(a_efg-h_opp_efg),
                    'v6_two_pct_gap':h_2p-a_2p,
                    'v6_two_matchup_gap':(h_2p-a_opp_2p)-(a_2p-h_opp_2p),
                    'v6_two_attempt_share_gap':a_share-h_share if np.isfinite(h_share) and np.isfinite(a_share) else np.nan,
                    'v6_tov_gap':a_tov-h_tov, # fewer turnovers better
                    'v6_tov_matchup_gap':(a_tov+h_tov_def-h_tov-a_tov_def)/2.0,
                    'v6_orb_rate_gap':h_orb-a_orb,
                    'v6_orb_matchup_gap':(h_orb-a_orb_def)-(a_orb-h_orb_def),
                    'v6_ft_rate_gap':h_ft-a_ft,
                    'v6_ft_matchup_gap':(h_ft-a_ft_def)-(a_ft-h_ft_def),
                    'v6_elo_opponent_adjusted_eff_gap':(
                        float(np.mean(h.adjusted_eff))-float(np.mean(a.adjusted_eff))
                        if h.adjusted_eff and a.adjusted_eff else np.nan),
                    'v6_neutral_elo_gap':elo_h if neutral else 0.0,
                    'v6_early_elo_gap':elo_h if min(h.n_shots,a.n_shots)<10 else 0.0,
                    'v6_early_sample_uncertainty':1/np.sqrt(1+min(h.n_shots,a.n_shots)),
                    'v6_venue_adjusted_gap':(h.venue('neutral')-a.venue('neutral') if neutral else
                                             h.venue('home')-a.venue('away')),
                    'v6_shoot_sample_min':min(h.n_shots,a.n_shots),
                    'v6_tov_sample_min':min(h.n_tov,a.n_tov),
                    'v6_reb_sample_min':min(h.n_reb,a.n_reb),
                    'v6_ft_sample_min':min(h.n_ft,a.n_ft),
                })
                # No updating until after snapshots for ALL games today.
                changes.append(game)
            for g in changes:
                h, a = teams[str(g.home_team_id)], teams[str(g.away_team_id)]
                if not v5.valid_box(g):
                    valid_counts['invalid_base_box'] += 1
                    continue
                poss_h, poss_a = number(g.h_possessions), number(g.a_possessions)
                if any(not np.isfinite(x) for x in (poss_h,poss_a)):
                    continue
                # A game's box score counts only when BOTH teams have the data.
                h_fgm,h_fga,h_3m,h_3a = [number(getattr(g,'home_'+n))
                                          for n in ('fg_m','fg_a','three_m','three_a')]
                a_fgm,a_fga,a_3m,a_3a = [number(getattr(g,'away_'+n))
                                          for n in ('fg_m','fg_a','three_m','three_a')]
                if all(np.isfinite(x) for x in (h_fgm,h_fga,h_3m,h_3a,a_fgm,a_fga,a_3m,a_3a)) \
                   and all(0 <= x <= y for x,y in ((h_3m,h_3a),(a_3m,a_3a),(h_3a,h_fga),(a_3a,a_fga),
                                                   (h_fgm,h_fga),(a_fgm,a_fga))) \
                   and h_fgm >= h_3m and a_fgm >= a_3m and h_fga>0 and a_fga>0:
                    h.fgm+=h_fgm; h.fga+=h_fga; h.three_m+=h_3m; h.three_a+=h_3a
                    h.opp_fgm+=a_fgm; h.opp_fga+=a_fga; h.opp_three_m+=a_3m; h.opp_three_a+=a_3a
                    a.fgm+=a_fgm; a.fga+=a_fga; a.three_m+=a_3m; a.three_a+=a_3a
                    a.opp_fgm+=h_fgm; a.opp_fga+=h_fga; a.opp_three_m+=h_3m; a.opp_three_a+=h_3a
                    h.n_shots+=1; a.n_shots+=1; valid_counts['shooting']+=1
                h_tov,a_tov=number(g.home_turnovers),number(g.away_turnovers)
                if all(np.isfinite(x) and 0 <= x <= 50 for x in (h_tov,a_tov)):
                    h.tov+=h_tov; a.tov+=a_tov
                    h.opp_tov+=a_tov; a.opp_tov+=h_tov
                    h.poss+=poss_h; a.poss+=poss_a
                    h.opp_poss+=poss_a; a.opp_poss+=poss_h
                    h.n_tov+=1; a.n_tov+=1; valid_counts['turnovers']+=1
                h_orb,a_orb,h_drb,a_drb=[number(getattr(g,k)) for k in (
                    'home_off_rebounds','away_off_rebounds','home_def_rebounds','away_def_rebounds')]
                if all(np.isfinite(x) and 0 <= x <= 60 for x in (h_orb,a_orb,h_drb,a_drb)):
                    h.orebs+=h_orb; h.drebs+=h_drb; h.opp_orebs+=a_orb; h.opp_drebs+=a_drb
                    a.orebs+=a_orb; a.drebs+=a_drb; a.opp_orebs+=h_orb; a.opp_drebs+=h_drb
                    h.n_reb+=1; a.n_reb+=1; valid_counts['rebounds']+=1
                h_ft,a_ft=number(g.home_ft_a),number(g.away_ft_a)
                if all(np.isfinite(x) and 0 <= x <= 75 for x in (h_ft,a_ft)) and h_fga>0 and a_fga>0:
                    h.fta+=h_ft; h.ft_fga+=h_fga; h.opp_fta+=a_ft; h.opp_ft_fga+=a_fga
                    a.fta+=a_ft; a.ft_fga+=a_fga; a.opp_fta+=h_ft; a.opp_ft_fga+=h_fga
                    h.n_ft+=1; a.n_ft+=1; valid_counts['free_throws']+=1
                # historical opponent Elo at *that* game's prior-day rating
                # is available from v4 replay. Residual net efficiency is
                # added only AFTER today and never uses an end-season rating.
                if np.isfinite(number(getattr(g,'elo_gap',np.nan))):
                    pair_poss=(poss_h+poss_a)/2
                    net=100*(float(g.home_score)-float(g.away_score))/pair_poss
                    adjust=.06*float(g.elo_gap)+(0.0 if int(g.neutral_site)==1 else 3.0)  # fixed prior scale
                    h.adjusted_eff.append(net-adjust)
                    a.adjusted_eff.append(-net+adjust)
                    valid_counts['elo_opponent_adjusted_games']+=1
                if np.isfinite(number(getattr(g,'elo_gap',np.nan))):
                    venue_h = 'neutral' if int(g.neutral_site)==1 else 'home'
                    venue_a = 'neutral' if int(g.neutral_site)==1 else 'away'
                    h.venue_nets[venue_h].append(net-adjust)
                    a.venue_nets[venue_a].append(-net+adjust)
    frame=pd.DataFrame(results)
    return frame, dict(valid_counts)


def availability_features(scores: pd.DataFrame, reports: pd.DataFrame):
    """Fail closed on historical player data; unreported != available.

    Only usable if pre-tipoff, ID-linked records exist on BOTH sides for most
    training/validation games. No 'all healthy' assumption for zero reports.
    """
    empty = scores[['season','game_id']].copy()
    for key in PLAYER:
        empty[key] = np.nan
    audit = {'historical_pretipoff_reports_total':len(reports),
             'player_feature_enabled':False,'games_with_two_sided_reports':0,
             'player_availability_reason':'Historical pretipoff player data are absent or insufficient'}
    if reports.empty:
        return empty, audit
    r=reports.copy()
    r['game_id']=r.game_id.astype(str)
    r['team_id']=r.team_id.astype(str)
    r=r.dropna(subset=['game_id','team_id','player_id','status','reported_at','retrieved_at'])
    r['observed_at']=pd.to_datetime(r['retrieved_at'],utc=True,errors='coerce')
    r['reported_at']=pd.to_datetime(r['reported_at'],utc=True,errors='coerce')
    r['tipoff_utc']=pd.to_datetime(r['tipoff_utc'],utc=True,errors='coerce')
    r=r[(r.observed_at<r.tipoff_utc)&(r.reported_at<r.tipoff_utc)]
    r=r.sort_values(['game_id','team_id','player_id','observed_at'])
    r=r.drop_duplicates(['game_id','team_id','player_id'], keep='last')
    r['out']=r.status.str.lower().str.strip().isin(('out','inactive','ruled out')).astype(int)
    r['questionable']=r.status.str.lower().str.strip().isin(('questionable','doubtful','game-time decision')).astype(int)
    by_team=r.groupby(['game_id','team_id'],as_index=False)[['out','questionable']].sum()
    joined = scores[['season','game_id','home_team_id','away_team_id']].copy()
    for side in ('home','away'):
        lookup=by_team.rename(columns={'team_id':side+'_team_id','out':side+'_out',
                                        'questionable':side+'_questionable'})
        joined=joined.merge(lookup,on=['game_id',side+'_team_id'],how='left',validate='many_to_one')
    both=joined[['home_out','away_out']].notna().all(axis=1)
    for key in PLAYER:
        joined[key]=np.nan
    joined.loc[both,PLAYER[0]]=joined.loc[both,'away_out']-joined.loc[both,'home_out']
    joined.loc[both,PLAYER[1]]=(joined.loc[both,'away_questionable']-
                                joined.loc[both,'home_questionable'])
    # Require historical coverage in BOTH train and selection seasons.
    train_mask=joined.season.isin(v4.TRAIN)
    val_mask=joined.season==v4.VALIDATE
    train_coverage=float(both[train_mask].mean()) if train_mask.any() else 0.
    val_coverage=float(both[val_mask].mean()) if val_mask.any() else 0.
    audit.update({'games_with_two_sided_reports':int(both.sum()),
                  'two_sided_coverage_training_pct':round(train_coverage*100,2),
                  'two_sided_coverage_2025_pct':round(val_coverage*100,2)})
    if train_coverage >= .50 and val_coverage >= .50:
        audit['player_feature_enabled']=True
        audit['player_availability_reason']='Pregame reports have >=50% coverage in both training and 2025 validation.'
    else:
        joined[PLAYER]=np.nan
    return joined[['season','game_id']+PLAYER], audit


def make_frame(raw, scores, boxes, optional, reports, extra_audit):
    frame, audit = v5.make_frame(raw,scores,boxes)
    # Feed strict prior-day Elo from V4 to opponent/context replayer. These
    # values were reconstructed from previous calendar dates.
    lagged_elo=v4.chronological_features(scores)[['season','game_id','elo_gap']]
    score_input=scores.merge(lagged_elo,on=['season','game_id'],how='left',validate='one_to_one')
    additions, valid=replay(score_input,boxes,optional)
    players, player_audit=availability_features(scores,reports)
    frame=frame.merge(additions,on=['season','game_id'],how='left',validate='one_to_one')
    frame=frame.merge(players,on=['season','game_id'],how='left',validate='one_to_one')
    if frame[ALL].isna().all(axis=1).any():
        raise RuntimeError('Missing V6 replay records for eligible games')
    for key in ALL+PLAYER:
        frame[key]=pd.to_numeric(frame[key],errors='coerce')
    frame.replace([np.inf,-np.inf],np.nan,inplace=True)
    audit.update(extra_audit)
    audit.update(player_audit)
    audit['v6_valid_historical_box_games_by_category']=valid
    audit['v6_lagged_game_cutoff']='previous calendar date (not game day)'
    audit['v6_same_eligible_games_as_v5']=True
    audit['v6_pretrained_upstream_hybrid_still_not_proven_leakage_free']=True
    audit['v6_2026_previously_reviewed_not_holdout']=True
    for group,name in ((SHOOTING,'shooting'),(FOUR_FACTORS,'four_factors'),
                       (OPPONENT,'opponent'),(CONTEXT,'context')):
        audit['new_'+name+'_non_null_pct']=round(float(frame[group].notna().any(axis=1).mean())*100,2)
    frame.attrs['player_feature_enabled'] = bool(player_audit['player_feature_enabled'])
    return frame,audit


def day_bootstrap(diff, dates, seed=SEED, iters=1000):
    """Resample calendar DAYS, not games, to respect daily result clustering."""
    t=pd.DataFrame({'day':pd.to_datetime(dates,utc=True).dt.date,'diff':diff})
    grouped=t.groupby('day',sort=True).agg(total=('diff','sum'),n=('diff','size'))
    rng=np.random.default_rng(seed)
    inds=rng.integers(0,len(grouped),size=(iters,len(grouped)))
    sums=grouped['total'].to_numpy()[inds].sum(axis=1)
    counts=grouped['n'].to_numpy()[inds].sum(axis=1)
    bs=100*sums/counts
    return round(float(np.percentile(bs,2.5)),3),round(float(np.percentile(bs,97.5)),3)


def evaluate(frame):
    # Exact V5 reference: selected on 2025, recomputed on identical rows.
    old_metrics,old_leaderboard,old_paired,old_hist,old_selected,_,_=v5.predict_and_evaluate(frame)
    train=frame[frame.season.isin(v4.TRAIN)]
    val=frame[frame.season==v4.VALIDATE]
    rev=frame[frame.season==v4.REVIEW]
    ytr,yval,yrev=[x.home_win.astype(int).to_numpy() for x in (train,val,rev)]
    base_strict=list(v5.STRICT_FIELDS)
    base_hybrid=list(v5.HYBRID_FIELDS)
    player=PLAYER if frame.attrs.get('player_feature_enabled', False) else []
    new_all=ALL+player
    models={}
    def add(label,group,fields,pipe):
        fields=list(dict.fromkeys(fields))
        pipe.fit(train[fields],ytr)
        models[label]={'group':group,
                       'val':pipe.predict_proba(val[fields])[:,1],
                       'review':pipe.predict_proba(rev[fields])[:,1]}
    # Controlled ablations: fixed model and fixed original V5 strict features.
    for name,fields in (('shooting',SHOOTING),('four_factors',FOUR_FACTORS),
                        ('opponent',OPPONENT),('context',CONTEXT)):
        add('V6 ablation +'+name,'ablation',base_strict+fields,v4.logistic(3.0))
    if player:
        add('V6 ablation +player','ablation',base_strict+PLAYER,v4.logistic(3.0))
    for family,base in (('strict',base_strict),('hybrid',base_hybrid)):
        fields=base+new_all
        add('V6 '+family+' logistic C=0.3',family,fields,v4.logistic(.3))
        add('V6 '+family+' logistic C=3.0',family,fields,v4.logistic(3.0))
        add('V6 '+family+' boost A',family,fields,v4.boosted(3,220,100,.045))
        models['V6 '+family+' blend']={'group':family,
            'val':.60*models['V6 '+family+' boost A']['val']+.40*models['V6 '+family+' logistic C=3.0']['val'],
            'review':.60*models['V6 '+family+' boost A']['review']+.40*models['V6 '+family+' logistic C=3.0']['review']}
    selection=pd.DataFrame([{'family':p['group'],**v4.quality(yval,p['val'],name,v4.VALIDATE)}
                             for name,p in models.items()])
    selection=selection.sort_values(['family','winner_accuracy_pct','log_loss'],
                                    ascending=[True,False,True])
    chosen={group:str(selection[selection.family==group].iloc[0].model)
            for group in ('strict','hybrid')}
    comparison=[]
    # We preserve the REAL V5 selections; no choosing a new V5 after viewing V6.
    for year,y,key in ((2025,yval,'val'),(2026,yrev,'review')):
        comparison += old_metrics[(old_metrics.season==year)&(
            old_metrics.model.isin(['V5 strict '+old_selected['strict'].split('V5 strict ')[-1],
                                   'V5 hybrid '+old_selected['hybrid'].split('V5 hybrid ')[-1]]))].to_dict('records')
        for group in ('strict','hybrid'):
            choice=chosen[group]
            comparison.append(v4.quality(y,models[choice][key],choice,year))
    comparison=pd.DataFrame(comparison)
    rev_output=old_hist.copy()
    assert rev_output.game_id.astype(str).to_list()==rev.game_id.astype(str).to_list(), 'Games not aligned'
    for group in ('strict','hybrid'):
        rev_output['v6_'+group+'_home_win_prob']=models[chosen[group]]['review']
    rev_output['v6_game_date']=rev.game_date.to_numpy()
    for field in ('v6_shoot_sample_min','v6_tov_sample_min','v6_reb_sample_min','v6_ft_sample_min'):
        rev_output[field]=rev[field].to_numpy()
    paired=[]
    for group in ('strict','hybrid'):
        new=models[chosen[group]]['review']
        old=old_hist['v5_'+group+'_home_win_prob'].to_numpy()
        new_ok=((new>=.5)==yrev).astype(int)
        old_ok=((old>=.5)==yrev).astype(int)
        diff=new_ok-old_ok
        low,high=day_bootstrap(diff,rev.game_date)
        paired.append({'comparison':f'V6 {group} - V5 {group}',
                       'accuracy_difference_pp':round(float(diff.mean())*100,3),
                       'daily_bootstrap_95pct_low_pp':low,
                       'daily_bootstrap_95pct_high_pp':high,
                       'v6_correct_v5_wrong':int(((new_ok==1)&(old_ok==0)).sum()),
                       'v5_correct_v6_wrong':int(((new_ok==0)&(old_ok==1)).sum())})
    ablations=[]
    for name,p in models.items():
        if p['group']!='ablation':continue
        for year,y,key in ((2025,yval,'val'),(2026,yrev,'review')):
            ablations.append(v4.quality(y,p[key],name,year))
    breakdown=[]
    for group in ('strict','hybrid'):
        for label,mask in (
           ('nonneutral',rev.neutral_site==0),('neutral',rev.neutral_site==1),
           ('early <10 box games',rev.v6_shoot_sample_min<10),
           ('later >=10 box games',rev.v6_shoot_sample_min>=10)):
            y=yrev[mask]
            for version,p in (('V5',old_hist['v5_'+group+'_home_win_prob'].to_numpy()),
                              ('V6',models[chosen[group]]['review'])):
                if mask.sum():
                    breakdown.append({'category':label,'group':group,'model':version,
                        **v4.quality(y,p[mask],version,2026)})
    # Model calibration on the reviewed season; descriptive only.
    calibration=[]
    for group in ('strict','hybrid'):
        probs=models[chosen[group]]['review']
        for lo,hi in ((0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.000001)):
            m=(probs>=lo)&(probs<hi)
            calibration.append({'group':group,'prob_bin':f'{lo:.0%}-{min(hi,1):.0%}',
                'games':int(m.sum()),
                'mean_predicted_home_win_pct':round(float(probs[m].mean())*100,2) if m.any() else None,
                'observed_home_win_pct':round(float(yrev[m].mean())*100,2) if m.any() else None})
    return (comparison,selection,pd.DataFrame(paired),pd.DataFrame(ablations),
            pd.DataFrame(breakdown),pd.DataFrame(calibration),rev_output,chosen,old_selected)


def coverage_table(raw,totals,scores,frame):
    # Report what we know, not invented reasons for why V3 rejected a game.
    rows=[]
    for y in sorted(totals.season.astype(int).unique()):
        completed=int(totals.loc[totals.season.astype(int)==y,'completed_d1_games'].sum())
        eligible_v3=int((raw.season==y).sum())
        eligible_v6=int((frame.season==y).sum())
        rows.append({'season':y,'completed_d1_games':completed,
            'eligible_v3_base':eligible_v3,'excluded_in_v3_quality_filters':completed-eligible_v3,
            'eligible_v6_same_as_v5':eligible_v6,
            'excluded_by_v4_prior_day_replay':eligible_v3-eligible_v6,
            'v6_coverage_pct':round(100*eligible_v6/max(completed,1),2)})
    return pd.DataFrame(rows)


def report_html(comparison,selection,paired,ablation,breakdown,calibration,coverage,audit,selected):
    def table(df):
        return df.to_html(index=False,border=0,escape=True,classes='result')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NCAA MBB V6 Research</title>
<style>body{{font-family:system-ui,Arial;max-width:1200px;margin:32px auto;padding:0 22px;color:#162532;line-height:1.5}}
h1,h2{{line-height:1.2}}table.result{{border-collapse:collapse;display:block;overflow:auto;margin:10px 0 26px}}
th,td{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}}th{{background:#eff3f8}}
.note{{background:#edf3f7;padding:14px;border-left:4px solid #56718f}}pre{{white-space:pre-wrap;background:#f2f5f7;padding:12px}}</style></head><body>
<h1>NCAA MBB — V6 Historical Research</h1><p>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
<div class="note"><b>Historical research, not a prospective accuracy claim.</b> Models train on 2022–24,
select on 2025, and compare on reviewed 2026 (NOT an untouched holdout). Additional box-score inputs
are frozen at least one calendar day before each game. V5/V6 <b>hybrid</b> reuse earlier upstream
features not independently audited for time leakage. Missing historical player reports are NOT
assumed to mean everyone was available. No future game selections.</div>
<h2>Model choices: selected only on 2025</h2><p>V6 strict: {html.escape(selected['strict'])};
V6 hybrid: {html.escape(selected['hybrid'])}.</p>
<h2>V5 vs V6 — identical eligible games</h2>{table(comparison)}
<h2>2026 paired V6 minus V5 differences</h2>{table(paired)}
<h2>New feature groups tested independently (exploratory)</h2>{table(ablation)}
<h2>Neutral-site and early-season breakdown</h2>{table(breakdown)}
<h2>2026 probability calibration</h2>{table(calibration)}
<h2>Model selection on 2025</h2>{table(selection)}
<h2>Coverage and excluded game categories</h2>{table(coverage)}
<h2>Data freshness and player availability audit</h2><pre>{html.escape(json.dumps(audit,indent=2,default=str))}</pre>
<h2>Important limitations</h2><p>Shot, rebound and turnover history can only be evaluated retrospectively
when box scores exist. Historical source correctness requires separate audit; same-day cutoff alone
cannot prove imported timestamps. If early-year player reports lack stable game/team/player IDs,
features are disabled. A future pregame log with frozen V6 probabilities is required to establish live accuracy.</p>
</body></html>'''


def run(raw,totals,scores,base_boxes,optional,reports,extra_audit):
    frame,audit=make_frame(raw,scores,base_boxes,optional,reports,extra_audit)
    comparison,selection,paired,ablation,breakdown,calibration,historic,selected,old_selected=evaluate(frame)
    coverage=coverage_table(raw,totals,scores,frame)
    audit.update({'selected_v6_on_2025':selected,'selected_v5_on_2025':old_selected,
                  'new_pregame_players_effectively_in_model':bool(audit['player_feature_enabled'])})
    OUT.mkdir(exist_ok=True,parents=True)
    export={ 'model_comparison.csv':comparison,'selection_2025.csv':selection,
             'paired_v6_vs_v5_2026.csv':paired,'feature_ablation.csv':ablation,
             'neutral_early_diagnostics.csv':breakdown,'calibration_2026.csv':calibration,
             'coverage.csv':coverage,'historical_2026_predictions.csv':historic}
    for filename,df in export.items():
        df.to_csv(OUT/filename,index=False)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,default=str),encoding='utf-8')
    (OUT/'report.html').write_text(report_html(comparison,selection,paired,ablation,breakdown,
                                              calibration,coverage,audit,selected),encoding='utf-8')
    summary={'selected_v6':selected,'selected_v5':old_selected,
             'model_comparison':comparison.to_dict('records'),
             'paired_v6_vs_v5_2026':paired.to_dict('records'),
             'player_features_enabled':audit['player_feature_enabled'],
             'historical_only':True,'2026_untouched_holdout':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    print('\nV6 HISTORICAL RESEARCH — reviewed 2026, not untouched holdout')
    print(comparison[comparison.season==2026].to_string(index=False))
    print('\nV6 vs V5:\n'+paired.to_string(index=False))
    print('Report: '+str((OUT/'report.html').resolve()))
    return summary


def self_test():
    """Full end-to-end synthetic test, including future-result non-leakage."""
    rng=np.random.default_rng(861)
    raw,scores,boxes,optionals=[],[],[],[]
    for season in range(2022,2027):
        strengths=rng.normal(0,9,28)
        for d in range(1,118):
            day=date(season-1,11,1)+pd.Timedelta(days=d)
            for i in range(5):
                h,a=((d*3+2*i)%28),((d*3+2*i+7)%28)
                gap=strengths[h]-strengths[a]
                home_score=int(round(73+(gap+rng.normal(0,11))/2))
                away_score=int(round(73-(gap+rng.normal(0,11))/2))
                if home_score==away_score:home_score+=1
                gid=f'{season}-{d}-{i}'
                neutral=int(d%5==0)
                scores.append(dict(season=season,game_id=gid,game_date=str(day),
                    home_team_id=str(h),away_team_id=str(a),neutral_site=neutral,
                    home_score=home_score,away_score=away_score))
                raw.append(dict(season=season,game_id=gid,game_date=str(day),
                    neutral_site=neutral,home_score=home_score,away_score=away_score,
                    h_games_before=12,a_games_before=12,
                    h_rating_date=str(day-pd.Timedelta(days=1)),a_rating_date=str(day-pd.Timedelta(days=1)),
                    h_ppg=73+strengths[h]/5,a_ppg=73+strengths[a]/5,
                    h_allowed=72-strengths[h]/5,a_allowed=72-strengths[a]/5,
                    h_raw_off=105+strengths[h]/3,a_raw_off=105+strengths[a]/3,
                    h_raw_def=100-strengths[h]/3,a_raw_def=100-strengths[a]/3,
                    h_raw_net=strengths[h]/2,a_raw_net=strengths[a]/2,
                    h_pace=70,a_pace=70,h_recent_margin=strengths[h]/2,
                    a_recent_margin=strengths[a]/2,h_rest=2,a_rest=2,
                    h_adj_net=strengths[h],a_adj_net=strengths[a],h_sos=0,a_sos=0))
                ha,aa=22,23
                hm=int(rng.binomial(ha,.34));am=int(rng.binomial(aa,.34))
                hm_fg=hm+int(rng.binomial(60-ha,.49))
                am_fg=am+int(rng.binomial(60-aa,.49))
                boxes.append(dict(season=season,game_id=gid,
                    h_box_points=home_score,a_box_points=away_score,h_possessions=70,a_possessions=70,
                    h_three_m=hm,h_three_a=ha,h_fg_a=60,a_three_m=am,a_three_a=aa,a_fg_a=60))
                optionals.append(dict(season=season,game_id=gid,
                    home_fg_m=hm_fg,home_fg_a=60,home_three_m=hm,home_three_a=ha,
                    away_fg_m=am_fg,away_fg_a=60,away_three_m=am,away_three_a=aa,
                    home_turnovers=12,away_turnovers=13,
                    home_off_rebounds=10,away_off_rebounds=11,
                    home_def_rebounds=22,away_def_rebounds=21,
                    home_ft_a=19,away_ft_a=18,home_ft_m=14,away_ft_m=13))
    raw,scores,boxes,optionals=map(pd.DataFrame,(raw,scores,boxes,optionals))
    lagged=v4.chronological_features(scores)[['season','game_id','elo_gap']]
    with_elo=scores.merge(lagged,on=['season','game_id'],validate='one_to_one')
    original,_=replay(with_elo,boxes,optionals)
    day0=scores.loc[0,'game_date']
    day_ids=set(scores.loc[scores.game_date==day0,'game_id'])
    modified=scores.copy()
    modified.loc[modified.game_id.isin(day_ids),'home_score']+=25
    other_elo=v4.chronological_features(modified)[['season','game_id','elo_gap']]
    other,_=replay(modified.merge(other_elo,on=['season','game_id']),boxes,optionals)
    merge=original.merge(other,on=['season','game_id'],suffixes=('_before','_after'))
    same=merge[merge.game_id.isin(day_ids)]
    for field in ALL:
        assert np.allclose(same[field+'_before'],same[field+'_after'],equal_nan=True),field
    optionals2=optionals.copy()
    optionals2.loc[optionals2.game_id.isin(day_ids),'home_turnovers']=0
    optionals2.loc[optionals2.game_id.isin(day_ids),'home_fg_m']=10
    other2,_=replay(with_elo,boxes,optionals2)
    merge=original.merge(other2,on=['season','game_id'],suffixes=('_before','_after'))
    same=merge[merge.game_id.isin(day_ids)]
    for field in ALL:
        assert np.allclose(same[field+'_before'],same[field+'_after'],equal_nan=True),field
    totals=pd.DataFrame([{'season':y,'completed_d1_games':585} for y in range(2022,2027)])
    extra={'synthetic_input':True,'team_game_stats_mapped_columns':{key:key for key in ALIASES}}
    run(raw,totals,scores,boxes,optionals,pd.DataFrame(),extra)
    assert (OUT/'report.html').exists()
    print('V6 SYNTHETIC FULL TEST PASSED; ACCURACY IS FAKE AND MUST NOT BE REPORTED AS REAL.')


if __name__=='__main__':
    try:
        if '--self-test' in sys.argv:
            self_test()
        else:
            raw,totals,scores,boxes=v5.fetch()
            optional,reports,audit=optional_data()
            run(raw,totals,scores,boxes,optional,reports,audit)
    except Exception as exc:
        print(f'V6 failed: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1) from exc
