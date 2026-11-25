"""NCAA MBB V9 — six-part, READ-ONLY historical research experiment.

1. Prior-game, minutes-weighted player rotation, never today's appearances.
2. Shrunk opponent-adjusted efficiency / strength with measured sample sizes.
3. Separate expected-possession and scoring-efficiency model for total points.
4. Offense-versus-defense matchup interactions.
5. Prior-game recent form and roster continuity (not assumed injury status).
6. Chronological 2022-24 training, 2025 selection, 2026 same-game retrospective
   comparison against V5/V8, with bootstrap uncertainty and pregame audit.

NO upcoming-game picks or wagering actions. The 2026 season has repeatedly
been inspected and is NOT an untouched holdout. Upstream V5 hybrid feature
provenance is not fully verified. The read-only script does not update the site.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import v4_research as v4
import v5_research as v5
import v6_research as v6
import v8_research as v8

OUT = Path(os.getenv('V9_OUTPUT_DIR', 'v9_results'))
SEED = 91
PLAYER_PRIOR_MINUTES = 100.0
PLAYER_PRIOR_PTS_PER_MIN = .30
MIN_PLAYER_COVERAGE = .60  # two-team, usable *historical* boxes in train + 2025
PREGAME_ONLY = True

PLAYER_FEATURES = [
    'v9_rotation_scoring_gap', 'v9_rotation_top3_gap',
    'v9_rotation_continuity_gap', 'v9_rotation_sample_min',
]
STRENGTH_FEATURES = [
    'v9_reliable_adj_net_gap', 'v9_elo_sos_interaction',
    'v9_strength_uncertainty',
]
PACE_FEATURES = [
    'v9_expected_possessions', 'v9_scoring_baseline_total',
    'v9_pace_sample_min', 'v9_pace_gap',
]
MATCHUP_FEATURES = [
    'v9_style_efg_clash', 'v9_style_three_clash',
    'v9_pressure_efficiency_clash',
]
FORM_FEATURES = [
    'v9_form_weighted_gap', 'v9_form_vs_season_gap',
    'v9_recent_sample_min',
]
ALL_FEATURES = STRENGTH_FEATURES + PACE_FEATURES + MATCHUP_FEATURES + FORM_FEATURES


def num(x: Any) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except (ValueError, TypeError):
        return np.nan


def minutes(x: Any) -> float:
    if x is None or pd.isna(x):
        return np.nan
    s = str(x).strip()
    try:
        if ':' in s:
            m, sec = s.split(':', 1)
            z = float(m) + float(sec)/60.0
        else:
            z = float(s)
        return z if 0 < z <= 60 else np.nan
    except (ValueError, TypeError):
        return np.nan


def fetch_players() -> tuple[pd.DataFrame, dict]:
    """Discover actual player-game column names; missing optional fields remain NULL.

    Each row is from a completed *previous* game; final-game player data are
    never read into same-day features. Database access is transaction read-only.
    """
    url = os.getenv('DATABASE_URL')
    if not url:
        raise RuntimeError('DATABASE_URL GitHub secret is missing')
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row

    aliases = {
        'minutes': ('minutes', 'minutes_played', 'min'),
        'points': ('points', 'pts'),
        'rebounds': ('rebounds', 'reb'),
        'assists': ('assists', 'ast'),
        'turnovers': ('turnovers', 'tov'),
    }
    with psycopg.connect(url, sslmode='require', connect_timeout=30,
                         prepare_threshold=None, row_factory=dict_row) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("""SELECT column_name FROM information_schema.columns
                         WHERE table_schema='public' AND table_name='player_game_stats'""")
            cols = {r['column_name'] for r in cur.fetchall()}
            required = {'game_id', 'team_id', 'player_id'}
            if not required <= cols:
                return pd.DataFrame(), {'usable': False,
                    'reason': 'player_game_stats missing game_id, team_id or player_id'}
            mapped = {k: next((c for c in variants if c in cols), None)
                      for k, variants in aliases.items()}
            if mapped['minutes'] is None or mapped['points'] is None:
                return pd.DataFrame(), {'usable': False,
                    'reason': 'player minutes or points missing from historical schema',
                    'column_map': mapped}
            names = [sql.SQL('g.season AS season'),
                     sql.SQL('g.game_id::text AS game_id'),
                     sql.SQL('p.team_id::text AS team_id'),
                     sql.SQL('p.player_id::text AS player_id')]
            for label, col in mapped.items():
                expr = (sql.SQL('NULL::text') if col is None else
                        sql.SQL('p.{}::text').format(sql.Identifier(col)))
                names.append(sql.SQL('{} AS {}').format(expr, sql.Identifier(label)))
            q = sql.SQL('''SELECT {} FROM public.player_game_stats p
               JOIN public.games g ON g.game_id=p.game_id
               JOIN public.teams t ON t.team_id=p.team_id AND t.is_d1=1
               WHERE g.season BETWEEN 2022 AND 2026 AND g.completed=1
                 AND g.game_date IS NOT NULL
               ORDER BY g.season,g.game_date,g.game_id,p.team_id,p.player_id''').format(
                   sql.SQL(', ').join(names))
            cur.execute(q)
            rows = pd.DataFrame(cur.fetchall())
    return rows, {'usable': not rows.empty, 'column_map': mapped,
                  'retrieved_player_rows': len(rows),
                  'note': 'Only prior-day player boxes; not a source of game-day availability.'}


@dataclass
class PlayerPast:
    played_minutes: float = 0.
    points: float = 0.


@dataclass
class Rotation:
    players: dict[str, PlayerPast] = field(default_factory=dict)
    game_rosters: deque = field(default_factory=lambda: deque(maxlen=5))
    valid_games: int = 0

    def features(self) -> tuple[float, float, float, int]:
        if not self.game_rosters:
            return np.nan, np.nan, np.nan, 0
        # Expected playing time = the observed minutes in the *last five*
        # previous team games, averaged across the whole window. Missing
        # appearances do not prove injuries or future nonparticipation.
        five = list(self.game_rosters)
        exp_minutes = defaultdict(float)
        for roster in five:
            for pid, observed in roster.items():
                exp_minutes[pid] += observed/len(five)
        total = sum(exp_minutes.values())
        if total <= 0:
            return np.nan, np.nan, np.nan, 0
        production = 0.
        for pid, m in exp_minutes.items():
            player = self.players[pid]
            ppm = ((player.points + PLAYER_PRIOR_MINUTES*PLAYER_PRIOR_PTS_PER_MIN) /
                   (player.played_minutes + PLAYER_PRIOR_MINUTES))
            production += m*ppm
        per40 = 40.*production/total
        top3 = sum(sorted(exp_minutes.values(), reverse=True)[:3])/total
        continuity = np.nan
        if len(five) >= 2:
            previous, recent = five[-2], five[-1]
            recent_total = sum(recent.values())
            if recent_total > 0:
                continuity = sum(min(m, previous.get(p,0.)) for p,m in recent.items())/recent_total
        return per40, top3, continuity, self.valid_games

    def add_game(self, roster: dict[str, tuple[float,float]]):
        clean = {}
        for pid, (mins, pts) in roster.items():
            past = self.players.setdefault(pid, PlayerPast())
            past.played_minutes += mins
            past.points += pts
            clean[pid] = mins
        self.game_rosters.append(clean)
        self.valid_games += 1


def player_replay(scores: pd.DataFrame, player_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Player data are committed after all games of a date have been snapshotted."""
    output = scores[['season','game_id']].copy()
    for f in PLAYER_FEATURES:
        output[f] = np.nan
    output['v9_player_both_prior_games'] = 0
    if player_rows.empty:
        return output, {'player_source_rows': 0, 'valid_player_team_boxes': 0,
                        'player_availability_verified': False,
                        'player_rotation_usable': False}
    p = player_rows.copy()
    p['game_id'] = p['game_id'].astype(str)
    p['team_id'] = p['team_id'].astype(str)
    p['player_id'] = p['player_id'].astype(str)
    p['minutes_num'] = p['minutes'].map(minutes)
    p['points_num'] = pd.to_numeric(p['points'],errors='coerce')
    p = p[p.minutes_num.notna() & p.points_num.notna() &
          p.points_num.between(0,100)].copy()
    # Nonidentical duplicate rows must be examined, not silently double-counted.
    p = p.drop_duplicates(['game_id','team_id','player_id','minutes_num','points_num'])
    if p.duplicated(['game_id','team_id','player_id']).any():
        raise RuntimeError('Conflicting duplicate player-game records; review historical source')
    by_pair = {}
    for (gid,tid), rows in p.groupby(['game_id','team_id'],sort=False):
        mins = rows.minutes_num.sum()
        if len(rows) < 5 or not (150 <= mins <= 270):
            continue
        by_pair[(str(gid),str(tid))] = {
            str(r.player_id): (float(r.minutes_num),float(r.points_num))
            for r in rows.itertuples(index=False)}
    s = scores.copy()
    s['game_id'] = s['game_id'].astype(str)
    s['day'] = pd.to_datetime(s.game_date, utc=True, errors='coerce').dt.date
    if s.duplicated(['season','game_id']).any():
        raise RuntimeError('Duplicate historical score keys')
    results=[]; complete_count=0
    for season, sg in s.sort_values(['season','day','game_id']).groupby('season'):
        teams = defaultdict(Rotation)
        for day, games in sg.groupby('day',sort=True):
            updates=[]
            for g in games.itertuples(index=False):
                h, a = str(g.home_team_id),str(g.away_team_id)
                h_rot, a_rot = teams[h],teams[a]
                h_f,a_f=h_rot.features(),a_rot.features()
                has_both = (str(g.game_id),h) in by_pair and (str(g.game_id),a) in by_pair
                if has_both:
                    # Guard against mismatched player totals and partial boxes.
                    ph = sum(v[1] for v in by_pair[(str(g.game_id),h)].values())
                    pa = sum(v[1] for v in by_pair[(str(g.game_id),a)].values())
                    has_both = (abs(ph-float(g.home_score)) <= 3 and
                                abs(pa-float(g.away_score)) <= 3)
                if has_both:
                    complete_count+=1
                    updates += [(h, by_pair[(str(g.game_id),h)]),
                                (a, by_pair[(str(g.game_id),a)])]
                results.append({'season':int(season),'game_id':str(g.game_id),
                    'v9_rotation_scoring_gap':h_f[0]-a_f[0],
                    'v9_rotation_top3_gap':h_f[1]-a_f[1],
                    'v9_rotation_continuity_gap':h_f[2]-a_f[2],
                    'v9_rotation_sample_min':min(h_f[3],a_f[3]),
                    'v9_player_both_prior_games': int(h_f[3]>=3 and a_f[3]>=3),
                    })
            for team, roster in updates:
                teams[team].add_game(roster)
    addition = pd.DataFrame(results)
    output = output[['season','game_id']].merge(addition,on=['season','game_id'],
        how='left',validate='one_to_one')
    return output, {'player_source_rows':len(player_rows),
                    'valid_player_team_boxes':2*complete_count,
                    'complete_player_game_pairs':complete_count,
                    'player_availability_verified':False,
                    'player_rotation_usable':True,
                    'player_current_game_never_used_as_feature':True,
                    'player_replay_window':'previous calendar dates, last five team games'}


@dataclass
class ScorePast:
    possessions: list[float] = field(default_factory=list)
    scored_eff: list[float] = field(default_factory=list)
    allowed_eff: list[float] = field(default_factory=list)
    margins: list[float] = field(default_factory=list)

    def features(self):
        n=len(self.possessions)
        # Conservative fixed prior: 69 possessions and 100 points / 100 poss.
        pace=(sum(self.possessions)+8*69.)/(n+8.)
        off=(sum(self.scored_eff)+8*100.)/(n+8.)
        defence=(sum(self.allowed_eff)+8*100.)/(n+8.)
        recent=self.margins[-5:]
        if not recent:
            form=0.
        else:
            weights=np.exp(np.arange(len(recent))*.35)
            form=float(np.average(recent,weights=weights))
        season_margin=float(np.mean(self.margins)) if self.margins else 0.
        return pace,off,defence,form,season_margin,n


def scoring_replay(scores: pd.DataFrame, boxes: pd.DataFrame):
    """Use prior-game points and possessions; current outcomes update next day."""
    b = boxes.copy()
    b['game_id'] = b.game_id.astype(str)
    s = scores.copy()
    s['game_id'] = s.game_id.astype(str)
    if b.duplicated(['season','game_id']).any():
        raise RuntimeError('Duplicate paired team boxes')
    s = s.merge(b,on=['season','game_id'],how='left',validate='one_to_one')
    s['day']=pd.to_datetime(s.game_date,utc=True,errors='coerce').dt.date
    rows=[]; qualified=0
    for season, seasons in s.sort_values(['season','day','game_id']).groupby('season'):
        teams=defaultdict(ScorePast)
        for day, gs in seasons.groupby('day',sort=True):
            updates=[]
            for g in gs.itertuples(index=False):
                h,a=teams[str(g.home_team_id)],teams[str(g.away_team_id)]
                hp,ho,hd,hr,hs,hn=h.features()
                ap,ao,ad,ar,ass,an=a.features()
                exp_poss=(hp+ap)/2
                home_per100=(ho+ad)/2
                away_per100=(ao+hd)/2
                total=exp_poss*(home_per100+away_per100)/100
                rows.append({'season':int(season),'game_id':str(g.game_id),
                    'v9_expected_possessions':exp_poss,
                    'v9_scoring_baseline_total':total,
                    'v9_pace_sample_min':min(hn,an),
                    'v9_pace_gap':hp-ap,
                    'v9_form_weighted_gap':hr-ar,
                    'v9_form_vs_season_gap':(hr-hs)-(ar-ass),
                    'v9_recent_sample_min':min(hn,an),
                    'v9_projected_home_points_base':exp_poss*home_per100/100,
                    'v9_projected_away_points_base':exp_poss*away_per100/100})
                if v5.valid_box(g):
                    poss=(float(g.h_possessions)+float(g.a_possessions))/2
                    if 35 <= poss <= 110:
                        qualified+=1
                        hpoints,apoints=float(g.home_score),float(g.away_score)
                        updates.extend([
                            (str(g.home_team_id), poss,hpoints,apoints),
                            (str(g.away_team_id), poss,apoints,hpoints)])
            for tid, poss, points_for, points_against in updates:
                st=teams[tid]
                st.possessions.append(poss)
                st.scored_eff.append(100*points_for/poss)
                st.allowed_eff.append(100*points_against/poss)
                st.margins.append(points_for-points_against)
    return pd.DataFrame(rows), {'scoring_valid_historical_box_games':qualified,
        'scoring_replayer_same_day_leakage':False}


def assemble_features(frame: pd.DataFrame, scores: pd.DataFrame, boxes: pd.DataFrame,
                      players: pd.DataFrame, audit: dict):
    pframe, player_audit=player_replay(scores,players)
    score,score_audit=scoring_replay(scores,boxes)
    data=v8.add_matchup_features(frame)
    data=data.merge(pframe,on=['season','game_id'],how='left',validate='one_to_one')
    data=data.merge(score,on=['season','game_id'],how='left',validate='one_to_one')
    for col in PLAYER_FEATURES+ALL_FEATURES:
        if col not in data:
            data[col]=np.nan
    # Fixed, disclosed interactions; no labels or same-day scores.
    reliable=np.minimum(pd.to_numeric(data['v5_min_box_games'],errors='coerce').fillna(0),40)
    weight=reliable/(reliable+12)
    data['v9_reliable_adj_net_gap']=pd.to_numeric(data.v5_adjusted_net_gap,errors='coerce')*weight
    data['v9_elo_sos_interaction']=(pd.to_numeric(data.elo_gap,errors='coerce')/250)*(
        pd.to_numeric(data.v5_lagged_sos_gap,errors='coerce')/20)
    data['v9_strength_uncertainty']=1/np.sqrt(reliable+1)
    data['v9_style_efg_clash']=pd.to_numeric(data.v6_efg_matchup_gap,errors='coerce')*(
        1+pd.to_numeric(data.v6_two_attempt_share_gap,errors='coerce').abs())
    data['v9_style_three_clash']=pd.to_numeric(data.v5_three_matchup_gap,errors='coerce')*(
        1+pd.to_numeric(data.v5_three_attempt_share_gap,errors='coerce').abs())
    data['v9_pressure_efficiency_clash']=pd.to_numeric(data.v6_efg_matchup_gap,errors='coerce')*(
        1+pd.to_numeric(data.v6_tov_matchup_gap,errors='coerce').abs())
    data.replace([np.inf,-np.inf],np.nan,inplace=True)
    # Coverage gate BEFORE fitting any model, evaluated only on train + 2025.
    has_prior=(data['v9_rotation_sample_min'].fillna(0)>=3)
    train = data.season.isin(v4.TRAIN)
    val = data.season==v4.VALIDATE
    train_cov=float(has_prior[train].mean()) if train.any() else 0.
    val_cov=float(has_prior[val].mean()) if val.any() else 0.
    enabled=train_cov>=MIN_PLAYER_COVERAGE and val_cov>=MIN_PLAYER_COVERAGE
    audit.update(player_audit)
    audit.update(score_audit)
    audit.update({'v9_player_prior_game_coverage_training_pct':round(train_cov*100,2),
                  'v9_player_prior_game_coverage_validation_pct':round(val_cov*100,2),
                  'v9_player_features_enabled':bool(enabled),
                  'v9_injury_reports_used':False,
                  'v9_player_coverage_gate':f'>={MIN_PLAYER_COVERAGE*100:g}% both-team prior rotations in train and validation',
                  'v9_full_score_and_roster_replay_previous_dates_only':True,
                  'v9_hybrid_upstream_feature_provenance_verified':False,
                  'v9_2026_not_fresh_holdout':True,
                  'v9_all_2027_prospective_claims':False})
    return data, audit, enabled


def rating_metrics(y, p, label, season, population):
    yy=np.asarray(y,dtype=int)
    pp=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    if not len(yy):
        return {'season':season,'population':population,'model':label,'games':0,
                'correct':0,'winner_accuracy_pct':None,'brier':None,'log_loss':None}
    return {'season':season,'population':population,'model':label,'games':len(yy),
        'correct':int(((pp>=.5)==yy).sum()),
        'winner_accuracy_pct':round(100*accuracy_score(yy,pp>=.5),2),
        'brier':round(float(brier_score_loss(yy,pp)),5),
        'log_loss':round(float(log_loss(yy,pp,labels=[0,1])),5)}


def ridge(alpha):
    return Pipeline([('fill',SimpleImputer(strategy='median',keep_empty_features=True)),
      ('scale',StandardScaler()),('model',Ridge(alpha=alpha))])


def evaluate(data: pd.DataFrame, audit: dict, player_enabled: bool):
    if data.duplicated(['season','game_id']).any():
        raise RuntimeError('Duplicate eligible historical games')
    # Get V5 and V8 *retrained on the exact same games*, not copied percentages.
    v8comp,_,_,_,older,older_selection,audit=v8.evaluate(data,audit)
    tr=data[data.season.isin(v4.TRAIN)]
    va=data[data.season==v4.VALIDATE]
    re=data[data.season==v4.REVIEW]
    y_tr=tr.home_win.astype(int).to_numpy()
    y_va=va.home_win.astype(int).to_numpy()
    y_re=re.home_win.astype(int).to_numpy()
    if len(tr)<500 or len(va)<200 or len(re)<200:
        raise RuntimeError('Insufficient historical games in 2022-24 / 2025 / 2026')
    if not np.array_equal(older.game_id.astype(str).to_numpy(),re.game_id.astype(str).to_numpy()):
        raise RuntimeError('V5/V8 game order does not match V9 review data')
    base = {'V5 strict':older.v5_strict_home_win_prob.to_numpy(),
            'V5 hybrid':older.v5_hybrid_home_win_prob.to_numpy(),
            'V8 strict':older.v8_strict_home_win_prob.to_numpy(),
            'V8 hybrid':older.v8_hybrid_home_win_prob.to_numpy()}
    chosen={}; selections=[]; comparisons=[]; ablations=[]
    output=re[['season','game_id','game_date','home_win','neutral_site','home_score','away_score']].copy().reset_index(drop=True)
    for label,probs in base.items():
        output[label.lower().replace(' ','_')+'_prob']=probs
    close = v8.close_mask(re)
    for family in ('strict','hybrid'):
        baseline=list(v5.STRICT_FIELDS if family=='strict' else v5.HYBRID_FIELDS)
        groups={
          'opponent_and_pace':STRENGTH_FEATURES+PACE_FEATURES,
          'matchup_and_form':MATCHUP_FEATURES+FORM_FEATURES,
          'full_nonplayer':ALL_FEATURES+v8.MATCHUP_FIELDS,
        }
        if player_enabled:
            groups['full_with_rotation']=ALL_FEATURES+v8.MATCHUP_FIELDS+PLAYER_FEATURES
        candidate=[]
        for tag,extra in groups.items():
            fields=list(dict.fromkeys(baseline+extra))
            for kind,model in [('logistic_C3',v4.logistic(3.0)),
                               ('boost',v4.boosted(3,180,100,.045))]:
                label=f'V9 {family} {tag} {kind}'
                model.fit(tr[fields],y_tr)
                v=model.predict_proba(va[fields])[:,1]
                r=model.predict_proba(re[fields])[:,1]
                m=rating_metrics(y_va,v,label,2025,'all')
                selections.append({'family':family,'features':tag,**m})
                candidate.append((label,fields,v,r,m))
                if kind=='logistic_C3':
                    ablations += [rating_metrics(y_va,v,label,2025,'all'),
                                  rating_metrics(y_re,r,label,2026,'all')]
        # 2025-only selection: accuracy first, then probability quality.
        selected=min(candidate,key=lambda item:(-item[4]['winner_accuracy_pct'],
                     item[4]['brier'],item[4]['log_loss'],item[0]))
        name,fields,p_val,p_rev,stat=selected
        chosen[family]={'name':name,'columns':fields,'selection_year':2025,
                        'review_year_not_used_for_selection':2026}
        output[f'v9_{family}_home_win_prob']=p_rev
        for population,mask in [('all',np.ones(len(re),bool)),('elo_close',close),('other',~close)]:
            for lbl,probs in [(f'V5 {family}',base[f'V5 {family}']),
                              (f'V8 {family}',base[f'V8 {family}']),
                              (f'V9 {family}',p_rev)]:
                comparisons.append(rating_metrics(y_re[mask],probs[mask],lbl,2026,population))
    # Independent score-total model: estimate possession count and offensive /
    # defensive efficiency separately, then calibrate total using ONLY past
    # season scores. No actual 2026 totals enter training or configuration.
    score_fields=['v9_expected_possessions','v9_scoring_baseline_total',
                  'v9_pace_gap','v9_form_weighted_gap','v9_strength_uncertainty',
                  'v9_rotation_scoring_gap'] if player_enabled else [
                  'v9_expected_possessions','v9_scoring_baseline_total',
                  'v9_pace_gap','v9_form_weighted_gap','v9_strength_uncertainty']
    y_train_total=tr.home_score.to_numpy(dtype=float)+tr.away_score.to_numpy(dtype=float)
    y_val_total=va.home_score.to_numpy(dtype=float)+va.away_score.to_numpy(dtype=float)
    y_review_total=re.home_score.to_numpy(dtype=float)+re.away_score.to_numpy(dtype=float)
    scores=[]
    for alpha in (10.,100.,1000.):
        model=ridge(alpha)
        model.fit(tr[score_fields],y_train_total)
        pv=model.predict(va[score_fields]); pr=model.predict(re[score_fields])
        scores.append((alpha,model,pv,pr,float(mean_absolute_error(y_val_total,pv))))
    alpha,score_model,_,review_total,_=min(scores,key=lambda x:(x[-1],x[0]))
    simple_total=re.v9_scoring_baseline_total.to_numpy(dtype=float)
    output['v9_scoring_baseline_total']=simple_total
    output['v9_scoring_model_total']=review_total
    output['actual_total']=y_review_total
    score_metrics=pd.DataFrame([
      {'model':'V9 uncalibrated possession x efficiency','season':2026,
       'games':len(re),'total_mae':round(float(mean_absolute_error(y_review_total,simple_total)),3)},
      {'model':f'V9 calibrated total Ridge alpha={alpha:g}','season':2026,
       'games':len(re),'total_mae':round(float(mean_absolute_error(y_review_total,review_total)),3)},
    ])
    paired=[]
    for family in ('strict','hybrid'):
        new=(output[f'v9_{family}_home_win_prob'].to_numpy()>=.5)==y_re
        for predecessor in ('V5','V8'):
            old=(base[f'{predecessor} {family}']>=.5)==y_re
            for pop,mask in [('all',np.ones(len(re),bool)),('elo_close',close)]:
                improvement=new[mask].astype(int)-old[mask].astype(int)
                lo,hi=v8.bootstrap_days(improvement,re.game_date.loc[mask].reset_index(drop=True))
                paired.append({'comparison':f'V9 {family} minus {predecessor} {family}',
                    'population':pop,'games':int(mask.sum()),
                    'delta_accuracy_pp':round(float(improvement.mean()*100),3),
                    'day_bootstrap_low_pp':lo,'day_bootstrap_high_pp':hi,
                    'v9_right_previous_wrong':int(((new[mask]==1)&(old[mask]==0)).sum()),
                    'previous_right_v9_wrong':int(((new[mask]==0)&(old[mask]==1)).sum())})
    coverage=data.groupby('season').agg(eligible_games=('game_id','size'),
        player_prior3_both=('v9_player_both_prior_games','sum'),
        pace_prior5_both=('v9_pace_sample_min',lambda x:int((x>=5).sum()))).reset_index()
    coverage['player_prior3_pct']=(100*coverage.player_prior3_both/coverage.eligible_games).round(2)
    return (pd.DataFrame(comparisons),pd.DataFrame(selections),pd.DataFrame(ablations),
            pd.DataFrame(paired),score_metrics,coverage,output,chosen,audit)


def report_html(comparison,selection,ablations,paired,score,coverage,chosen,audit):
    def tbl(df): return df.to_html(index=False,escape=True,classes='data',border=0)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>NCAA MBB V9 research</title>
<style>body{{font:15px system-ui,Arial;line-height:1.55;color:#1c2933;max-width:1220px;margin:35px auto;padding:0 22px}}
h1,h2{{line-height:1.2}}.notice{{background:#edf2f7;border-left:4px solid #526c85;padding:15px}}
.data{{border-collapse:collapse;display:block;max-width:100%;overflow:auto;margin:12px 0 28px}}
.data th,.data td{{white-space:nowrap;text-align:left;border-bottom:1px solid #ddd;padding:8px 11px}}
.data th{{background:#edf2f8}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:14px}}</style></head><body>
<h1>NCAA MBB V9 — Six Improvements Historical Research</h1>
<p>Created {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</p>
<div class="notice"><b>Retrospective study only.</b> Train 2022–24, choose configurations using 2025.
2026 has been examined repeatedly; it is NOT an untouched holdout. Any 2026 increase is exploratory.
Hybrid baselines use some upstream features without independently verified point-in-time provenance.
Player-box data measures prior rotations, NOT verified future availability.
Nothing here establishes next-season accuracy or guaranteed performance.</div>
<h2>V5 vs V8 vs V9 on exactly matched historical games</h2>{tbl(comparison)}
<h2>Paired accuracy changes with day-level bootstrap uncertainty</h2>{tbl(paired)}
<h2>Possession and scoring-efficiency model — projected totals</h2>{tbl(score)}
<h2>Historical player and scoring coverage</h2>{tbl(coverage)}
<h2>Exploratory feature-group tests</h2>{tbl(ablations)}
<h2>2025-only model selection</h2>{tbl(selection)}
<h2>Chosen models</h2><pre>{html.escape(json.dumps(chosen,indent=2))}</pre>
<h2>Feature-provenance and missing-data audit</h2><pre>{html.escape(json.dumps(audit,indent=2,default=str))}</pre>
<p>Close games use the predeclared absolute prior-day Elo rating gap of 120 points.
Data availability and pregame timestamp checks remain essential before deploying any prospective comparison.</p></body></html>'''


def run(raw,totals,scores,boxes,optional,reports,availability_audit,players,player_fetch_audit):
    frame,audit=v6.make_frame(raw,scores,boxes,optional,reports,availability_audit)
    data,audit,enabled=assemble_features(frame,scores,boxes,players,audit)
    audit['v9_player_source']=player_fetch_audit
    res=evaluate(data,audit,enabled)
    comparison,selection,ablation,paired,score,coverage,rows,chosen,audit=res
    OUT.mkdir(parents=True,exist_ok=True)
    for filename,obj in [
        ('model_comparison.csv',comparison),('selection_2025.csv',selection),
        ('feature_group_ablation.csv',ablation),('paired_differences_2026.csv',paired),
        ('score_total_evaluation.csv',score),('coverage.csv',coverage),
        ('historical_2026_diagnostics.csv',rows)]:
        obj.to_csv(OUT/filename,index=False)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,default=str),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps({'chosen':chosen,
       'historical_2026_metrics':comparison.to_dict('records'),
       'player_features_enabled':enabled,'2026_untouched_holdout':False,
       'prospective_validation_completed':False},indent=2),encoding='utf-8')
    (OUT/'report.html').write_text(report_html(comparison,selection,ablation,paired,
       score,coverage,chosen,audit),encoding='utf-8')
    print('V9 HISTORICAL RESEARCH — completed games ONLY')
    print(comparison.to_string(index=False))
    print(score.to_string(index=False))
    print('HTML report:',(OUT/'report.html').resolve())


def self_test():
    # The same-day invariance is the crucial regression test, especially when
    # multiple teams play on the same calendar date.
    scores=pd.DataFrame([
      dict(season=2024,game_id='1',game_date='2023-11-01',home_team_id='A',away_team_id='B',
           home_score=72,away_score=68,neutral_site=0),
      dict(season=2024,game_id='2',game_date='2023-11-01',home_team_id='C',away_team_id='D',
           home_score=77,away_score=74,neutral_site=1),
      dict(season=2024,game_id='3',game_date='2023-11-04',home_team_id='A',away_team_id='C',
           home_score=80,away_score=78,neutral_site=0),
      dict(season=2024,game_id='4',game_date='2023-11-06',home_team_id='B',away_team_id='D',
           home_score=65,away_score=62,neutral_site=0)])
    players=[]
    for g in scores.itertuples():
        for team,points in [(g.home_team_id,g.home_score),(g.away_team_id,g.away_score)]:
            for j in range(8):
                # exactly 200 team minutes and exact points totals
                pts=points//8+(points%8>j)
                players.append(dict(season=g.season,game_id=g.game_id,team_id=team,
                    player_id=f'{team}-{j}',minutes='25:00',points=pts,rebounds=2,assists=2,
                    turnovers=1))
    players=pd.DataFrame(players)
    no_player,_=player_replay(scores,pd.DataFrame())
    assert (no_player['v9_player_both_prior_games']==0).all()
    ps,_=player_replay(scores,players)
    mut=scores.copy()
    mut.loc[mut.game_date=='2023-11-01','home_score']+=100
    ps2,_=player_replay(mut,players)
    # Today's snapshots are identical regardless of that day's results, even
    # when the mutations render the CURRENT box inconsistent.
    a=ps[ps.game_id.isin(['1','2'])]
    b=ps2[ps2.game_id.isin(['1','2'])]
    assert np.allclose(a[PLAYER_FEATURES],b[PLAYER_FEATURES],equal_nan=True)
    assert minutes('21:30')==21.5 and np.isnan(minutes('DNP'))
    # Score replay: ineligible current scores or boxes cannot affect today's features.
    box=pd.DataFrame([dict(season=g.season,game_id=g.game_id,h_box_points=g.home_score,
        a_box_points=g.away_score,h_possessions=69,a_possessions=69,
        h_three_m=8,a_three_m=8,h_three_a=22,a_three_a=22,h_fg_a=60,a_fg_a=60)
        for g in scores.itertuples()])
    sc,_=scoring_replay(scores,box)
    sc2,_=scoring_replay(mut,box)
    assert np.allclose(sc[sc.game_id.isin(['1','2'])][PACE_FEATURES+FORM_FEATURES],
                       sc2[sc2.game_id.isin(['1','2'])][PACE_FEATURES+FORM_FEATURES],equal_nan=True)
    # After an earlier day, historical changes SHOULD be reflected in later dates.
    assert len(sc)==4 and np.all(np.isfinite(sc.v9_scoring_baseline_total))
    print('V9 timing, roster, possession and safety tests PASSED (synthetic; not NCAA accuracy).')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='V9 historical completed-game research')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        raw,totals,scores,boxes=v5.fetch()
        optional,reports,availability_audit=v6.optional_data()
        players,player_fetch_audit=fetch_players()
        run(raw,totals,scores,boxes,optional,reports,availability_audit,players,player_fetch_audit)
