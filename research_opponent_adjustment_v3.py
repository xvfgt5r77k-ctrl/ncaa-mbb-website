"""Read-only research pilot: point-in-time opponent-adjusted NCAA MBB ratings.

Separate from V2: no production table writes, no injury assumptions, no picks.
V3 is a time-decayed, ridge-regularized two-way offensive/defensive model.
All matches on a calendar date use ratings fit exclusively on earlier dates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from itertools import groupby
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

import db_upgrade_v2 as v2

VERSION = "v3-oppadj-research-1"
BASE_EFF = 100.0
LEAGUE_PRIOR_OBSERVATIONS = 25.0


def within_date(v: Any) -> date:
    return v2.game_day(v)


@dataclass(frozen=True)
class Observation:
    game_id: str
    game_day: date
    home: str
    away: str
    home_eff: float
    away_eff: float
    venue: int    # 1 for nonneutral home, 0 for neutral
    possessions: float


def observe(g: dict, boxes: dict[tuple[str, str], dict], tolerance: float = 6.0) -> Observation | None:
    """Require matching, plausible team boxes and possession estimates on both sides."""
    if not v2.valid_score(g):
        return None
    gid, home, away = g["game_id"], g["home_team_id"], g["away_team_id"]
    hb = boxes.get((gid, home), {})
    ab = boxes.get((gid, away), {})
    if not hb or not ab:
        return None
    for bx, score in ((hb, g["home_score"]), (ab, g["away_score"])):
        stated = v2.number(bx.get("points"))
        if stated is not None and abs(stated - float(score)) > 0.01:
            return None
    ph, pa = v2.get_possessions(hb), v2.get_possessions(ab)
    if ph is None or pa is None or abs(ph-pa) > tolerance:
        return None
    poss = 0.5 * (ph+pa)
    if not (30 <= poss <= 130):
        return None
    neutral = v2.number(g.get("neutral_site")) == 1
    return Observation(gid, g["game_day"], home, away,
                       100.0 * float(g["home_score"]) / poss,
                       100.0 * float(g["away_score"]) / poss,
                       0 if neutral else 1, poss)


def fit_before_day(history: list[Observation], teams: list[str], day: date,
                   half_life_days: float = 45.0, shrinkage_games: float = 8.0,
                   home_offense_bonus: float = 1.0) -> tuple[dict[str, dict], float]:
    """Solve eff_i,j=mu + O_i - D_j + venue_i*home_bonus via ridge LS.

    Both O and D are centered around zero via ridge, avoiding an unidentifiable
    team-by-team two-way decomposition. History must be strictly earlier than day.
    """
    if half_life_days <= 0 or shrinkage_games <= 0:
        raise ValueError("half_life_days and shrinkage_games must be positive")
    if not math.isfinite(home_offense_bonus):
        raise ValueError("home_offense_bonus must be finite")
    team_ids = sorted(set(teams))
    lookup = {tid: i for i, tid in enumerate(team_ids)}
    n = len(team_ids)
    if not n:
        return {}, BASE_EFF
    if any(o.game_day >= day for o in history):
        raise ValueError("future/same-day observation would leak into prior-day ratings")
    weights = np.array([math.exp(-math.log(2.0) * (day - o.game_day).days / half_life_days)
                        for o in history], dtype=float)
    weighted_obs = float(2.0 * weights.sum())
    league = ((BASE_EFF * LEAGUE_PRIOR_OBSERVATIONS +
               sum(float(w) * (o.home_eff+o.away_eff) for o,w in zip(history,weights))) /
              (LEAGUE_PRIOR_OBSERVATIONS + weighted_obs))
    if not history:
        return ({tid: {"offense": league, "defense": league, "net": 0.0,
                       "effective_games": 0.0, "rated_games": 0, "sos": None,
                       "confidence": 0.0} for tid in team_ids}, league)

    indices_row: list[int] = []
    indices_col: list[int] = []
    values: list[float] = []
    y: list[float] = []
    opponent_history: dict[str, list[tuple[str, float]]] = defaultdict(list)
    eff_games: dict[str, int] = defaultdict(int)
    eff_weights: dict[str, float] = defaultdict(float)
    for k, (o,w) in enumerate(zip(history,weights)):
        for pos, (team,opp,eff,venue) in enumerate(((o.home,o.away,o.home_eff,o.venue),
                                                    (o.away,o.home,o.away_eff,-o.venue))):
            r=2*k+pos
            scale=math.sqrt(float(w))
            indices_row.extend([r,r])
            indices_col.extend([lookup[team], n+lookup[opp]])
            values.extend([scale,-scale])
            y.append(scale * (eff-league-venue*home_offense_bonus))
            opponent_history[team].append((opp,float(w)))
            eff_games[team]+=1
            eff_weights[team]+=float(w)
    design=coo_matrix((np.asarray(values), (indices_row, indices_col)),
                      shape=(len(y),2*n)).tocsr()
    fitted=lsqr(design,np.asarray(y,dtype=float),damp=math.sqrt(shrinkage_games),
                atol=1e-8,btol=1e-8,iter_lim=350)[0]
    out={}
    for tid,i in lookup.items():
        off=league+float(fitted[i])
        defense=league-float(fitted[n+i])
        out[tid]={"offense": off,"defense": defense,"net": off-defense,
                  "effective_games": eff_weights[tid], "rated_games": eff_games[tid],
                  "confidence": eff_weights[tid]/(eff_weights[tid]+shrinkage_games),
                  "sos": None}
    for tid, opps in opponent_history.items():
        den=sum(w for _,w in opps)
        out[tid]["sos"]=sum(w*out[opp]["net"] for opp,w in opps)/den if den else None
    return out, league


def generate(games: list[dict], boxes: dict[tuple[str,str],dict],
             now: datetime | None = None, half_life_days: float = 45,
             shrinkage_games: float = 8, home_offense_bonus: float = 1) -> tuple[list[dict],dict[str,Observation],dict]:
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    source=v2.d1_games_sorted(games)
    rows: list[dict] = []
    observations: dict[str,Observation] = {}
    counts=defaultdict(int)
    for season, season_iter in groupby(source, key=lambda g:g["season"]):
        entire_season=list(season_iter)
        team_ids=sorted({t for g in entire_season for t in (g["home_team_id"],g["away_team_id"])})
        history: list[Observation]=[]
        for day, day_iter in groupby(entire_season,key=lambda g:g["game_day"]):
            day_games=list(day_iter)
            fitted, league=fit_before_day(history, team_ids, day, half_life_days,
                                          shrinkage_games, home_offense_bonus)
            new_observations=[]
            for g in day_games:
                gid,hid,aid=g["game_id"],g["home_team_id"],g["away_team_id"]
                kind=v2.choose_kind(g,now)
                if kind is not None:
                    for tid,oid in ((hid,aid),(aid,hid)):
                        s=fitted[tid]
                        rows.append({"game_id":gid,"team_id":tid,"opponent_id":oid,
                                     "season":season,"game_day":day.isoformat(),
                                     "snapshot_kind":kind,"feature_version":VERSION,
                                     "games_before":s["rated_games"],
                                     "effective_games":round(s["effective_games"],5),
                                     "confidence":round(s["confidence"],5),
                                     "adjusted_offense_per100":round(s["offense"],6),
                                     "adjusted_defense_per100":round(s["defense"],6),
                                     "adjusted_net_per100":round(s["net"],6),
                                     "opponent_sos":round(s["sos"],6) if s["sos"] is not None else None,
                                     "league_efficiency":round(league,6),
                                     "home_offense_bonus_per100":home_offense_bonus,
                                     "history_games_in_fit":len(history)})
                    counts[kind+"_games"]+=1
                obs=observe(g,boxes)
                if obs is not None:
                    # This day's result only participates in a LATER day's fit.
                    new_observations.append(obs)
                    observations[gid]=obs
                    counts["usable_completed_box_game"]+=1
                elif g["completed"]:
                    counts["completed_without_usable_pair_boxes"]+=1
            history.extend(new_observations)
    counts["source_games"]=len(source)
    counts["team_snapshot_rows"]=len(rows)
    return rows,observations,dict(counts)


def load_database(url: str, start_season: int, end_season: int):
    """Read-only source retrieval, matching the schema handled by V2 without player loads."""
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
    with psycopg.connect(url,sslmode="require",connect_timeout=30,prepare_threshold=None,
                         row_factory=dict_row) as conn:
        conn.read_only=True
        conn.isolation_level=psycopg.IsolationLevel.REPEATABLE_READ
        with conn.cursor() as cur:
            cur.execute("""
                SELECT g.game_id::text game_id, g.season, g.game_date::text game_date,
                       g.start_time_utc::text start_time_utc,
                       g.home_team_id::text home_team_id,g.away_team_id::text away_team_id,
                       g.neutral_site,g.home_score,g.away_score,g.completed
                  FROM public.games g
                  JOIN public.teams ht ON ht.team_id=g.home_team_id AND ht.is_d1=1
                  JOIN public.teams at ON at.team_id=g.away_team_id AND at.is_d1=1
                 WHERE g.season BETWEEN %s AND %s AND g.game_date IS NOT NULL
                 ORDER BY g.season,g.game_date,g.game_id
            """,(start_season,end_season))
            games=cur.fetchall()
            if not games:
                raise RuntimeError("No D1 games found for selected seasons")
            cur.execute("""SELECT column_name FROM information_schema.columns
                            WHERE table_schema='public' AND table_name='team_game_stats'""")
            cols={r["column_name"] for r in cur.fetchall()}
            if not {"game_id","team_id"}<=cols:
                raise RuntimeError("team_game_stats must have game_id and team_id")
            tmap={name:next((a for a in opts if a in cols),None)
                  for name,opts in v2.TEAM_ALIASES.items()}
            selected=[sql.SQL("s.game_id::text game_id"),sql.SQL("s.team_id::text team_id")]
            for name,column in tmap.items():
                expr=sql.SQL("NULL::text") if column is None else sql.SQL("s.{}::text").format(sql.Identifier(column))
                selected.append(sql.SQL("{} AS {}").format(expr,sql.Identifier(name)))
            stmt=sql.SQL("""SELECT {} FROM public.team_game_stats s
                JOIN public.games g ON g.game_id=s.game_id
                WHERE g.season BETWEEN %s AND %s AND g.completed=1""").format(sql.SQL(", ").join(selected))
            cur.execute(stmt,(start_season,end_season))
            boxes={}
            for r in cur:
                key=(r["game_id"],r["team_id"])
                if key in boxes:
                    raise RuntimeError(f"Duplicate team box: {key}")
                boxes[key]=dict(r)
            cur.execute("SELECT to_regclass('public.mbb_v2_team_pregame') IS NOT NULL AS exists")
            v2_rows={}
            if cur.fetchone()["exists"]:
                cur.execute("""
                    SELECT t.game_id::text game_id,t.team_id::text team_id,t.season,
                           t.adjusted_offense_per100,t.adjusted_defense_per100
                    FROM public.mbb_v2_team_pregame t
                    WHERE t.season BETWEEN %s AND %s AND t.snapshot_kind='historical_replay'
                      AND t.feature_version='v2.0'
                """,(start_season,end_season))
                for r in cur:
                    key=(r["game_id"],r["team_id"])
                    if key in v2_rows:
                        raise RuntimeError(f"Duplicate V2 historical feature: {key}")
                    v2_rows[key]={"offense":float(r["adjusted_offense_per100"]),
                                   "defense":float(r["adjusted_defense_per100"])}
            return games,boxes,v2_rows,tmap


def compare(rows: list[dict], observations: dict[str,Observation],v2_rows:dict,
            home_offense_bonus:float=1.0) -> tuple[list[dict],dict]:
    by_game=defaultdict(dict)
    for r in rows:
        if r["snapshot_kind"]=="historical_replay":
            by_game[r["game_id"]][r["team_id"]]=r
    paired=[]
    for gid,o in observations.items():
        sides=by_game.get(gid,{})
        if o.home not in sides or o.away not in sides:
            continue
        h,a=sides[o.home],sides[o.away]
        league=h["league_efficiency"]
        h_pred=(h["adjusted_offense_per100"]+a["adjusted_defense_per100"]-league+
                o.venue*home_offense_bonus)
        a_pred=(a["adjusted_offense_per100"]+h["adjusted_defense_per100"]-league-
                o.venue*home_offense_bonus)
        old_h,old_a=v2_rows.get((gid,o.home)),v2_rows.get((gid,o.away))
        if old_h is None or old_a is None:
            continue  # fair exact same-game sample
        v2_h=old_h["offense"]+old_a["defense"]-BASE_EFF+o.venue*home_offense_bonus
        v2_a=old_a["offense"]+old_h["defense"]-BASE_EFF-o.venue*home_offense_bonus
        paired.append({"game_id":gid,"season":h["season"],"game_day":h["game_day"],
                       "venue":"home_away" if o.venue else "neutral", "possessions":round(o.possessions,3),
                       "actual_home_eff":round(o.home_eff,5),"actual_away_eff":round(o.away_eff,5),
                       "v3_home_eff":round(h_pred,5),"v3_away_eff":round(a_pred,5),
                       "v2_home_eff":round(v2_h,5),"v2_away_eff":round(v2_a,5),
                       "v3_abs_error_mean":0.5*(abs(h_pred-o.home_eff)+abs(a_pred-o.away_eff)),
                       "v2_abs_error_mean":0.5*(abs(v2_h-o.home_eff)+abs(v2_a-o.away_eff)),
                       "v3_sq_error_mean":0.5*((h_pred-o.home_eff)**2+(a_pred-o.away_eff)**2),
                       "v2_sq_error_mean":0.5*((v2_h-o.home_eff)**2+(v2_a-o.away_eff)**2),
                       "home_games_before":h["games_before"],"away_games_before":a["games_before"]})
    summary={}
    for season in sorted(set(r["season"] for r in paired)):
        subset=[r for r in paired if r["season"]==season]
        n=len(subset)
        v3_mae=sum(r["v3_abs_error_mean"] for r in subset)/n
        v2_mae=sum(r["v2_abs_error_mean"] for r in subset)/n
        v3_rmse=math.sqrt(sum(r["v3_sq_error_mean"] for r in subset)/n)
        v2_rmse=math.sqrt(sum(r["v2_sq_error_mean"] for r in subset)/n)
        # Paired game-level bootstrap; repeated outcomes of one game stay together.
        differences=np.array([r["v2_abs_error_mean"]-r["v3_abs_error_mean"] for r in subset])
        rng=np.random.default_rng(20261127+int(season))
        means=np.array([float(np.mean(differences[rng.integers(0,n,size=n)])) for _ in range(400)])
        summary[str(season)]={"matched_games":n,"v3_mae":round(v3_mae,5),
            "v2_mae":round(v2_mae,5),"v3_rmse":round(v3_rmse,5),
            "v2_rmse":round(v2_rmse,5),"v2_minus_v3_mae":round(v2_mae-v3_mae,5),
            "bootstrap_95pct_mae_improvement_interval":list(map(lambda x:round(float(x),5),
                np.quantile(means,[0.025,0.975])))}
    return paired,summary


def diagnostic_slices(paired: list[dict]) -> dict:
    """Same-sample quality by matchup age and venue to detect hidden regressions."""
    subsets={
      "both_teams_5plus_prior_games": [r for r in paired if min(r["home_games_before"],r["away_games_before"])>=5],
      "either_team_under_5_prior_games": [r for r in paired if min(r["home_games_before"],r["away_games_before"])<5],
      "neutral_site": [r for r in paired if r["venue"]=="neutral"],
      "home_away_site": [r for r in paired if r["venue"]=="home_away"],
    }
    answer={}
    for name,rows in subsets.items():
        if rows:
            answer[name]={"games":len(rows),
                "v2_mae":round(sum(r["v2_abs_error_mean"] for r in rows)/len(rows),5),
                "v3_mae":round(sum(r["v3_abs_error_mean"] for r in rows)/len(rows),5)}
    return answer


def write_csv(path:Path,rows:list[dict],fallback_fields:list[str]):
    fields=list(rows[0]) if rows else fallback_fields
    with path.open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description="Read-only NCAA team opponent adjustment pilot")
    p.add_argument("--min-season",type=int,default=2022)
    p.add_argument("--max-season",type=int,default=2026,
                   help="Last retrospective season; avoid using a holdout for tuning")
    p.add_argument("--half-life-days",type=float,default=45.0)
    p.add_argument("--shrinkage-games",type=float,default=8.0)
    p.add_argument("--home-offense-bonus",type=float,default=1.0)
    p.add_argument("--output-dir",default="opponent_v3_report")
    args=p.parse_args()
    if args.min_season>args.max_season:
        p.error("min-season cannot exceed max-season")
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    db=os.environ.get('DATABASE_URL')
    if not db:
        raise SystemExit('DATABASE_URL must be supplied server-side for read-only research')
    games,boxes,v2_rows,tmap=load_database(db,args.min_season,args.max_season)
    team,observed,stats=generate(games,boxes,half_life_days=args.half_life_days,
                shrinkage_games=args.shrinkage_games,home_offense_bonus=args.home_offense_bonus)
    paired,by_season=compare(team,observed,v2_rows,args.home_offense_bonus)
    write_csv(out/'v3_historical_team_snapshots.csv',
              [r for r in team if r['snapshot_kind']=='historical_replay'],
              ['game_id','team_id','season','adjusted_net_per100'])
    write_csv(out/'v3_vs_v2_matched_efficiency.csv',paired,
              ['game_id','season','v3_abs_error_mean','v2_abs_error_mean'])
    report={"version":VERSION,"research_only":True,"db_writes":False,
      "feature_source":"historical_replay; as-of source publication timestamps unverified",
      "no_injuries_used":True,"params":{"half_life_days":args.half_life_days,
      "shrinkage_games":args.shrinkage_games,"home_offense_bonus":args.home_offense_bonus},
      "coverage":{"source_games":len(games),"source_team_boxes":len(boxes),
                  "existing_v2_rows":len(v2_rows),**stats,"matched_comparison_games":len(paired)},
      "box_column_mapping":tmap,"performance_by_season":by_season,
      "diagnostic_slices_all_matched_seasons":diagnostic_slices(paired),
      "notes":["Lower MAE/RMSE is better; these are offensive-efficiency errors, NOT winner accuracy.",
               "Equal sample and identical venue correction for V2 and V3.",
               "All game-day features exclude same-day and future results.",
               "Historical ESPN backfills may include later source corrections.",
               "2026 outcomes have been inspected in previous research; not untouched holdout.",
               "No current-production models or Supabase tables are modified."]}
    (out/'summary.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    summary=["NCAA MBB V3 opponent-adjustment research (READ ONLY)",
             f"Source games: {len(games)} | usable box games: {len(observed)} | paired V2 comparison games: {len(paired)}",
             "MAE/RMSE below refer to offensive efficiency (points per 100 possessions), NOT winner accuracy.",""]
    for season,v in by_season.items():
        summary.append(f"{season}: n={v['matched_games']}  V2 MAE={v['v2_mae']}  V3 MAE={v['v3_mae']}  V2-V3={v['v2_minus_v3_mae']}")
    if not by_season:
        summary.append('NO PAIRED DATA: Check that V2 2022-2026 historical_replay feature rows exist.')
    summary.append('Do not promote this experimental rating without independent prospective evaluation.')
    (out/'summary.txt').write_text('\n'.join(summary)+'\n',encoding='utf-8')
    print('\n'.join(summary))

if __name__=='__main__':
    main()
