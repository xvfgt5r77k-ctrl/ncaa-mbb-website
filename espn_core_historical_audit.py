"""NCAA MBB retrospective, read-only ESPN CORE predictor feasibility test.

Queries only FINISHED 2026 games from a pre-existing archived research dataset.
No future games, pregame alerts, odds, betting logic, database modifications, or model training.
Retrieved-after-final ESPN predictor fields do NOT establish pretipoff validity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import pandas as pd
import requests
import espn_probability_pilot as previous

CORE_BASE = ('https://sports.core.api.espn.com/v2/sports/basketball/'
             'leagues/mens-college-basketball/events')
ID_IN_REF = re.compile(r'/teams/([0-9]+)(?:/)?(?:\?.*)?$')


def _team_id(team: dict) -> str | None:
    """Core responses frequently link teams through $ref, not literal id."""
    if not isinstance(team, dict):
        return None
    if team.get('id') is not None:
        return str(team['id'])
    obj = team.get('team')
    if isinstance(obj, dict):
        if obj.get('id') is not None:
            return str(obj['id'])
        val = obj.get('$ref')
    else:
        val = obj
    if isinstance(val, str):
        m = ID_IN_REF.search(val)
        if m:
            return m.group(1)
    return None


def _stat_name(obj: dict) -> str:
    fields = [obj.get('name'), obj.get('abbreviation'), obj.get('shortName')]
    t = obj.get('type')
    if isinstance(t, dict):
        fields += [t.get('name'), t.get('abbreviation')]
    fields = [str(x).lower().replace('_', '').replace(' ', '') for x in fields if x]
    return next((x for x in fields if x in ('gameprojection', 'winprobability')), '')


def _projection(team: dict):
    """Accept only explicitly named win-probability fields, never score projections."""
    for key in ('gameProjection', 'winProbability'):
        if team.get(key) is not None:
            return team.get(key)
    stats = team.get('statistics')
    if isinstance(stats, dict):
        for key in ('gameProjection', 'gameprojection', 'winProbability'):
            if stats.get(key) is not None:
                v = stats[key]
                if isinstance(v, dict):
                    return v.get('value', v.get('displayValue'))
                return v
        stats = stats.get('statistics', [])
    if isinstance(stats, list):
        for item in stats:
            if not isinstance(item, dict) or not _stat_name(item):
                continue
            v = item.get('value', item.get('displayValue'))
            if v is not None:
                return v
    return None


def parse_core_predictor(payload: dict, home_id: str, away_id: str):
    """Return normalized home probability, status; reject identity ambiguity."""
    if not isinstance(payload, dict):
        return None, 'not_an_object'
    home, away = payload.get('homeTeam'), payload.get('awayTeam')
    if not isinstance(home, dict) or not isinstance(away, dict):
        return None, 'missing_team_objects'
    actual_home, actual_away = _team_id(home), _team_id(away)
    if actual_home is None or actual_away is None:
        return None, 'team_identity_unverifiable'
    if (actual_home, actual_away) != (str(home_id), str(away_id)):
        return None, 'team_identity_mismatch'
    p = previous.as_probability(_projection(home), _projection(away))
    if p is None:
        return None, 'missing_or_invalid_projection'
    return p, 'found'


def fetch_core_predictor(session: requests.Session, game_id: str,
                         home_id: str, away_id: str, pause: float = .5):
    # Most ESPN MBB events have a competition ID identical to the event ID.
    # If this assumption is false, record 404/missing rather than invent data.
    url = f'{CORE_BASE}/{game_id}/competitions/{game_id}/predictor'
    row = {'game_id':str(game_id),
           'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
           'source_url':url, 'source_field':'homeTeam/awayTeam gameProjection',
           'historical_timing_verified':False, 'status':'unknown',
           'espn_home_probability':None}
    try:
        resp = session.get(url, timeout=18)
        if resp.status_code != 200:
            row['status'] = 'rate_limited' if resp.status_code == 429 else f'http_{resp.status_code}'
        else:
            p, status = parse_core_predictor(resp.json(),home_id,away_id)
            row['espn_home_probability'], row['status'] = p,status
    except requests.RequestException:
        row['status'] = 'request_error'
    except (ValueError, TypeError):
        row['status'] = 'invalid_json'
    finally:
        if pause > 0:
            time.sleep(pause)
    return row


def export_results(outdir: Path, pool: pd.DataFrame, sample: pd.DataFrame,
                   fetched: list[dict]):
    outdir.mkdir(parents=True, exist_ok=True)
    retrieval = pd.DataFrame(fetched)
    retrieval.to_csv(outdir/'core_endpoint_audit.csv',index=False)
    joined = sample.merge(retrieval[['game_id','status','espn_home_probability',
                                     'retrieved_at_utc','source_url']],
                           on='game_id',how='left',validate='one_to_one')
    joined.to_csv(outdir/'all_games.csv',index=False)
    usable=joined.loc[joined.status.eq('found') & joined.espn_home_probability.notna()].copy()
    usable.to_csv(outdir/'same_game_comparison.csv',index=False)
    metrics, calibration=previous.evaluate(usable)
    metrics.to_csv(outdir/'model_metrics.csv',index=False)
    calibration.to_csv(outdir/'calibration.csv',index=False)
    counts = {str(k):int(v) for k,v in retrieval.status.value_counts().items()}
    summary = {
        'purpose':'Historical ESPN Core predictor source feasibility, academic research only',
        'endpoint':'/events/{event}/competitions/{event}/predictor',
        'historical_predictions_were_saved_pretipoff':False,
        'data_time_provenance_verified':False,
        'future_games_queried':0,
        'writes_to_database':False,
        'historical_reference_eligible_games':int(len(pool)),
        'requested_games':int(len(sample)),
        'competitive_requested':int(sample.competitive.sum()),
        'other_requested':int((~sample.competitive).sum()),
        'espn_forecasts_found':int(len(usable)),
        'coverage_percent':round(100*len(usable)/len(sample),2) if len(sample) else 0.0,
        'status_counts':counts,
        'sampling':'Deterministic, approximately 50:50 competitive and noncompetitive',
        'warning':'Retrospectively retrieved values are NOT verified historical pregame predictions.'
    }
    (outdir/'core_audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    lines = ['# ESPN Core Predictor — Historical Source Feasibility', '',
             '**This research queries completed 2026 NCAA MBB games only.**',
             '**ESPN predictor values retrieved now are not verified pretipoff snapshots.**', '',
             f'- Games checked: **{len(sample)}**',
             f'- Competitive games checked: **{int(sample.competitive.sum())}**',
             f'- Core predictor probabilities found: **{len(usable)}** ({summary["coverage_percent"]}%)',
             f'- Status counts: **{counts}**', '',
             '## Identical-game model comparison', '']
    if metrics.empty:
        lines.append('No comparable forecasts were found. See core_endpoint_audit.csv; no accuracy claims are possible.')
    else:
        lines.append(metrics.to_markdown(index=False))
        lines.extend(['', 'These are retrospectively retrieved ESPN values; no pretipoff accuracy claims.'])
    (outdir/'README_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',default='reference/2026_model_reference.csv')
    parser.add_argument('--output-dir',default='espn_core_historical_audit_results')
    parser.add_argument('--max-games',type=int,default=30)
    parser.add_argument('--pause-seconds',type=float,default=.5)
    args=parser.parse_args(argv)
    if not 2<=args.max_games<=120:
        parser.error('--max-games must be between 2 and 120')
    if not .3<=args.pause_seconds<=10:
        parser.error('--pause-seconds must be between 0.3 and 10 seconds')
    reference=pd.read_csv(args.reference,usecols=['game_id'],dtype={'game_id':str})
    identity=previous.fetch_identity_and_rating_gaps(reference.game_id.tolist())
    full=previous.load_eligible(Path(args.reference),identity)
    sample=previous.sample_balanced(full,args.max_games)
    client=requests.Session()
    client.headers.update({'User-Agent':'NCAA-MBB-academic-ESPN-core-audit/1.0',
                           'Accept':'application/json'})
    results=[]
    for i,row in enumerate(sample.itertuples(index=False),1):
        result=fetch_core_predictor(client,row.game_id,row.home_team_id,row.away_team_id,args.pause_seconds)
        results.append(result)
        if i%10 == 0:
            print(f'Historical records checked {i}/{len(sample)}; valid forecasts {sum(r["status"]=="found" for r in results)}',flush=True)
        if result['status']=='rate_limited':
            print('Rate limited; stopping without further requests.',flush=True)
            break
    # If rate limited, evaluate only actually-requested games; do not count unqueried rows.
    tested_ids={r['game_id'] for r in results}
    sample=sample[sample.game_id.isin(tested_ids)].copy()
    audit=export_results(Path(args.output_dir),full,sample,results)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
