"""Synthetic offline tests; never connect to a live database."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_player_ratings import prepare,evaluate,metrics,run,FEATURES


def fake_games(n=70):
    rng=np.random.default_rng(14)
    rows=[]
    for season in range(2022,2027):
        for i in range(n):
            when=pd.Timestamp(f'{season-1}-11-01T20:00:00Z') + pd.Timedelta(days=i)
            home_net=float(rng.normal(1,8))
            away_net=float(rng.normal(1,8))
            home_won = rng.random() < 1/(1+np.exp(-(home_net-away_net)/12))
            row={"game_id":f"{season}{i:04}","season":season,"game_date":str(when.date()),
                 "start_time_utc":when.isoformat(),"neutral_site":int(i%4==0),
                 "home_score":76 if home_won else 65,
                 "away_score":65 if home_won else 76}
            for side,net in (("home",home_net),("away",away_net)):
                prefix=side+"_"
                games=5+i//3
                row.update({
                    prefix+"game_day":str(when.date()),
                    prefix+"tipoff_utc":when.isoformat(),
                    prefix+"feature_version":"v2.0",
                    prefix+"quality_flags":["reconstructed_not_saved_pretipoff"],
                    prefix+"games_before":games,
                    prefix+"elo_rating":1500 + 2*net,
                    prefix+"opponent_elo_mean":1485+net,
                    prefix+"rest_days":2 if i%2==0 else 1,
                    prefix+"adjusted_offense_per100":100+net,
                    prefix+"adjusted_defense_per100":100,
                    prefix+"adjusted_net_per100":net,
                    prefix+"player_coverage":"sufficient" if i%6!=0 else "partial",
                    prefix+"expected_rotation_points_per40":10+net/5,
                    prefix+"rotation_top3_minutes_share":.5,
                    prefix+"rotation_continuity":.73,
                    prefix+"player_count":10 if i%6!=0 else 0,
                    prefix+"player_expected_minutes":190 if i%6!=0 else np.nan,
                    prefix+"non_box_source":0,
                })
            rows.append(row)
    return pd.DataFrame(rows)


def test_valid_synthetic_game_coverage():
    x,coverage,audit=prepare(fake_games())
    assert len(x)==350
    assert audit['excluded_games_unique']==0
    assert coverage.shape[0]==5
    assert set(x.season)==set(range(2022,2027))


def test_absence_not_inferred_from_missing_player_box():
    x,coverage,audit=prepare(fake_games())
    assert (~x.player_feature_ready).any()
    assert x.unverified_availability_flag.all()
    assert audit['historical_player_availability_status_verified'] is False
    assert x.loc[~x.player_feature_ready,'rotation_points_gap'].isna().all()
    assert (coverage.player_coverage_pct < 100).all()


def test_own_game_timing_and_rating_are_screened():
    d=fake_games()
    d.at[0,'home_quality_flags']=[]
    d.loc[1,'away_adjusted_net_per100']=1000
    d.loc[2,'home_tipoff_utc']='2020-01-01T00:00:00+00:00'
    x,quality,audit=prepare(d)
    assert len(x)==len(d)-3
    assert audit['home_bad_provenance_or_date']==2
    assert audit['away_bad_rating']==1


def test_no_tied_or_administrative_score():
    d=fake_games()
    d.loc[0,['home_score','away_score']]=[0,2]
    d.loc[1,['home_score','away_score']]=[81,81]
    x,_,audit=prepare(d)
    assert len(x)==len(d)-2
    assert audit['invalid_score_venue_or_tipoff']==2


def test_missing_column_fails_closed():
    with pytest.raises(ValueError,match='missing required'):
        prepare(fake_games().drop(columns=['away_tipoff_utc']))


def test_duplicate_game_id_fails_closed():
    d=fake_games()
    d.loc[1,'game_id']=d.loc[0,'game_id']
    with pytest.raises(ValueError,match='Duplicate'):
        prepare(d)


def test_chronological_selection_and_output(tmp_path):
    x,_,_=prepare(fake_games())
    val,backtest,subgroups,summary=evaluate(x)
    assert len(val)==len(FEATURES)==len(backtest)
    assert backtest.n.eq(70).all()
    assert val.n.eq(70).all()
    assert backtest.selected_on_2025.sum()==1
    assert summary['2026_is_not_untouched_holdout'] is True
    run(fake_games(),str(tmp_path))
    for name in ('audit.json','summary.json','player_and_rating_quality_by_season.csv',
                 'validation_2025.csv','retrospective_2026.csv',
                 'subgroups_2026.csv','READ_ME_FIRST.txt'):
        assert (tmp_path/name).is_file()


def test_metrics_and_empty_subgroup():
    assert metrics([1,0],[.9,.1])['correct']==2
    assert metrics([],[])['n']==0
