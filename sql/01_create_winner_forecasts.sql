-- Run once in Supabase SQL Editor. Only adds a new private table.
-- Persistent forecasts are frozen before tipoff. No existing table is changed.
BEGIN;
CREATE TABLE IF NOT EXISTS public.mbb_winner_forecasts (
  game_id text NOT NULL,
  season integer NOT NULL,
  home_team_id text NOT NULL,
  away_team_id text NOT NULL,
  tipoff_utc timestamptz NOT NULL,
  feature_version text NOT NULL,
  model_version text NOT NULL,
  home_win_probability double precision NOT NULL CHECK (home_win_probability>=0 AND home_win_probability<=1),
  predicted_winner_team_id text NOT NULL,
  home_feature_recorded_at timestamptz NOT NULL,
  away_feature_recorded_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT prediction_before_tipoff CHECK(created_at < tipoff_utc),
  CONSTRAINT home_features_before_tipoff CHECK(home_feature_recorded_at<tipoff_utc),
  CONSTRAINT away_features_before_tipoff CHECK(away_feature_recorded_at<tipoff_utc),
  CONSTRAINT winner_is_one_of_the_teams CHECK (predicted_winner_team_id IN (home_team_id,away_team_id)),
  PRIMARY KEY(game_id, model_version)
);
CREATE INDEX IF NOT EXISTS mbb_winner_forecast_upcoming_idx ON public.mbb_winner_forecasts(tipoff_utc);
ALTER TABLE public.mbb_winner_forecasts ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.mbb_winner_forecasts FROM PUBLIC, anon, authenticated;
COMMIT;
-- The DATABASE_URL user needs INSERT and SELECT rights to this private table.
-- Only server-side credentials may access it. Do not put DATABASE_URL in Streamlit UI.
