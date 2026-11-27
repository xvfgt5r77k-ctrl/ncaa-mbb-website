-- NCAA men's basketball feature store V2.0
-- Run ONCE in Supabase SQL Editor as the database owner. Safe to rerun.
-- Does not alter games, team_game_stats, player_game_stats or predictions.
BEGIN;

CREATE TABLE IF NOT EXISTS public.mbb_v2_team_pregame (
    game_id text NOT NULL,
    team_id text NOT NULL,
    season integer NOT NULL,
    game_day date NOT NULL,
    tipoff_utc timestamptz,
    snapshot_kind text NOT NULL CHECK (snapshot_kind IN ('historical_replay','prospective')),
    feature_version text NOT NULL DEFAULT 'v2.0',
    recorded_at timestamptz NOT NULL DEFAULT now(),
    games_before integer NOT NULL CHECK (games_before >= 0),
    elo_rating double precision,
    adjusted_offense_per100 double precision,
    adjusted_defense_per100 double precision,
    adjusted_net_per100 double precision,
    opponent_elo_mean double precision,
    estimated_possessions double precision,
    average_points_for double precision,
    average_points_against double precision,
    effective_fg_pct double precision,
    three_point_rate double precision,
    turnover_per_possession double precision,
    free_throw_rate double precision,
    rest_days integer,
    prior_player_box_games integer NOT NULL DEFAULT 0,
    player_coverage text NOT NULL CHECK(player_coverage IN ('unavailable','partial','sufficient')),
    expected_rotation_points_per40 double precision,
    rotation_top3_minutes_share double precision,
    rotation_continuity double precision,
    quality_flags text[] NOT NULL DEFAULT ARRAY[]::text[],
    PRIMARY KEY (game_id, team_id, snapshot_kind, feature_version)
);

CREATE TABLE IF NOT EXISTS public.mbb_v2_player_pregame (
    game_id text NOT NULL,
    team_id text NOT NULL,
    player_id text NOT NULL,
    season integer NOT NULL,
    game_day date NOT NULL,
    tipoff_utc timestamptz,
    snapshot_kind text NOT NULL CHECK (snapshot_kind IN ('historical_replay','prospective')),
    feature_version text NOT NULL DEFAULT 'v2.0',
    recorded_at timestamptz NOT NULL DEFAULT now(),
    prior_games_seen integer NOT NULL,
    prior_minutes double precision NOT NULL,
    previous_five_expected_minutes double precision NOT NULL,
    smoothed_points_per_minute double precision,
    prior_box_score_only boolean NOT NULL DEFAULT TRUE,
    PRIMARY KEY (game_id, team_id, player_id, snapshot_kind, feature_version)
);

CREATE TABLE IF NOT EXISTS public.mbb_v2_refresh_log (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    status text NOT NULL CHECK (status IN ('running','success','failed')),
    source_games integer NOT NULL DEFAULT 0,
    source_team_boxes integer NOT NULL DEFAULT 0,
    source_player_boxes integer NOT NULL DEFAULT 0,
    team_rows_inserted integer NOT NULL DEFAULT 0,
    player_rows_inserted integer NOT NULL DEFAULT 0,
    details jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS mbb_v2_team_season_day_idx
    ON public.mbb_v2_team_pregame (season, game_day, snapshot_kind);
CREATE INDEX IF NOT EXISTS mbb_v2_player_season_day_idx
    ON public.mbb_v2_player_pregame (season, game_day, snapshot_kind);
CREATE INDEX IF NOT EXISTS mbb_v2_team_recorded_idx
    ON public.mbb_v2_team_pregame (recorded_at);

-- These are research feature tables, NOT public APIs.
ALTER TABLE public.mbb_v2_team_pregame ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mbb_v2_player_pregame ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mbb_v2_refresh_log ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.mbb_v2_team_pregame, public.mbb_v2_player_pregame,
    public.mbb_v2_refresh_log FROM PUBLIC, anon, authenticated;

-- Owner/service connection used through DATABASE_URL retains its existing privileges.
-- Never expose DATABASE_URL or these records in client-side code.
COMMIT;

-- Run after your first successful import to inspect coverage:
-- SELECT season, snapshot_kind, COUNT(*) AS team_feature_rows,
--        COUNT(*) FILTER (WHERE player_coverage='sufficient') AS sufficient_player_rows
-- FROM public.mbb_v2_team_pregame GROUP BY season, snapshot_kind ORDER BY 1,2;
