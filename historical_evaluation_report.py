"""Read-only report from a frozen, SHA-verified NCAA men's basketball history ZIP.

No database access, model training, wagers, future-game picks, or third-party APIs.
Run: python historical_evaluation_report.py --archive path/to/ncaa_mbb_history_*.zip
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, date, timezone
import hashlib
from html import escape as html_escape
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

REQUIRED = {"historical_coverage_audit", "historical_score_baseline_audit", "pregame_team_features"}
MAX_CSV_BYTES = 200 * 1024 * 1024
MAX_TOTAL_BYTES = 750 * 1024 * 1024


def _float(value):
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _int(value):
    try:
        f = float(value)
        return int(f) if math.isfinite(f) and f.is_integer() else None
    except (ValueError, TypeError, OverflowError):
        return None


def _date(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _pct(a, b):
    return round(100 * a / b, 2) if b and a is not None else None


def _r(value, digits=2):
    return round(value, digits) if value is not None and math.isfinite(value) else None


def verified_archive(archive: Path):
    """Verify CRC, manifest's stable content ID, CSV hashes, paths, and size limits."""
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("Duplicate entries or no manifest.json in historical ZIP")
        if any(i.is_dir() or i.filename.startswith("/") or "/" in i.filename or "\\" in i.filename
               for i in infos):
            raise ValueError("Unexpected ZIP entry path")
        if sum(i.file_size for i in infos) > MAX_TOTAL_BYTES or any(
                i.file_size > MAX_CSV_BYTES for i in infos if i.filename.endswith(".csv")):
            raise ValueError("Archive exceeds safe historical export size limits")
        manifest = json.loads(z.read("manifest.json"))
        files = manifest.get("files")
        if not isinstance(files, dict) or not REQUIRED <= files.keys():
            raise ValueError("Required historical tables missing from manifest")
        if set(names) != {"manifest.json"} | {f"{k}.csv" for k in files}:
            raise ValueError("ZIP and manifest file lists disagree")
        hashes = {}
        for key, meta in files.items():
            name = f"{key}.csv"
            if key not in ("games", "teams", "team_game_stats", "pregame_team_features",
                           "historical_team_snapshots", "team_adjusted_ratings",
                           "historical_coverage_audit", "historical_score_baseline_audit"):
                raise ValueError("Unexpected dataset table")
            if meta.get("file") != name:
                raise ValueError(f"Unexpected manifest filename: {name}")
            digest = hashlib.sha256()
            with z.open(name) as source:
                for part in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(part)
            hashes[key] = digest.hexdigest()
            if hashes[key] != meta.get("sha256"):
                raise ValueError(f"SHA-256 mismatch in {name}")
        content_id = hashlib.sha256(json.dumps(
            {k: hashes[k] for k in sorted(hashes)}, sort_keys=True
        ).encode("utf-8")).hexdigest()[:12]
        if content_id != manifest.get("data_content_id"):
            raise ValueError("Historical dataset content ID does not match CSV hashes")
        if not isinstance(manifest.get("season_range_inclusive"), list) or len(manifest["season_range_inclusive"]) != 2:
            raise ValueError("Invalid season range in manifest")
        return manifest


def rows_from_archive(archive: Path, key: str, manifest: dict):
    expected = manifest["files"][key]
    with zipfile.ZipFile(archive) as z, z.open(f"{key}.csv") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or not set(expected["columns"]) <= set(reader.fieldnames):
                raise ValueError(f"Columns missing from {key}.csv")
            n = 0
            for row in reader:
                n += 1
                yield row
    if n != expected["rows"]:
        raise ValueError(f"Row count mismatch in {key}.csv: {n} versus {expected['rows']}")


def _true(value) -> bool:
    """Interpret booleans serialized from PostgreSQL into the frozen CSV."""
    return str(value).strip().lower() in ("true", "t", "1", "yes")


def _valid_d1_boxscore(game: dict, home: dict | None, away: dict | None) -> bool:
    """Match the original pregame_quality._usable criteria for one completed game."""
    if not home or not away or _date(game.get("game_date")) is None:
        return False
    if game.get("home_team_id") == game.get("away_team_id"):
        return False
    for score, box in ((game.get("home_score"), home.get("points")),
                       (game.get("away_score"), away.get("points"))):
        a, b = _float(score), _float(box)
        if a is None or b is None or a < 0 or b < 0 or a != b:
            return False
    return all(_float(s.get("possessions")) is not None
               and _float(s.get("possessions")) > 0 for s in (home, away))


def d1_coverage_from_archive(archive: Path, manifest: dict) -> dict[int, dict]:
    """Recount completed D1 vs D1 matchups and their usable box scores.

    The stored historical_coverage_audit.usable_stat_games includes non-D1
    games, so it must NEVER be divided by its D1-only d1_matchups column.
    """
    needed = {"games", "teams", "team_game_stats"}
    if not needed <= manifest["files"].keys():
        raise ValueError("Frozen export lacks games/teams/team_game_stats; cannot calculate comparable D1 coverage")
    is_d1 = {}
    for row in rows_from_archive(archive, "teams", manifest):
        team_id = row["team_id"]
        if team_id in is_d1:
            raise ValueError(f"Duplicate exported team ID {team_id}")
        is_d1[team_id] = _true(row["is_d1"])
    eligible = {}
    counts = defaultdict(lambda: {"d1_matchups": 0, "usable_d1_games": 0})
    for g in rows_from_archive(archive, "games", manifest):
        if not _true(g.get("completed")):
            continue
        if not (is_d1.get(g["home_team_id"], False) and is_d1.get(g["away_team_id"], False)):
            continue
        year = _int(g.get("season"))
        if year is None:
            raise ValueError("Invalid season in exported games")
        counts[year]["d1_matchups"] += 1
        if g["game_id"] in eligible:
            raise ValueError(f"Duplicate exported game ID {g['game_id']}")
        eligible[g["game_id"]] = g
    stat_index = defaultdict(dict)
    for s in rows_from_archive(archive, "team_game_stats", manifest):
        game = eligible.get(s["game_id"])
        if game is None or s["team_id"] not in (game["home_team_id"], game["away_team_id"]):
            continue
        indexed = stat_index[s["game_id"]]
        if s["team_id"] in indexed:
            raise ValueError(f"Duplicate box score for {s['game_id']} / {s['team_id']}")
        indexed[s["team_id"]] = s
    for game_id, g in eligible.items():
        scores = stat_index.get(game_id, {})
        if _valid_d1_boxscore(g, scores.get(g["home_team_id"]), scores.get(g["away_team_id"])):
            counts[int(g["season"])]["usable_d1_games"] += 1
    return dict(counts)


def report_tables(manifest: dict, coverage_rows: list[dict], baseline_rows: list[dict], features: list[dict], d1_coverage: dict[int, dict] | None = None):
    start, end = [int(y) for y in manifest["season_range_inclusive"]]
    years = list(range(start, end + 1))
    expected = set(years)
    coverage_by_year = {int(r["season"]): r for r in coverage_rows}
    baseline_by_year = {int(r["season"]): r for r in baseline_rows}
    if len(coverage_by_year) != len(coverage_rows) or len(baseline_by_year) != len(baseline_rows):
        raise ValueError("Duplicate season rows in historical audit")
    feature_by_year = defaultdict(lambda: {"pregame_rows": 0, "five_plus": 0,
                                           "prior_ratings": 0, "violations": 0,
                                           "unsafe_ratings": 0, "negative_history": 0})
    drift_series = defaultdict(lambda: defaultdict(list))
    for row in features:
        yr = _int(row.get("season"))
        if yr not in expected:
            raise ValueError("Pregame features contain a season outside exported season range")
        data = feature_by_year[yr]
        data["pregame_rows"] += 1
        history = _int(row.get("games_before"))
        if history is None or history < 0:
            data["negative_history"] += 1
        snap = _date(row.get("rating_snapshot_date")) if row.get("rating_snapshot_date") else None
        game = _date(row.get("game_date"))
        invalid_date = (game is None or (row.get("rating_snapshot_date") and snap is None)
                        or (snap is not None and game is not None and snap >= game))
        if invalid_date:
            data["violations"] += 1
        has_rating = bool(row.get("adjusted_net_before") or row.get("opponent_sos_before"))
        if has_rating and (snap is None or invalid_date):
            data["unsafe_ratings"] += 1
        if history is not None and history >= 5:
            data["five_plus"] += 1
            if snap and not invalid_date and _float(row.get("adjusted_net_before")) is not None:
                data["prior_ratings"] += 1
            for metric in ("avg_points_for", "avg_possessions", "adjusted_net_before"):
                # A flagged rating must not enter a drift calculation.
                if metric == "adjusted_net_before" and (not snap or invalid_date):
                    continue
                value = _float(row.get(metric))
                if value is not None:
                    drift_series[metric][yr].append(value)

    coverage = []
    pregame = []
    baseline = []
    flags = []
    for year in years:
        c = coverage_by_year.get(year)
        b = baseline_by_year.get(year)
        f = feature_by_year[year]
        if not c:
            flags.append(f"Missing historical coverage audit for season {year}.")
        if not b:
            flags.append(f"Missing historical scoring audit for season {year}.")
        if f["pregame_rows"] == 0:
            flags.append(f"No pregame feature rows for season {year}.")
        den = _int(c.get("d1_matchups")) if c else None
        all_usable = _int(c.get("usable_stat_games")) if c else None
        comparable = d1_coverage.get(year) if d1_coverage is not None else None
        usable = _int(comparable.get("usable_d1_games")) if comparable else None
        if d1_coverage is not None and comparable is None:
            flags.append(f"Season {year}: no D1 games in the frozen archive.")
        if comparable and _int(comparable.get("d1_matchups")) != den:
            flags.append(f"Season {year}: archived D1 matchup count disagrees with stored audit.")
        mismatches = _int(c.get("score_mismatches")) if c else None
        possessions = _int(c.get("incomplete_possessions")) if c else None
        pct = _pct(usable, den)
        coverage.append({"Season": f"{year-1}–{str(year)[-2:]}", "ESPN season": year,
                         "D1 completed matchups": den, "Usable D1 games": usable,
                         "D1 usable %": pct, "All usable completed games": all_usable,
                         "All-game score mismatches": mismatches,
                         "All-game invalid possessions": possessions})
        if pct is None or pct < 95 or pct > 100:
            flags.append(f"Season {year}: comparable D1 coverage {pct if pct is not None else 'unavailable'}%; expected 95–100%.")
        if mismatches is None or mismatches > 0:
            flags.append(f"Season {year}: score mismatches {mismatches if mismatches is not None else 'unavailable'}.")
        if possessions is None or possessions > 0:
            flags.append(f"Season {year}: invalid/missing possessions {possessions if possessions is not None else 'unavailable'}.")
        pregame.append({"Season": f"{year-1}–{str(year)[-2:]}", "ESPN season": year,
                        "Pregame team rows": f["pregame_rows"],
                        "Rows with 5+ previous games": f["five_plus"],
                        "Valid prior adj. rating rows": f["prior_ratings"],
                        "Prior rating coverage %": _pct(f["prior_ratings"], f["five_plus"]),
                        "Non-prior/invalid dates": f["violations"],
                        "Unsafe rating values": f["unsafe_ratings"],
                        "Invalid previous-game counts": f["negative_history"]})
        if f["violations"] or f["unsafe_ratings"] or f["negative_history"]:
            flags.append(f"Season {year}: {f['violations']} bad dates, {f['unsafe_ratings']} unsafe ratings, {f['negative_history']} invalid history counts.")
        rating_pct = _pct(f["prior_ratings"], f["five_plus"])
        if rating_pct is None or rating_pct < 75:
            flags.append(f"Season {year}: prior adjusted rating coverage {rating_pct if rating_pct is not None else 'unavailable'}% (<75% review threshold).")
        games = _int(b.get("evaluated_games")) if b else None
        eligible = _int(b.get("eligible_d1_games")) if b else None
        score_count = _int(b.get("evaluated_team_scores")) if b else None
        if b and (games is None or games <= 0 or score_count is None or score_count != 2 * games):
            flags.append(f"Season {year}: benchmark game/team-score counts need review.")
        baseline.append({"Season": f"{year-1}–{str(year)[-2:]}", "ESPN season": year,
                         "Historical split": b.get("split_name") if b else "Unavailable",
                         "Eligible D1 games": eligible, "Evaluated games": games,
                         "Evaluation coverage %": _pct(games, eligible),
                         "Team scores": score_count,
                         "MAE (pts)": _r(_float(b.get("points_mae"))) if b else None,
                         "RMSE (pts)": _r(_float(b.get("points_rmse"))) if b else None,
                         "Bias (pts)": _r(_float(b.get("points_bias"))) if b else None})

    drift = []
    if {2022, 2023, 2024, 2025, 2026} <= expected:
        for metric in ("avg_points_for", "avg_possessions", "adjusted_net_before"):
            dev = [value for yr in (2022, 2023, 2024) for value in drift_series[metric][yr]]
            sd = statistics.stdev(dev) if len(dev) > 1 else None
            mean = statistics.mean(dev) if dev else None
            for year, label in ((2025, "2024–25 validation"), (2026, "2025–26 holdout")):
                newer = drift_series[metric][year]
                shift = ((statistics.mean(newer) - mean) / sd
                         if newer and mean is not None and sd is not None and sd > 0 else None)
                drift.append({"Pregame statistic": metric,
                              "Comparison": label,
                              "Development rows": len(dev), "Comparison rows": len(newer),
                              "Mean change / dev SD": _r(shift, 3),
                              "Review shift ≥0.5 SD": "Review" if shift is not None and abs(shift) >= 0.5
                              else "Within threshold" if shift is not None else "Unavailable"})
                if shift is not None and abs(shift) >= 0.5:
                    flags.append(f"{metric}: {label} distribution shift of {shift:+.2f} development SD.")
    return {"coverage": coverage, "pregame": pregame, "baseline": baseline,
            "drift": drift, "flags": flags,
            "content_id": manifest["data_content_id"],
            "created_at_utc": manifest.get("created_at_utc"),
            "seasons": years}


def _printable(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_table(rows: list[dict]):
    if not rows:
        return "_No rows available._\n"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_printable(row.get(k)).replace("|", "\\|") for k in cols) + " |")
    return "\n".join(lines) + "\n"


def html_table(rows: list[dict]):
    if not rows:
        return "<p>No rows available.</p>"
    cols = list(rows[0])
    out = ["<table><thead><tr>" + "".join(f"<th>{html_escape(k)}</th>" for k in cols) + "</tr></thead><tbody>"]
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{html_escape(_printable(row.get(k)))}</td>" for k in cols) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def write_report(summary: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {**summary, "report_created_at_utc": created, "method": "Read-only descriptive audit of frozen historical export"}
    tables = [("Completed-game coverage", "coverage"),
              ("Pregame chronology and feature coverage", "pregame"),
              ("Historical scoring-average benchmark", "baseline"),
              ("Distribution stability", "drift")]
    status = "Review" if summary["flags"] else "No review flags from these checks"
    intro = (f"Dataset content ID: `{summary['content_id']}`  \n"
             f"Export created (UTC): `{summary['created_at_utc']}`  \n"
             f"Report created (UTC): `{created}`  \n"
             f"Audit result: **{status}**\n\n"
             "This is a retrospective DATA and baseline evaluation, not a trained model, a future-game forecast, or proof of predictive accuracy. "
             "The scoring benchmark uses averages from previous games; baseline MAE/RMSE/bias are copied from the exported audit, not independently recomputed here.\n\n")
    md = ["# Historical NCAA Men's Basketball Evaluation Report\n\n" + intro]
    for heading, key in tables:
        md.extend([f"## {heading}\n", markdown_table(summary[key]), "\n"])
    md.append("## Review notes\n")
    md.extend([f"- {entry}\n" for entry in summary["flags"]] or ["No flags under the stated descriptive thresholds. This is not a guarantee of full coverage or model reliability.\n"])
    md.append("\n## Interpretation and limitations\n")
    md.extend(["- All CSV SHA-256 checksums and the manifest content ID were verified before reporting.\n",
               "- D1 usable coverage counts only completed matchups between two D1 teams, with both valid box scores and possessions; all-game exceptions use a wider set of games.\n",
               "- Canceled and postponed records are not classified as missing completed games.\n",
               "- Historical adjusted team snapshots are end-of-day; this report checks only earlier-dated ratings stored in pregame features.\n",
               "- Stored coverage audits use current Division I flags, which may not match earlier-season membership.\n",
               "- Conference metadata and player availability may be incomplete or reflect current labels.\n",
               "- Distribution shifts and scoring errors are descriptive, not causal or a substitute for independent trained-model evaluation.\n",
               "- The final historical holdout should not be repeatedly tuned against.\n",
               "- Report source: immutable, verified historical dataset archive; the live Supabase database was not queried.\n"])
    (output_dir / "report.md").write_text("".join(md), encoding="utf-8")
    head = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NCAA MBB Historical Evaluation</title><style>body{font-family:system-ui,Arial,sans-serif;max-width:1100px;margin:32px auto;padding:0 18px;background:#fafafa;color:#1d2939}h1{margin-bottom:5px}p{line-height:1.6}.meta{color:#475467}.wrap{overflow:auto;background:white;border:1px solid #ddd;border-radius:8px;margin:14px 0 25px}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{border-bottom:1px solid #e7e7e7;padding:10px 12px;text-align:left}th{background:#f2f4f7}section{margin-top:32px}.review{background:#fff7ed;padding:14px 18px;border:1px solid #fed7aa;border-radius:8px}.ok{background:#f1f5f9;padding:14px 18px;border-radius:8px}li{margin-bottom:5px}@media print{body{background:white;margin:0}.wrap{break-inside:avoid}}</style></head><body>"""
    doc = [head, "<h1>NCAA MBB — Historical Evaluation</h1>",
           f"<p class='meta'>Frozen dataset ID: <code>{html_escape(summary['content_id'])}</code> · Created {html_escape(str(created))}</p>",
           "<p>Retrospective data quality and scoring-baseline analysis. D1 usable percentages use matched D1-only numerators and denominators. All-game exceptions are shown separately. Not a trained forecast model, future-game prediction, or betting tool.</p>",
           f"<div class={'review' if summary['flags'] else 'ok'}><strong>{html_escape(status)}</strong><p>Thresholds are diagnostics, not guarantees of completeness.</p></div>"]
    for heading, key in tables:
        doc.extend([f"<section><h2>{html_escape(heading)}</h2><div class='wrap'>", html_table(summary[key]), "</div></section>"])
    doc.append("<section><h2>Review notes</h2><ul>")
    doc.extend(f"<li>{html_escape(item)}</li>" for item in summary["flags"])
    if not summary["flags"]:
        doc.append("<li>No warnings under the defined checks; independent verification may still be needed.</li>")
    doc.extend(["</ul></section><section><h2>Limitations</h2><p>Checksums verified. Previous adjusted ratings must predate their games; end-of-day snapshots are not pregame records. Historical D1 and conference labels may differ. Historical scoring baseline is not model accuracy, and statistical shifts are not causal.</p></section></body></html>"])
    (output_dir / "report.html").write_text("".join(doc), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for heading, key in tables:
        del heading
        rows = summary[key]
        if rows:
            with (output_dir / f"{key}.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return summary


def generate(archive: Path, output_dir: Path):
    manifest = verified_archive(archive)
    coverage = list(rows_from_archive(archive, "historical_coverage_audit", manifest))
    baseline = list(rows_from_archive(archive, "historical_score_baseline_audit", manifest))
    features = list(rows_from_archive(archive, "pregame_team_features", manifest))
    d1_coverage = d1_coverage_from_archive(archive, manifest)
    result = report_tables(manifest, coverage, baseline, features, d1_coverage)
    write_report(result, output_dir)
    print(f"Verified dataset ID: {result['content_id']}")
    print(f"Seasons: {result['seasons'][0]}–{result['seasons'][-1]}")
    print(f"Review notes: {len(result['flags'])}")
    print(f"Open {output_dir / 'report.html'} for the consolidated evaluation")
    return result


def main():
    p = argparse.ArgumentParser(description="Build consolidated historical basketball data evaluation")
    p.add_argument("--archive", type=Path, required=True, help="Path to verified NCAA MBB dataset ZIP")
    p.add_argument("--output-dir", type=Path, default=Path("historical_reports"))
    args = p.parse_args()
    generate(args.archive, args.output_dir)


if __name__ == "__main__":
    main()
