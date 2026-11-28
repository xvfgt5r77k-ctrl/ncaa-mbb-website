#!/usr/bin/env python3
"""Read-only, historical ACC availability archive inspection.

Only reads the public ACC *archive*, never a current-status feed. It produces
review candidates, not verified player availability or ESPN game matches.
It NEVER writes to Supabase and never derives injury/medical diagnoses.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ARCHIVE_URL = 'https://theacc.com/sports/2025/12/29/availability-reporting-mbb-archive.aspx'
ALLOWED_HOSTS = {'app.hdintelligence.com', 'par.hdintelligence-app.com'}
VALID_STATUSES = {
    'out': 'Out', 'questionable': 'Questionable', 'doubtful': 'Doubtful',
    'probable': 'Probable', 'available': 'Available',
    'game time decision': 'Game Time Decision',
    'game-time decision': 'Game Time Decision',
    'gtd': 'Game Time Decision',
}
CANDIDATE_FIELDS = ['source_page', 'source_frame', 'captured_at_utc',
                    'candidate_status', 'candidate_name_unverified',
                    'row_text', 'review_state']


def clean(value: object) -> str:
    return re.sub(r'\s+', ' ', str(value or '')).strip()


def is_historical_archive_frame(url: str) -> bool:
    """No data capture unless frame explicitly identifies historical archive."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or '').lower()
        query = {k.lower(): v for k, v in parse_qs(parsed.query).items()}
        return (parsed.scheme == 'https' and host in ALLOWED_HOSTS
                and any(x.lower() == 'archive' for x in query.get('type', [])))
    except Exception:
        return False


def candidate_from_cells(cells: list[str]) -> dict | None:
    """Only exact status cells qualify; no inference from prose or no-show DNP."""
    values = [clean(c) for c in cells if clean(c)]
    if len(values) < 2:
        return None
    status_positions = [i for i, v in enumerate(values) if v.lower() in VALID_STATUSES]
    if len(status_positions) != 1:
        return None
    position = status_positions[0]
    others = [v for i, v in enumerate(values) if i != position]
    candidates = [v for v in others if re.search(r'[A-Za-z]{2,}', v)
                  and v.lower() not in ('name', 'player', 'status', 'school', 'team',
                                        'availability', 'report')]
    if not candidates:
        return None
    # Label is explicitly unverified: a school name or matchup could appear here.
    return {'candidate_status': VALID_STATUSES[values[position].lower()],
            'candidate_name_unverified': candidates[0],
            'row_text': ' | '.join(values)}


async def evaluate_or_default(frame, js: str, fallback):
    try:
        return await frame.evaluate(js)
    except Exception:
        return fallback


async def scrape(output_dir: Path) -> dict:
    from playwright.async_api import async_playwright

    output_dir.mkdir(parents=True, exist_ok=True)
    captured = datetime.now(timezone.utc).isoformat()
    frame_summaries = []
    candidates = []
    raw_sections = []
    controls = []
    screenshot_error = None

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, args=['--no-sandbox'])
        context = await browser.new_context(viewport={'width': 1440, 'height': 1000})
        page = await context.new_page()
        page.set_default_timeout(10000)
        try:
            await page.goto(ARCHIVE_URL, wait_until='domcontentloaded', timeout=60000)
            # Browser-rendered iframe requires time; this is not a live data poll.
            await page.wait_for_timeout(6500)
            for _ in range(5):
                if any(is_historical_archive_frame(f.url) for f in page.frames):
                    break
                await page.wait_for_timeout(1500)
            for frame in page.frames:
                frame_url = frame.url
                archive = is_historical_archive_frame(frame_url)
                # Parent text is page framing only; only historical iframe is candidate data.
                if frame != page.main_frame and not archive:
                    continue
                body = await evaluate_or_default(
                    frame, "document.body ? document.body.innerText : ''", '')
                raw_sections.append(f'FRAME URL: {frame_url}\nARCHIVE DATA: {archive}\n' +
                                    str(body)[:120000] + '\n\n')
                rows = await evaluate_or_default(frame, """() => Array.from(
                    document.querySelectorAll('table tr, [role="row"]'))
                    .slice(0,3000).map(el => ({
                       cells: Array.from(el.querySelectorAll('th,td,[role="cell"],[role="columnheader"]'))
                         .map(x => (x.innerText || '').trim()),
                       text: (el.innerText || '').trim()
                    }))""", [])
                frame_summaries.append({'frame_url': frame_url, 'historical_archive': archive,
                                        'visible_chars': len(str(body)), 'table_rows': len(rows)})
                if not archive:
                    continue
                controls.extend(await evaluate_or_default(frame, """() => Array.from(
                    document.querySelectorAll('select, input, button'))
                    .slice(0,200).map(el => ({
                      tag: el.tagName, type: el.type || '',
                      label: (el.innerText || el.getAttribute('aria-label') || '').trim().slice(0,180),
                      value: String(el.value || '').slice(0,100),
                      options: el.tagName === 'SELECT' ? Array.from(el.options)
                           .slice(0,80).map(o => ({text:o.text,value:o.value})) : []
                    }))""", []))
                for row in rows:
                    parsed = candidate_from_cells(row.get('cells') or [])
                    if parsed:
                        candidates.append({
                            'source_page': ARCHIVE_URL,
                            'source_frame': frame_url,
                            'captured_at_utc': captured,
                            **parsed,
                            'review_state': 'unverified_do_not_import',
                        })
            try:
                await page.screenshot(path=str(output_dir / 'acc_archive_view.png'),
                                      full_page=True, timeout=25000)
            except Exception as exc:
                screenshot_error = str(exc)[:500]
        finally:
            await browser.close()

    with (output_dir / 'candidate_status_rows.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(candidates)
    (output_dir / 'archive_visible_text.txt').write_text(''.join(raw_sections), encoding='utf-8')
    (output_dir / 'archive_controls.json').write_text(json.dumps(controls, indent=2), encoding='utf-8')
    summary = {
        'source': ARCHIVE_URL, 'captured_at_utc': captured,
        'archive_iframe_found': any(x['historical_archive'] for x in frame_summaries),
        'candidate_rows_unverified': len(candidates),
        'verified_player_statuses': 0,
        'espn_game_ids_matched': 0,
        'frames': frame_summaries,
        'screenshot_error': screenshot_error,
        'meaning': 'Archive discovery only. Candidate rows are not official verified statuses, '
                   'are not matched to completed games, and MUST NOT be loaded directly to Supabase.',
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='acc_historical_archive_report')
    args = p.parse_args()
    report = asyncio.run(scrape(Path(args.output)))
    print(json.dumps(report, indent=2))
    # If archive iframe inaccessible, fail with actionable artifacts still uploaded.
    if not report['archive_iframe_found']:
        print('NOTICE: historical iframe not accessible; inspect archive_controls.json and screenshot.')


if __name__ == '__main__':
    main()
