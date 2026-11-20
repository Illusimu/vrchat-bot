"""Serve a read-only local research explorer, or export the current snapshot."""
import argparse
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from collector import DEFAULT_DB
from investigations import queue


TEMPLATE = Path(__file__).with_name("explorer.html")


def snapshot(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        runs = db.execute("""
            SELECT r.id,r.finished_at FROM collection_runs r WHERE EXISTS
            (SELECT 1 FROM collection_queries q JOIN observations o ON o.query_id=q.id
             WHERE q.run_id=r.id) ORDER BY r.id DESC LIMIT 12
        """).fetchall()
        ids = [r[0] for r in runs]
        placeholders = ",".join("?" for _ in ids) or "NULL"
        worlds = {}
        for wid, name, run, ts, players, visits, favs, tags in db.execute(f"""
            SELECT s.world_id,w.name,s.run_id,s.observed_at,s.occupants,s.visits,s.favorites,
                   o.tags_json FROM world_samples s JOIN worlds w ON w.id=s.world_id
            JOIN collection_queries q ON q.run_id=s.run_id AND q.source=s.source
                AND q.finished_at=s.observed_at AND q.target IS s.target
            JOIN observations o ON o.query_id=q.id AND o.world_id=s.world_id
            WHERE s.run_id IN ({placeholders}) ORDER BY s.run_id
        """, ids):
            tags = [t.removeprefix("author_tag_") for t in json.loads(tags or "[]")
                    if t.startswith("author_tag_")]
            worlds.setdefault(wid, {"name": name, "samples": {}})["samples"][run] = {
                "t": ts, "p": players, "v": visits, "f": favs, "tags": tags, "pools": []}
        for wid, run, pool in db.execute(f"""
            SELECT DISTINCT world_id,run_id,pool FROM world_research_observations
            WHERE run_id IN ({placeholders})
        """, ids):
            worlds[wid]["samples"][run]["pools"].append(pool)
        launches = {}
        for wid, name, phase, ts in db.execute("""
            SELECT e.world_id,coalesce(w.name,e.world_id),e.milestone,e.launched_at
            FROM launch_study_enrollment e LEFT JOIN worlds w ON w.id=e.world_id
            WHERE julianday(e.launched_at)>=julianday('now','-90 days')
            ORDER BY e.launched_at DESC
        """):
            launches[wid, phase, ts] = {"name": name, "id":wid, "phase":phase,
                                        "t":ts, "samples":[]}
        for wid, phase, ts, age, players, visits, favorites in db.execute("""
            SELECT world_id,milestone,launched_at,hours_since_launch,occupants,visits,favorites
            FROM launch_study_samples WHERE julianday(launched_at)>=julianday('now','-90 days')
            ORDER BY hours_since_launch
        """):
            launches[wid, phase, ts]["samples"].append([age, players, visits, favorites])
        dates = db.execute("""
            SELECT milestone,launched_at FROM world_launch_times
            WHERE julianday(launched_at) BETWEEN julianday('now','-90 days') AND julianday('now')
        """).fetchall()
        flags = queue(db, datetime.now(UTC))
    return dict(runs=runs, worlds=worlds, launches=list(launches.values()), dates=dates, flags=flags)


def fragment(path):
    data = json.dumps(snapshot(path), ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return TEMPLATE.read_text(encoding="utf-8").replace("__RESEARCH_DATA__", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--export-fragment", type=Path)
    args = parser.parse_args()
    if args.export_fragment:
        args.export_fragment.write_text(fragment(args.db), encoding="utf-8")
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            try:
                page = fragment(args.db)
            except (OSError, sqlite3.Error):
                self.send_error(503, "Unable to read the local research database")
                return
            head = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width, initial-scale=1">'
                    '<title>VRChat world research</title>')
            body = (head + "<body>" + page + "</body></html>").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"Local research explorer at http://127.0.0.1:{args.port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
