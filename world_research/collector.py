import argparse
import json
import os
import re
import sqlite3
import time
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://api.vrchat.cloud/api/1"
SCHEMA = Path(__file__).with_name("schema.sql")
DEFAULT_DB = SCHEMA.parent.parent / "world-research.sqlite3"


def timestamp():
    return datetime.now(UTC).isoformat(timespec="microseconds")


def world_id(value):
    if not re.fullmatch(r"wrld_[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value, re.I):
        raise argparse.ArgumentTypeError("expected a VRChat world ID starting with wrld_")
    return value.lower()


def tag_search(value):
    tag, separator, size = value.rpartition(":")
    if not separator or not tag.strip() or not size.isascii() or not size.isdecimal():
        raise argparse.ArgumentTypeError("expected TAG:LIMIT, such as game:50")
    limit = int(size)
    if not 1 <= limit <= 100:
        raise argparse.ArgumentTypeError("tag search limit must be between 1 and 100")
    return tag.strip(), limit


class VRChatClient:
    def __init__(self, contact):
        self.user_agent = f"VRChatWorldResearch/0.1 {contact}"
        self.next_request_at = 0.0

    def get(self, path, **params):
        delay = self.next_request_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        query = f"?{urlencode(params)}" if params else ""
        request = Request(
            f"{API_URL}{path}{query}",
            headers={"User-Agent": self.user_agent, "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=20) as response:
                return json.load(response)
        finally:
            self.next_request_at = time.monotonic() + 2


def count(world, key, suppress_zero=False):
    value = world.get(key)
    if type(value) is not int or value < 0 or (suppress_zero and value == 0):
        return None
    return value


def save_world(db, query_id, world, rank, anonymous_detail):
    tags = world.get("tags")
    tags_json = json.dumps(tags, ensure_ascii=False) if isinstance(tags, list) else None
    publication = world.get("publicationDate")
    if publication in ("none", ""):
        publication = None
    db.execute(
        """INSERT INTO worlds VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET name = excluded.name,
               description = coalesce(excluded.description, worlds.description),
               tags_json = coalesce(excluded.tags_json, worlds.tags_json),
               published_at = coalesce(excluded.published_at, worlds.published_at)""",
        (world["id"], world["name"], world.get("description"), tags_json, publication),
    )
    db.execute(
        """INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(query_id, world_id) DO NOTHING""",
        (query_id, world["id"], rank, tags_json,
         count(world, "occupants", anonymous_detail),
         count(world, "publicOccupants", anonymous_detail),
         count(world, "visits", anonymous_detail),
         count(world, "favorites", anonymous_detail)),
    )


def import_active_snapshot(db, path):
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict) or snapshot.get("version") != 1 or snapshot.get("source") != "active":
        raise ValueError("expected a version 1 active-world snapshot")
    raw_time = snapshot.get("observedAt")
    if not isinstance(raw_time, str):
        raise ValueError("missing snapshot timestamp")
    observed_at = datetime.fromisoformat(raw_time)
    if observed_at.tzinfo is None:
        raise ValueError("snapshot timestamp must include a time zone")
    observed_at = observed_at.astimezone(UTC).isoformat(timespec="microseconds")
    worlds = snapshot.get("worlds")
    if not isinstance(worlds, list) or len(worlds) > 10:
        raise ValueError("expected up to ten public worlds")
    for world in worlds:
        if (not isinstance(world, dict) or not isinstance(world.get("name"), str)
                or not isinstance(world.get("id"), str) or world.get("releaseStatus") != "public"):
            raise ValueError("invalid public world in snapshot")
        world_id(world["id"])

    with db:
        db.execute("BEGIN IMMEDIATE")
        exists = db.execute(
            "SELECT 1 FROM collection_queries WHERE source = 'active' AND target = 'website' AND finished_at = ?",
            (observed_at,),
        ).fetchone()
        if exists:
            print("This snapshot is already in the database.")
            return
        run_id = db.execute(
            "INSERT INTO collection_runs(started_at, finished_at) VALUES (?, ?)",
            (observed_at, observed_at),
        ).lastrowid
        query_id = db.execute(
            """INSERT INTO collection_queries(run_id, source, target, finished_at, status)
               VALUES (?, 'active', 'website', ?, 'ok')""",
            (run_id, observed_at),
        ).lastrowid
        for rank, world in enumerate(worlds, 1):
            save_world(db, query_id, world, rank, anonymous_detail=False)
    print(f"Imported {len(worlds)} active worlds observed at {observed_at}.")


def record_query(db, client, run_id, source, target, path, **params):
    error = None
    status = "ok"
    worlds = []
    try:
        data = client.get(path, **params)
        if source == "watchlist":
            if not isinstance(data, dict) or data.get("id") != target:
                raise ValueError("response did not match the requested world")
            data = [data]
        if not isinstance(data, list) or any(
            not isinstance(world, dict)
            or not isinstance(world.get("id"), str)
            or not isinstance(world.get("name"), str)
            for world in data
        ):
            raise ValueError("unexpected world response")
        worlds = [(rank, world) for rank, world in enumerate(data, 1)
                  if world.get("releaseStatus") == "public"]
    except HTTPError as exc:
        status = f"http_{exc.code}"
        error = exc.reason
        exc.close()
    except (URLError, TimeoutError, ValueError) as exc:
        status = "error"
        error = str(exc)

    with db:
        query_id = db.execute(
            "INSERT INTO collection_queries VALUES (NULL, ?, ?, ?, ?, ?, ?)",
            (run_id, source, target, timestamp(), status, error),
        ).lastrowid
        for rank, world in worlds:
            detail = source == "watchlist"
            save_world(db, query_id, world, None if detail else rank, detail)
    label = f"{source} {target}" if target else source
    print(f"{label}: {len(worlds)} worlds ({status})", flush=True)
    return status


def scan(db, client, limit, tags, watch_only):
    queries = []
    if not watch_only:
        queries.extend([
            ("active", None, "/worlds/active", {"sort": "popularity", "n": limit}),
            ("published", None, "/worlds", {"sort": "publicationDate", "n": limit}),
        ])
        for tag, tag_limit in dict(tags).items():
            queries.append(("tag_search", tag, "/worlds", {"tag": tag, "n": tag_limit}))

    watched = db.execute(
        "SELECT world_id FROM watchlist WHERE review_on >= ? ORDER BY world_id",
        (datetime.now(UTC).date().isoformat(),),
    )
    for (target,) in watched:
        queries.append(("watchlist", target, f"/worlds/{target}", {}))

    if not queries:
        print("No active watchlist entries.")
        return 1

    with db:
        run_id = db.execute(
            "INSERT INTO collection_runs(started_at) VALUES (?)", (timestamp(),),
        ).lastrowid
    failed = False
    discovery_unauthorized = False
    try:
        for source, target, path, params in queries:
            if source != "watchlist" and discovery_unauthorized:
                continue
            status = record_query(db, client, run_id, source, target, path, **params)
            failed |= status != "ok"
            if status in {"http_401", "http_403"} and source != "watchlist":
                discovery_unauthorized = True
            if status == "http_429" or status.startswith("http_5"):
                print("Stopped this run after the API returned a server or rate-limit error.")
                break
    finally:
        with db:
            db.execute("UPDATE collection_runs SET finished_at = ? WHERE id = ?",
                       (timestamp(), run_id))
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description="Collect public VRChat world observations.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="create the SQLite database")
    snapshot = commands.add_parser("import-active", help="import an active-world snapshot from the website")
    snapshot.add_argument("snapshot", type=Path)
    watch = commands.add_parser("watch", help="add or renew a watchlist entry")
    watch.add_argument("world", type=world_id)
    watch.add_argument("--reason", required=True)
    watch.add_argument("--review-on", type=date.fromisoformat)
    collect = commands.add_parser("scan", help="run one collection pass")
    collect.add_argument("--limit", type=int, choices=range(1, 101), metavar="1-100", default=50)
    collect.add_argument("--tag", action="append", type=tag_search, default=[], metavar="TAG:LIMIT",
                         help="add a tag search with its own result limit, such as game:50")
    collect.add_argument("--watch-only", action="store_true")
    args = parser.parse_args()
    contact = os.getenv("VRCHAT_API_CONTACT", "").strip()
    if args.command == "scan" and not contact:
        parser.error("set VRCHAT_API_CONTACT to the app's contact email or project URL")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(args.db)) as db:
        db.executescript(SCHEMA.read_text(encoding="utf-8"))
        if args.command == "init":
            print(args.db.resolve())
        elif args.command == "import-active":
            try:
                import_active_snapshot(db, args.snapshot)
            except (OSError, ValueError, argparse.ArgumentTypeError) as exc:
                parser.error(str(exc))
        elif args.command == "watch":
            review_on = args.review_on or (datetime.now(UTC).date() + timedelta(days=28))
            with db:
                db.execute(
                    """INSERT INTO watchlist VALUES (?, ?, ?, ?)
                       ON CONFLICT(world_id) DO UPDATE SET reason = excluded.reason,
                           review_on = excluded.review_on""",
                    (args.world, args.reason, timestamp(), review_on.isoformat()),
                )
            print(f"Watching {args.world} through {review_on} UTC")
        else:
            return scan(db, VRChatClient(contact), args.limit, args.tag, args.watch_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
