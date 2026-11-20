import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from collector import DEFAULT_DB, SCHEMA, world_id
from pilot_report import utc_timestamp


def coverage(db):
    db.row_factory = sqlite3.Row
    launches = db.execute("""SELECT milestone,count(*) AS dates_observed,count(DISTINCT world_id) AS worlds
                             FROM world_launch_times GROUP BY milestone""").fetchall()
    study = db.execute("""SELECT milestone,count(*) AS enrolled,
                                 count(CASE WHEN first_measured_hours<=2 THEN 1 END) AS measured_in_two_hours,
                                 count(CASE WHEN day_two_samples>0 THEN 1 END) AS measured_on_day_two
                          FROM launch_study_outcomes GROUP BY milestone""").fetchall()
    sources = db.execute("""SELECT source,status,count(*) AS queries,min(finished_at) AS first_at,max(finished_at) AS last_at
                            FROM collection_queries GROUP BY source,status""").fetchall()
    checkpoints = db.execute("""SELECT milestone,
       count(DISTINCT CASE WHEN hours_since_launch BETWEEN 168 AND 172 THEN world_id END) AS day_7,
       count(DISTINCT CASE WHEN hours_since_launch BETWEEN 336 AND 340 THEN world_id END) AS day_14,
       count(DISTINCT CASE WHEN hours_since_launch BETWEEN 672 AND 676 THEN world_id END) AS day_28
       FROM launch_longitudinal_samples WHERE occupants IS NOT NULL GROUP BY milestone""").fetchall()
    return {"launch_timestamps": [dict(row) for row in launches], "launch_study": [dict(row) for row in study],
            "checkpoints": [dict(row) for row in checkpoints], "queries": [dict(row) for row in sources]}


def main():
    parser = argparse.ArgumentParser(description="Inspect coverage and record known events for VRChat research.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("coverage")
    event = commands.add_parser("event")
    event.add_argument("world", type=world_id)
    event.add_argument("when", type=utc_timestamp)
    event.add_argument("kind", choices=("announcement", "opening_event", "feature", "update", "incident", "other"))
    event.add_argument("notes")
    args = parser.parse_args()
    with closing(sqlite3.connect(args.db, timeout=30)) as db:
        db.executescript(SCHEMA.read_text(encoding="utf-8"))
        if args.command == "coverage":
            print(json.dumps(coverage(db), indent=2))
        else:
            with db:
                if not db.execute("SELECT 1 FROM worlds WHERE id=?", (args.world,)).fetchone():
                    parser.error("world must be present in the local database")
                db.execute("INSERT INTO world_context_events(world_id,event_at,kind,notes) VALUES (?,?,?,?)",
                           (args.world, args.when, args.kind, args.notes))


if __name__ == "__main__":
    main()
