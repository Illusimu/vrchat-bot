import argparse
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean

from collector import DEFAULT_DB, SCHEMA, world_id
from investigation_signals import ADMISSION_ORDER, LOOKBACK, Sample, detect_reasons


def time_text(value):
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def parse_time(value):
    if not value or value == "none":
        return None
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("timestamps must include a time zone")
    return result.astimezone(UTC)


def load_signals(db, now):
    start = time_text(now - LOOKBACK)
    history = defaultdict(list)
    rows = db.execute("""SELECT world_id,observed_at,occupants,visits,favorites FROM world_samples
                        WHERE run_id IN (SELECT id FROM collection_runs WHERE finished_at IS NULL OR finished_at>=?)
                          AND observed_at BETWEEN ? AND ?
                        ORDER BY observed_at,run_id""", (start, start, time_text(now)))
    for world, observed, players, visits, favorites in rows:
        history[world].append(Sample(parse_time(observed), players, visits, favorites))
    publication = {world: parse_time(date) for world, date in db.execute("SELECT id,published_at FROM worlds")
                   if world in history}
    reasons = detect_reasons(history, publication, now)
    cutoff = now - timedelta(days=7)
    priority = {world: mean([s.occupants for s in history[world] if s.at >= cutoff and s.occupants is not None] or [0])
                for world in reasons}
    return reasons, priority


def choose_candidates(reasons, priority, existing, capacity):
    ordered = sorted(reasons.keys() - existing.keys(), key=lambda wid: (-len(reasons[wid]), -priority[wid], wid))
    chosen = []
    for kind in ADMISSION_ORDER:
        candidate = next((wid for wid in ordered if wid not in chosen
                          and any(reason.kind == kind for reason in reasons[wid])), None)
        if candidate is not None:
            chosen.append(candidate)
    return (chosen + [wid for wid in ordered if wid not in chosen])[:capacity]


def reason_evidence(db, world, reason):
    evidence = dict(reason.evidence)
    if reason.kind == "growing_established":
        first_higher = parse_time(evidence["first_higher_sample_utc"])
        evidence["nearby_updates"] = [updated for (updated,) in db.execute("""
            SELECT DISTINCT updated_at FROM world_metadata_history
            WHERE world_id=? AND updated_at BETWEEN ? AND ? ORDER BY updated_at
        """, (world, time_text(first_higher - timedelta(days=7)), time_text(reason.observed_at)))]
    return json.dumps(evidence, ensure_ascii=False)


def add_candidates(db, now):
    reasons, priority = load_signals(db, now)
    today = time_text(now.replace(hour=0, minute=0, second=0, microsecond=0))
    with db:
        db.execute("BEGIN IMMEDIATE")
        admitted = db.execute("SELECT count(*) FROM investigation_flags WHERE origin='automatic' AND flagged_at >= ?",
                              (today,)).fetchone()[0]
        existing = dict(db.execute("SELECT world_id,dismissed FROM investigation_flags"))
        candidates = choose_candidates(reasons, priority, existing, max(0, 5 - admitted))
        db.executemany("""INSERT INTO investigation_flags(world_id,reason,origin,priority,flagged_at)
                          VALUES (?,?,'automatic',?,?)""", [
            (wid, reasons[wid][0].summary, priority[wid], time_text(now)) for wid in candidates
        ])
        active = {wid for wid, dismissed in existing.items() if not dismissed} | set(candidates)
        db.executemany("""INSERT INTO investigation_reasons VALUES (?,?,?,?,?,?)
                          ON CONFLICT (world_id,kind) DO UPDATE SET summary=excluded.summary,
                              evidence_json=excluded.evidence_json, observed_at=excluded.observed_at
                          WHERE excluded.observed_at >= investigation_reasons.observed_at AND (
                              excluded.observed_at <> investigation_reasons.observed_at OR
                              excluded.evidence_json <> investigation_reasons.evidence_json)""", [
            (wid, reason.kind, reason.summary, reason_evidence(db, wid, reason),
             time_text(now), time_text(reason.observed_at))
            for wid in active & reasons.keys() for reason in reasons[wid]
        ])
    return len(candidates)


def queue(db, now):
    flags = db.execute("""SELECT f.world_id,w.name,w.published_at,f.reason,f.origin,f.retry_after
                          FROM investigation_flags f JOIN worlds w ON w.id=f.world_id
                          WHERE f.dismissed=0 ORDER BY f.origin='manual' DESC,f.priority DESC,f.flagged_at,f.world_id""")
    result = []
    for wid, name, published, reason, origin, retry in flags:
        publication = parse_time(published)
        update = db.execute("SELECT max(updated_at) FROM world_metadata_history WHERE world_id=?", (wid,)).fetchone()[0]
        latest = db.execute("""SELECT id,version_at,classification,changed_fields_json FROM investigation_details
                               WHERE world_id=? ORDER BY version_at DESC LIMIT 1""", (wid,)).fetchone()
        last_visit = db.execute("SELECT max(visited_at) FROM investigation_visits WHERE world_id=?", (wid,)).fetchone()[0]
        major = db.execute("""SELECT id,review_reason FROM investigation_details
                              WHERE world_id=? AND classification='major' AND version_at>?
                              ORDER BY version_at DESC LIMIT 1""", (wid, last_visit or "")).fetchone()
        eligible = publication + timedelta(days=7) if publication else None
        revisit = parse_time(last_visit) + timedelta(days=30) if last_visit else None
        needs_details = not latest or bool(update and update > latest[1])
        due = bool(eligible and now >= eligible and needs_details and (not retry or now >= parse_time(retry)))
        if not publication:
            stage = "publication_unknown"
        elif now < eligible:
            stage = "waiting_for_release_age"
        elif needs_details:
            stage = "waiting_for_details"
        elif not last_visit:
            stage = "ready_to_explore"
        elif major:
            stage = "ready_to_revisit" if now >= revisit else "revisit_cooldown"
        elif latest[2] == "unclassified" and latest[1] > last_visit:
            stage = "review_update"
        else:
            stage = "explored"
        reasons = [dict(kind=kind, summary=summary, evidence=json.loads(evidence),
                        first_detected_at=first, observed_at=seen)
                   for kind, summary, evidence, first, seen in db.execute("""
                       SELECT kind,summary,evidence_json,first_detected_at,observed_at
                       FROM investigation_reasons WHERE world_id=? ORDER BY first_detected_at,kind""", (wid,))]
        if origin == "manual" or not reasons:
            reasons.insert(0, dict(kind="manual" if origin == "manual" else "sustained_activity", summary=reason))
        if major:
            reasons.append(dict(kind="major_update", summary=major[1], snapshot_id=major[0]))
        result.append(dict(world_id=wid, name=name, reason=reason, origin=origin, stage=stage,
                           detail_due=due, update_at=update, snapshot_id=latest[0] if latest else None,
                           classification=latest[2] if latest else None,
                           changed_fields=json.loads(latest[3]) if latest else [],
                           detail_eligible_at=time_text(eligible) if eligible else None, last_visit_at=last_visit,
                           revisit_eligible_at=time_text(revisit) if revisit else None,
                           revisit_reason=major[1] if major else None, reasons=reasons))
    return result


def save_details(db, candidate, data):
    wid = candidate["world_id"]
    world = data["world"]
    if world["id"] != wid or world["releaseStatus"] != "public":
        raise ValueError("detail response did not match the requested public world")
    captured = time_text(parse_time(data["observedAt"]))
    reported = parse_time(world.get("updatedAt"))
    expected = candidate["update_at"]
    if expected and reported and time_text(reported) < expected:
        raise ValueError("world detail is older than the latest observed update")
    version = time_text(reported) if reported else expected or captured
    info = {key: world.get(key) for key in ("name", "description", "authorName", "authorId", "tags",
                                           "platforms", "capacity", "publicationDate", "version")}
    for key in ("tags", "platforms"):
        if info[key] is not None:
            info[key] = sorted(set(info[key]))
    with db:
        db.execute("BEGIN IMMEDIATE")
        previous = db.execute("""SELECT details_json,version_at FROM investigation_details
                                 WHERE world_id=? ORDER BY version_at DESC LIMIT 1""", (wid,)).fetchone()
        if previous and version <= previous[1]:
            db.execute("UPDATE investigation_flags SET retry_after=NULL WHERE world_id=?", (wid,))
            return False
        old = json.loads(previous[0]) if previous else {}
        changed = [key for key in info if info[key] != old.get(key)] if previous else []
        db.execute("""INSERT INTO investigation_details(world_id,version_at,captured_at,details_json,
                            changed_fields_json,classification) VALUES (?,?,?,?,?,?)""",
                   (wid, version, captured, json.dumps(info, ensure_ascii=False), json.dumps(changed),
                    "unclassified" if previous else "baseline"))
        db.execute("UPDATE investigation_flags SET retry_after=NULL WHERE world_id=?", (wid,))
        db.execute("UPDATE worlds SET description=? WHERE id=?", (info["description"], wid))
    return True


def refresh_details(db, fetch, now):
    candidate = next((item for item in queue(db, now) if item["detail_due"]), None)
    if candidate is None:
        return None
    wid = candidate["world_id"]
    # Reserve the attempt before using a shared upstream slot. A crash leaves it retryable.
    with db:
        claimed = db.execute("""UPDATE investigation_flags SET retry_after=? WHERE world_id=? AND dismissed=0
                               AND (retry_after IS NULL OR retry_after<=?)""",
                             (time_text(now + timedelta(hours=2)), wid, time_text(now))).rowcount
    if claimed:
        save_details(db, candidate, fetch(wid))
        return wid
    return None


def main():
    parser = argparse.ArgumentParser(description="Manage the local VRChat investigation queue.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("queue", help="list current flags and eligibility as JSON")
    commands.add_parser("refresh", help="evaluate the activity rule and fetch one eligible detail")
    flag = commands.add_parser("flag")
    flag.add_argument("world", type=world_id)
    flag.add_argument("reason")
    dismiss = commands.add_parser("dismiss")
    dismiss.add_argument("world", type=world_id)
    review = commands.add_parser("review")
    review.add_argument("snapshot", type=int)
    review.add_argument("classification", choices=("major", "minor", "unclassified"))
    review.add_argument("reason")
    visit = commands.add_parser("visit")
    visit.add_argument("world", type=world_id)
    visit.add_argument("notes")
    visit.add_argument("--at", type=parse_time)
    details = commands.add_parser("details")
    details.add_argument("world", type=world_id)
    args = parser.parse_args()
    now = datetime.now(UTC)
    with closing(sqlite3.connect(args.db, timeout=30)) as db:
        db.executescript(SCHEMA.read_text(encoding="utf-8"))
        with db:
            if args.command == "queue":
                print(json.dumps(queue(db, now), ensure_ascii=False, indent=2))
            elif args.command == "refresh":
                from hosted_sync import HostedBuffer, TOKEN_PATH, load_token
                added = add_candidates(db, now)
                saved = refresh_details(db, HostedBuffer(load_token(TOKEN_PATH)).details, now)
                print(json.dumps({"new_flags": added, "detail_world": saved}))
            elif args.command == "flag":
                if not db.execute("SELECT 1 FROM worlds WHERE id=?", (args.world,)).fetchone():
                    parser.error("world must be present in the local database")
                db.execute("""INSERT INTO investigation_flags(world_id,reason,origin,priority,flagged_at)
                              VALUES (?,?,'manual',0,?) ON CONFLICT(world_id) DO UPDATE SET
                              reason=excluded.reason, origin='manual', dismissed=0""",
                           (args.world, args.reason, time_text(now)))
            elif args.command == "dismiss":
                db.execute("UPDATE investigation_flags SET dismissed=1 WHERE world_id=?", (args.world,))
            elif args.command == "review":
                reviewed = db.execute("""UPDATE investigation_details SET classification=?,review_reason=?
                                        WHERE id=? AND classification<>'baseline'""",
                                      (args.classification, args.reason, args.snapshot)).rowcount
                if not reviewed:
                    parser.error("snapshot must reference a recorded update")
            elif args.command == "visit":
                if args.at and args.at > now:
                    parser.error("visit time cannot be in the future")
                if not db.execute("SELECT 1 FROM investigation_flags WHERE world_id=?", (args.world,)).fetchone():
                    parser.error("flag the world before recording a visit")
                db.execute("INSERT INTO investigation_visits(world_id,visited_at,notes) VALUES (?,?,?)",
                           (args.world, time_text(args.at or now), args.notes))
            else:
                db.row_factory = sqlite3.Row
                snapshots = db.execute("SELECT * FROM investigation_details WHERE world_id=? ORDER BY version_at DESC",
                                       (args.world,)).fetchall()
                visits = db.execute("SELECT visited_at,notes FROM investigation_visits WHERE world_id=? ORDER BY visited_at DESC",
                                    (args.world,)).fetchall()
                history = [dict(row) for row in snapshots]
                for item in history:
                    item["details"] = json.loads(item.pop("details_json"))
                    item["changed_fields"] = json.loads(item.pop("changed_fields_json"))
                print(json.dumps({"snapshots": history, "visits": [dict(row) for row in visits]},
                                 ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
