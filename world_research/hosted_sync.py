import argparse
import ctypes
import json
import logging
import os
import re
import sqlite3
import sys
from contextlib import closing
from ctypes import wintypes
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from collector import DEFAULT_DB, SCHEMA, save_world, timestamp, world_id
from investigations import add_candidates, refresh_details
from pilot_report import POOLS, page_target, store_page, utc_timestamp, validate_page, validate_world


DEFAULT_ORIGIN = "https://escapethroughreality.com"
TOKEN_PATH = SCHEMA.parent.parent / ".world-research-token"
MAX_BATCH_BYTES = 1_500_000
LOG = logging.getLogger("world-history")
SWEEP_PAGES = {(p, i) for p in ("active", "games", "social", "published") for i in range(1, POOLS[p][4] + 1)}


class DataBlob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.c_void_p)]


def load_token(path):
    protected = bytes.fromhex(path.read_text(encoding="ascii").strip())
    buffer = ctypes.create_string_buffer(protected)
    source = DataBlob(len(protected), ctypes.cast(buffer, ctypes.c_void_p))
    result = DataBlob()
    unprotect = ctypes.windll.crypt32.CryptUnprotectData
    unprotect.argtypes = [ctypes.POINTER(DataBlob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DataBlob)]
    unprotect.restype = wintypes.BOOL
    if not unprotect(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise OSError("the current Windows user could not unlock the world-sync token")
    free = ctypes.windll.kernel32.LocalFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = ctypes.c_void_p
    try:
        token = ctypes.string_at(result.data, result.size).decode("utf-16-le").rstrip("\0")
    finally:
        ctypes.memset(result.data, 0, result.size)
        free(result.data)
    if not re.fullmatch(r"[a-f0-9]{64}", token):
        raise ValueError("the protected world-sync token has an invalid format")
    return token


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HostedBuffer:
    def __init__(self, token, origin=None):
        origin = origin or os.environ.get("WORLD_RESEARCH_ORIGIN") or DEFAULT_ORIGIN
        location = urlsplit(origin)
        if (re.search(r"[\s\\]", origin) or location.scheme != "https" or not location.hostname
                or location.username is not None or location.path not in ("", "/")
                or location.query or location.fragment):
            raise ValueError("WORLD_RESEARCH_ORIGIN must be an HTTPS origin without a path")
        self.base_url = f"https://{location.netloc}/api/world-research"
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/json",
                        "User-Agent": "VRChatWorldResearchSync/1.0 (+https://github.com/Illusimu/vrchat-bot)"}
        self.opener = build_opener(NoRedirect())

    def details(self, world):
        query = urlencode({"id": world})
        request = Request(f"{self.base_url}/details?{query}", headers=self.headers)
        with self.opener.open(request, timeout=30) as response:
            content = response.read(256_001)
            if len(content) > 256_000:
                raise ValueError("world detail response exceeded the size limit")
            return json.loads(content)

    def next(self, after=None):
        query = f"?after={after}" if after is not None else ""
        request = Request(f"{self.base_url}/sync/next{query}", headers=self.headers)
        with self.opener.open(request, timeout=30) as response:
            if response.status == 204:
                return None
            if response.status != 200:
                raise ValueError("unexpected response status")
            sequence = int(response.headers["X-World-Batch-Sequence"])
            content = response.read(MAX_BATCH_BYTES + 1)
            if sequence < 1 or len(content) > MAX_BATCH_BYTES:
                raise ValueError("invalid hosted batch size or sequence")
            return sequence, json.loads(content)

    def acknowledge(self, sequence, batch_id):
        query = urlencode({"sequence": sequence, "id": batch_id})
        request = Request(f"{self.base_url}/sync/ack?{query}", data=b"", headers=self.headers, method="POST")
        with self.opener.open(request, timeout=30) as response:
            if response.status != 200:
                raise ValueError("hosted batch was not acknowledged")


def validate_batch(batch):
    if batch["version"] != 1 or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", batch["batchId"]):
        raise ValueError("invalid hosted batch version or ID")
    batch["startedAt"] = utc_timestamp(batch["startedAt"])
    batch["finishedAt"] = utc_timestamp(batch["finishedAt"])
    if batch["startedAt"] > batch["finishedAt"]:
        raise ValueError("invalid hosted batch time range")
    results = []
    seen = set()
    expected_pages = SWEEP_PAGES | {("labs", 1)} if "launchStudy" in batch else SWEEP_PAGES
    for entry in batch["results"]:
        status = entry["status"]
        if status == "ok":
            pool, page = validate_page(entry["data"])
        elif status in ("failed", "skipped") and isinstance(entry["reason"], str):
            pool, page = entry["pool"], entry["page"]
        else:
            raise ValueError("invalid hosted page result")
        if (pool, page) in seen or (pool, page) not in expected_pages:
            raise ValueError("unexpected or duplicate hosted page")
        seen.add((pool, page))
        results.append((pool, page, entry))
    if seen != expected_pages:
        raise ValueError("hosted sweep is missing page status information")
    return results


def record_metadata(db, data):
    for world in data["worlds"]:
        updated = world.get("updatedAt")
        updated = utc_timestamp(updated) if updated and updated != "none" else None
        platforms = world.get("platforms")
        if platforms is not None:
            platforms = json.dumps(sorted(set(platforms)), separators=(",", ":"))
        fields = (updated, platforms, world.get("capacity"))
        if fields == (None, None, None):
            continue
        previous = db.execute(
            """SELECT updated_at, platforms_json, capacity FROM world_metadata_history
               WHERE world_id = ? AND observed_at <= ? ORDER BY observed_at DESC LIMIT 1""",
            (world["id"], data["observedAt"]),
        ).fetchone()
        if previous != fields:
            db.execute("INSERT OR IGNORE INTO world_metadata_history VALUES (?, ?, ?, ?, ?)",
                       (world["id"], data["observedAt"], *fields))


def record_launch_context(db, query, data):
    observed = data["observedAt"]
    for world in data["worlds"]:
        for milestone, key in (("labs", "labsPublicationDate"), ("public", "publicationDate")):
            when = world.get(key)
            if not when or when == "none":
                continue
            db.execute("""INSERT INTO world_launch_times VALUES (?,?,?,?)
                          ON CONFLICT(world_id,milestone,launched_at) DO UPDATE
                          SET first_observed_at=excluded.first_observed_at
                          WHERE excluded.first_observed_at < world_launch_times.first_observed_at""",
                       (world["id"], milestone, utc_timestamp(when), observed))
        featured, creator, platforms = world.get("featured"), world.get("creatorId"), world.get("platforms")
        if any(field is not None for field in (featured, creator, platforms)):
            db.execute("INSERT OR IGNORE INTO launch_sample_context VALUES (?,?,?,?,?)",
                       (query, world["id"], creator, featured,
                        json.dumps(sorted(set(platforms))) if platforms is not None else None))


def store_launch_study(db, run, finished_at, study):
    targets = set()
    for item in study["enrollment"]:
        wid = world_id(item["worldId"])
        launched = utc_timestamp(item["launchedAt"])
        enrolled = utc_timestamp(item["enrolledAt"])
        if enrolled < launched:
            raise ValueError("study enrollment precedes launch")
        db.execute("INSERT OR IGNORE INTO launch_study_enrollment VALUES (?,?,?,?)",
                   (wid, item["milestone"], launched, enrolled))
        targets.add(wid)
    seen = set()
    for followup in study["followUps"]:
        wid, status = world_id(followup["worldId"]), followup["status"]
        if wid not in targets or wid in seen or status not in ("ok", "failed", "skipped"):
            raise ValueError("invalid launch follow-up")
        seen.add(wid)
        captured = utc_timestamp(followup["observedAt"]) if status == "ok" else finished_at
        query = db.execute("""INSERT INTO collection_queries(run_id,source,target,finished_at,status,error)
                              VALUES (?,'launch_followup',?,?,?,?)""",
                           (run, wid, captured, status, followup.get("reason"))).lastrowid
        if status == "ok":
            world = followup["world"]
            validate_world(world)
            if world["id"] != wid:
                raise ValueError("launch follow-up returned the wrong world")
            save_world(db, query, world, None, anonymous_detail=False)
            data = {"worlds": [world], "observedAt": captured}
            record_metadata(db, data)
            record_launch_context(db, query, data)


def import_batch(db, sequence, batch):
    results = validate_batch(batch)
    with db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM hosted_batches WHERE batch_id = ?", (batch["batchId"],)).fetchone():
            return False
        run = db.execute("INSERT INTO collection_runs(started_at, finished_at) VALUES (?, ?)",
                         (batch["startedAt"], batch["finishedAt"])).lastrowid
        for pool, page, entry in results:
            target = page_target("hosted-sweep", pool, page)
            if entry["status"] == "ok":
                query = store_page(db, run, entry["data"], target)
                record_metadata(db, entry["data"])
                if "launchStudy" in batch:
                    record_launch_context(db, query, entry["data"])
            else:
                db.execute("""INSERT INTO collection_queries(run_id, source, target, finished_at, status, error)
                              VALUES (?, ?, ?, ?, ?, ?)""",
                           (run, POOLS[pool][1], target, batch["finishedAt"], entry["status"], entry["reason"]))
        if "launchStudy" in batch:
            store_launch_study(db, run, batch["finishedAt"], batch["launchStudy"])
        db.execute("INSERT INTO hosted_batches VALUES (?, ?, ?, ?)", (batch["batchId"], sequence, run, timestamp()))
    return True


def sync(db, buffer, replay):
    previous = 0
    imported = 0
    while entry := buffer.next(previous if replay else None):
        sequence, batch = entry
        if sequence <= previous:
            raise ValueError("the hosted buffer returned an out-of-order batch")
        added = import_batch(db, sequence, batch)
        buffer.acknowledge(sequence, batch["batchId"])
        previous = sequence
        imported += int(added)
        counts = {status: sum(entry["status"] == status for entry in batch["results"])
                  for status in ("ok", "failed", "skipped")}
        LOG.info("%s batch %s (#%s), with %s successful, %s failed and %s skipped pages",
                 "Imported" if added else "Already imported", batch["batchId"], sequence,
                 counts["ok"], counts["failed"], counts["skipped"])
    return imported


def main():
    parser = argparse.ArgumentParser(description="Sync the hosted VRChat buffer into the local world history.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--origin", help="HTTPS origin of the research Worker")
    parser.add_argument("--replay", action="store_true", help="include recently acknowledged batches still hosted")
    parser.add_argument("--log", type=Path, help="write a rotating local log")
    parser.add_argument("--skip-investigations", action="store_true", help="only import hosted world scans")
    args = parser.parse_args()
    handlers = [RotatingFileHandler(args.log, maxBytes=250_000, backupCount=2, encoding="utf-8")] if args.log else []
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, handlers=handlers, format="%(asctime)s %(levelname)s %(message)s")
    try:
        buffer = HostedBuffer(load_token(TOKEN_PATH), origin=args.origin)
        args.db.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(args.db, timeout=30)) as db:
            db.executescript(SCHEMA.read_text(encoding="utf-8"))
            count = sync(db, buffer, args.replay)
            if not args.skip_investigations:
                now = datetime.now(UTC)
                added = add_candidates(db, now)
                if added:
                    LOG.info("Flagged %d worlds for investigation.", added)
                try:
                    world = refresh_details(db, buffer.details, now)
                    if world:
                        LOG.info("Checked investigation details for %s.", world)
                except HTTPError as error:
                    LOG.warning("Investigation detail request deferred after HTTP %s.", error.code)
                    error.close()
        LOG.info("Sync complete. %d new batches.", count)
        return 0
    except HTTPError as error:
        content_type = error.headers.get_content_type()
        if error.headers.get("cf-mitigated") == "challenge":
            LOG.error("Cloudflare challenged the sync request (HTTP %s)", error.code)
        else:
            LOG.error("Cloudflare sync request returned HTTP %s (%s)", error.code, content_type)
        error.close()
    except (OSError, URLError, ValueError, KeyError, TypeError, sqlite3.Error, argparse.ArgumentTypeError) as error:
        LOG.error("Sync stopped. %s", error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
