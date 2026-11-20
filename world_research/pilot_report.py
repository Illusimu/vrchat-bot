import argparse
import json
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

from collector import DEFAULT_DB, SCHEMA, save_world, world_id


POOLS = {
    "active": ("Active worlds", "active", "popularity", 100, 2),
    "games": ("Games", "tag_search", "popularity", 100, 2),
    "social": ("Social tag", "tag_search", "popularity", 100, 2),
    "hangouts": ("Social and hangout tags", "tag_search", "popularity", 100, 2),
    "published": ("Recently published", "published", "publicationDate", 50, 1),
    "labs": ("Community Labs", "labs", "labsPublicationDate", 50, 1),
}
GAME_TAG = {"author_tag_game"}
SOCIAL_TAG = {"author_tag_social"}
HANGOUT_TAG = {"author_tag_hangout"}
REQUIRED_TAGS = {"games": GAME_TAG, "social": SOCIAL_TAG, "hangouts": SOCIAL_TAG | HANGOUT_TAG}


def utc_timestamp(value):
    observed = datetime.fromisoformat(value)
    if observed.tzinfo is None:
        raise ValueError("each observation needs a timezone-aware timestamp")
    return observed.astimezone(UTC).isoformat(timespec="microseconds")


def validate_world(world):
    world["id"] = world_id(world["id"])
    if world["releaseStatus"] != "public" or not isinstance(world["name"], str):
        raise ValueError("expected public world metadata")
    for key in ("occupants", "visits", "favorites", "capacity"):
        value = world.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"invalid {key} count")


def validate_page(data):
    pool, page = data["pool"], data["page"]
    _, _, sort, limit, max_pages = POOLS[pool]
    if type(page) is not int or page not in range(1, max_pages + 1) or data["version"] != 2:
        raise ValueError(f"invalid {pool} page number or version")
    if data["sort"] != sort or data["limit"] != limit or data["offset"] != (page - 1) * limit:
        raise ValueError(f"invalid {pool} request metadata")
    if not isinstance(data["worlds"], list) or len(data["worlds"]) > limit:
        raise ValueError(f"invalid {pool} world list")
    data["observedAt"] = utc_timestamp(data["observedAt"])
    for world in data["worlds"]:
        validate_world(world)
    return pool, page


def read_pages(paths):
    pages = {}
    lines = (line for path in paths for line in path.read_text(encoding="utf-8-sig").splitlines())
    for line in lines:
        data = json.loads(line)
        pool, page = validate_page(data)
        if (pool, page) in pages:
            raise ValueError(f"duplicate {pool} page")
        pages[pool, page] = data
    if not pages:
        raise ValueError("the input contains no completed pages")
    return pages


def page_target(kind, pool, page):
    _, _, sort, limit, _ = POOLS[pool]
    return json.dumps({"kind": kind, "pool": pool, "page": page, "sort": sort,
                       "offset": (page - 1) * limit, "limit": limit}, separators=(",", ":"))


def store_page(db, run_id, data, target):
    query_id = db.execute(
        """INSERT INTO collection_queries(run_id, source, target, finished_at, status)
           VALUES (?, ?, ?, ?, 'ok')""", (run_id, POOLS[data["pool"]][1], target, data["observedAt"]),
    ).lastrowid
    for rank, world in enumerate(data["worlds"], data["offset"] + 1):
        save_world(db, query_id, world, rank, anonymous_detail=False)
    return query_id


def store_pilot(db, pages):
    snapshots = sorted(pages.values(), key=lambda data: data["observedAt"])
    started_at = snapshots[0]["observedAt"]
    finished_at = snapshots[-1]["observedAt"]
    with db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("""SELECT r.id FROM collection_runs r JOIN collection_queries q ON q.run_id = r.id
                                 WHERE r.started_at = ? AND q.target LIKE '{"kind":"world-pilot",%' LIMIT 1""",
                              (started_at,)).fetchone()
        run_id = existing[0] if existing else db.execute(
            "INSERT INTO collection_runs(started_at, finished_at) VALUES (?, ?)",
            (started_at, finished_at),
        ).lastrowid
        for data in snapshots:
            pool, page = data["pool"], data["page"]
            query = page_target("world-pilot", pool, page)
            if db.execute("SELECT 1 FROM collection_queries WHERE run_id = ? AND target = ?", (run_id, query)).fetchone():
                continue
            store_page(db, run_id, data, query)
        db.execute("UPDATE collection_runs SET finished_at = max(finished_at, ?) WHERE id = ?", (finished_at, run_id))
    return run_id


def distribution(worlds):
    ranked = sorted((w for w in worlds.values() if w.get("occupants") is not None),
                    key=lambda w: (-w["occupants"], w["id"]))
    counts = [w["occupants"] for w in ranked]
    total = sum(counts)
    bands = []
    for band in range(10):
        start = (band * len(counts) + 9) // 10
        end = ((band + 1) * len(counts) + 9) // 10
        players = sum(counts[start:end])
        bands.append([f"{band * 10}–{(band + 1) * 10}%", players / total if total and end > start else None,
                      players, total, end - start])
    return {
        "worlds": len(worlds), "measured": len(counts), "players": total,
        "mean": mean(counts) if counts else None, "median": median(counts) if counts else None,
        "peak": [ranked[0]["name"], counts[0]] if counts else None,
        "top10": sum(counts[:10]) / total if total else None, "bands": bands,
        "ranked": [[index, w["occupants"], w["name"]] for index, w in enumerate(ranked, 1)],
    }


def summarize(pages, floor):
    membership = {pool: {} for pool in POOLS}
    combined = {}
    for pool, page in sorted(pages, key=lambda item: (list(POOLS).index(item[0]), item[1])):
        for world in pages[pool, page]["worlds"]:
            for group in (membership[pool], combined):
                previous = group.setdefault(world["id"], world)
                if previous.get("occupants") is None and world.get("occupants") is not None:
                    group[world["id"]] = world
    appearances = Counter(w for worlds in membership.values() for w in worlds)
    fetched = sum(len(data["worlds"]) for data in pages.values())
    cutoffs = {}
    for pool in POOLS:
        first, second = pages.get((pool, 1)), pages.get((pool, 2))
        if not first or not second:
            continue
        tail = first["worlds"][-20:]
        known = [w["occupants"] for w in tail if w.get("occupants") is not None]
        first_ids = {w["id"] for w in first["worlds"]}
        new = {w["id"]: w for w in second["worlds"] if w["id"] not in first_ids}
        qualifying = [w["occupants"] for w in new.values() if (w.get("occupants") or 0) >= floor]
        above = sum(value >= floor for value in known)
        would_stop = len(first["worlds"]) < first["limit"] or (len(known) == 20 and above < 10)
        cutoffs[pool] = {"tail_size": len(tail), "tail_known": len(known), "tail_above": above,
                         "second_new": len(new), "second_above": len(qualifying),
                         "second_peak": max(qualifying, default=None),
                         "would_stop": would_stop}
    return {
        "pages": len(pages), "fetched": fetched, "unique": len(combined),
        "first_at": min(p["observedAt"] for p in pages.values()),
        "last_at": max(p["observedAt"] for p in pages.values()),
        "multiple_pools": sum(n > 1 for n in appearances.values()),
        "games_and_hangouts": len(membership["games"].keys() & membership["hangouts"].keys()),
        "games_and_social": len(membership["games"].keys() & membership["social"].keys()),
        "game_and_social_tags": sum((GAME_TAG | SOCIAL_TAG) <= set(w.get("tags") or []) for w in combined.values()),
        "floor": floor, "cutoffs": cutoffs,
        "tag_mismatches": {pool: sum(not (tags <= set(world.get("tags") or [])) for world in membership[pool].values())
                           for pool, tags in REQUIRED_TAGS.items()},
        "stats": {"all": distribution(combined), **{p: distribution(w) for p, w in membership.items() if w}},
    }


def write_chart(path, report):
    pools = {}
    payloads = []
    for pool, data in report["stats"].items():
        label = "All sampled worlds" if pool == "all" else POOLS[pool][0]
        pools[pool] = {**{key: data[key] for key in ("measured", "mean", "median", "peak", "top10")}, "label": label}
        chart = {"charts": [
            {"id": "vrchat-rank", "title": "Players by world rank", "height": 245,
             "spec": {"type": "line", "x": "rank", "y": "players", "xLabel": "Rank by player count",
                      "yLabel": "Players", "tooltipFields": [{"field": "world", "label": "World"}]},
             "columns": ["rank", "players", "world"], "rows": data["ranked"]},
            {"id": "vrchat-shares", "title": "Player share by rank band", "height": 245,
             "description": f"{data['players']:,} players summed across this sample",
             "spec": {"type": "bar", "x": "band", "y": "share", "xLabel": "Percent of worlds, busiest first",
                      "yLabel": "Share of players", "valueFormat": {"style": "percent"},
                      "tooltipFields": [{"field": "players", "label": "Players in band"},
                                        {"field": "total", "label": "Players in pool"},
                                        {"field": "worlds", "label": "Worlds in band"}]},
             "columns": ["band", "share", "players", "total", "worlds"], "rows": data["bands"]},
        ]}
        payloads.append(f'<script type="application/json" data-viz-chart data-pool="{pool}">'
                        + json.dumps(chart, ensure_ascii=False).replace("<", "\\u003c") + "</script>")
    summary = json.dumps({"unique": report["unique"], "multiplePools": report["multiple_pools"], "pools": pools},
                         ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("pilot-chart.html").read_text(encoding="utf-8")
    path.write_text(template.replace("<!-- CHART_DATA -->", "\n  ".join(payloads))
                    .replace("SUMMARY_DATA", summary), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Import and summarize a world paging pilot.")
    parser.add_argument("pages", type=Path, nargs="+")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--floor", type=int, default=20)
    parser.add_argument("--chart", type=Path)
    parser.add_argument("--summary", action="store_true", help="omit individual rank rows from console output")
    args = parser.parse_args()
    if args.floor < 1:
        parser.error("activity floor must be positive")
    pages = read_pages(args.pages)
    report = summarize(pages, args.floor)
    with closing(sqlite3.connect(args.db)) as db:
        db.executescript(SCHEMA.read_text(encoding="utf-8"))
        report["run_id"] = store_pilot(db, pages)
    if args.chart:
        write_chart(args.chart, report)
    if args.summary:
        for data in report["stats"].values():
            del data["ranked"]
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
