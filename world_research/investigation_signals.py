from calendar import day_name
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean, median
from zoneinfo import ZoneInfo


LOCAL_ZONE = ZoneInfo("America/Chicago")
LOOKBACK = timedelta(weeks=6)
ADMISSION_ORDER = ("promising_release", "recurring_gathering", "growing_established", "sustained_activity")


@dataclass(frozen=True)
class Sample:
    at: datetime
    occupants: int | None
    visits: int | None
    favorites: int | None


@dataclass(frozen=True)
class Reason:
    kind: str
    summary: str
    evidence: dict
    observed_at: datetime


def sustained(samples):
    measured = [s for s in samples if s.occupants is not None]
    hits = sum(s.occupants >= 20 for s in measured)
    if hits < 3 or hits * 2 < len(measured) or measured[-1].at - measured[0].at < timedelta(hours=20):
        return None
    average = mean(s.occupants for s in measured)
    return Reason("sustained_activity", f"At least 20 players in {hits} of {len(measured)} recent samples; mean {average:.1f}.",
                  {"samples_at_or_above_20": hits, "measured_samples": len(measured),
                   "mean_players": average}, measured[-1].at)


def local_slot(local):
    return local.weekday(), local.hour // 2 * 2


def recurring(samples, now):
    measured = [s for s in samples if s.occupants is not None]
    if not measured or measured[-1].at - measured[0].at < timedelta(weeks=2):
        return None
    typical = median(s.occupants for s in measured)
    floor = max(8, typical * 2, typical + 5)
    slots = defaultdict(dict)
    for sample in measured:
        if sample.occupants < floor:
            continue
        local = sample.at.astimezone(LOCAL_ZONE)
        week = local.date() - timedelta(days=local.weekday())
        by_week = slots[local_slot(local)]
        if week not in by_week or sample.occupants > by_week[week].occupants:
            by_week[week] = sample
    qualified = [(slot, list(weeks.values())) for slot, weeks in slots.items()
                 if len(weeks) >= 3 and max(s.at for s in weeks.values()) >= now - timedelta(weeks=2)]
    if not qualified:
        return None
    (day, hour), peaks = max(qualified, key=lambda item: (len(item[1]), median(s.occupants for s in item[1])))
    peak_median = median(s.occupants for s in peaks)
    window = f"{day_name[day]} {hour:02}:00–{hour + 2:02}:00 Central"
    return Reason("recurring_gathering", f"Busier on {window} in {len(peaks)} weeks; median {peak_median:g} at those peaks.",
                  {"time_zone": LOCAL_ZONE.key, "weekday": day, "start_hour": hour, "end_hour": hour + 2,
                   "peak_times_utc": [s.at.isoformat() for s in sorted(peaks, key=lambda s: s.at)],
                   "peak_median_players": peak_median, "overall_median_players": typical,
                   "activity_floor": floor}, max(s.at for s in peaks))


def established_growth(samples, published, now):
    if published is None or now - published < timedelta(days=30):
        return None
    recent, baseline = defaultdict(list), defaultdict(list)
    for sample in samples:
        if sample.occupants is None or sample.at < now - timedelta(days=21):
            continue
        local = sample.at.astimezone(LOCAL_ZONE)
        period = recent if sample.at >= now - timedelta(days=7) else baseline
        period[local_slot(local)].append(sample)
    matched = recent.keys() & baseline.keys()
    recent_samples = [s for slot in matched for s in recent[slot]]
    baseline_samples = [s for slot in matched for s in baseline[slot]]
    if (len(matched) < 4 or len(recent_samples) < 6 or len(baseline_samples) < 6
            or len({s.at.astimezone(LOCAL_ZONE).date() for s in recent_samples}) < 3):
        return None
    before = median(median(s.occupants for s in baseline[slot]) for slot in matched)
    after = median(median(s.occupants for s in recent[slot]) for slot in matched)
    if after < before + 8 or after < before * 1.5:
        return None
    first_higher = min(s.at for s in recent_samples if s.occupants >= max(before + 8, before * 1.5))
    return Reason("growing_established", f"Matched weekly time slots rose from a median of {before:g} to {after:g} players.",
                  {"prior_median_players": before, "recent_median_players": after, "matched_slots": len(matched),
                   "recent_samples": len(recent_samples), "prior_samples": len(baseline_samples),
                   "first_higher_sample_utc": first_higher.isoformat(), "time_zone": LOCAL_ZONE.key},
                  max(s.at for s in recent_samples))


RELEASE_METRICS = {"visits": 20, "favorites": 2, "occupants": 5}


def release_rates(samples):
    rates = {}
    for metric in RELEASE_METRICS:
        observed = [s for s in samples if getattr(s, metric) is not None]
        if len(observed) < 2 or observed[-1].at - observed[0].at < timedelta(hours=20):
            continue
        if metric == "occupants":
            # Compare medians so a single momentary crowd cannot qualify a launch.
            if len(observed) < 6:
                continue
            midpoint = len(observed) // 2
            early, late = observed[:midpoint], observed[midpoint:]
            rates[metric] = (median(s.occupants for s in late) - median(s.occupants for s in early), late[-1].at)
            continue
        days = (observed[-1].at - observed[0].at).total_seconds() / 86400
        rates[metric] = ((getattr(observed[-1], metric) - getattr(observed[0], metric)) / days, observed[-1].at)
    return rates


def promising_release(world, releases):
    age, rates = releases[world]
    evidence = {}
    times = []
    for metric, minimum in RELEASE_METRICS.items():
        if metric not in rates or rates[metric][0] < minimum:
            continue
        peers = sorted(other[metric][0] for peer, (peer_age, other) in releases.items()
                       if peer != world and abs(peer_age - age) <= 7 and metric in other)
        if len(peers) < 3:
            continue
        upper_quartile = peers[(3 * len(peers) - 1) // 4]
        value, observed = rates[metric]
        if value > upper_quartile:
            evidence[metric] = {"value": round(value, 2), "peer_upper_quartile": round(upper_quartile, 2),
                                "peers": len(peers), "unit": "players" if metric == "occupants" else "per day"}
            times.append(observed)
    if not evidence:
        return None
    labels = ["player-count growth" if metric == "occupants" else metric + " gained" for metric in evidence]
    return Reason("promising_release", "New release with above-peer " + ", ".join(labels) + ".",
                  {"age_days": age, "metrics": evidence}, max(times))


def detect_reasons(history, publication, now):
    recent_start = now - timedelta(days=7)
    recent = {world: [s for s in samples if s.at >= recent_start] for world, samples in history.items()}
    releases = {}
    for world, samples in recent.items():
        published = publication.get(world)
        if published and timedelta(days=7) <= now - published < timedelta(days=30):
            releases[world] = ((now - published).days, release_rates([s for s in samples if s.at >= published]))
    found = {}
    for world, samples in history.items():
        reasons = [promising_release(world, releases) if world in releases else None, recurring(samples, now),
                   established_growth(samples, publication.get(world), now), sustained(recent[world])]
        if matches := [reason for reason in reasons if reason is not None]:
            found[world] = matches
    return found
