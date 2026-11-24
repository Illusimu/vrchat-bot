import type { Env } from './core.ts';
import type { ResearchPage } from './vrchat.ts';

const HOUR = 3600000;
const DAY = 24 * HOUR;
type Milestone = 'labs' | 'public';
export type Enrollment = {worldId:string; milestone:Milestone; launchedAt:string; enrolledAt:string};

export function recentLaunches(pages: ResearchPage[], now: number, slot: number) {
  const preference = (Math.floor(slot / DAY) + Math.floor(slot % DAY / (2 * HOUR))) % 2 ? 'labs' : 'public';
  const candidates: {worldId:string; milestone:Milestone; launched:number}[] = [];
  for (const page of pages) {
    const milestone = page.pool === 'labs' ? 'labs' : page.pool === 'published' ? 'public' : null;
    if (!milestone) continue;
    for (const world of page.worlds) {
      const launched = Date.parse((milestone === 'labs' ? world.labsPublicationDate : world.publicationDate) ?? '');
      if (Number.isFinite(launched) && (now - launched) >= 0 && (now - launched) <= 4 * HOUR) {
        candidates.push({worldId:world.id, milestone, launched});
      }
    }
  }
  return candidates.sort((a,b) => Number(b.milestone === preference) - Number(a.milestone === preference)
                        || b.launched - a.launched || a.worldId.localeCompare(b.worldId));
}

export async function trackLaunches(env: Env, pages: ResearchPage[], slot: number, sweepId: string): Promise<Enrollment[]> {
  const db = env.WORLD_RESEARCH_BUFFER;
  if (!db) throw new Error('World history buffer binding is missing');
  const now = Date.now();
  const candidates = recentLaunches(pages, now, slot);
  // Keep one admission per sweep, including Workflow retries. Limit lookup work if manual sweeps overlap.
  const available = await db.prepare('SELECT world_id,milestone,launched_at FROM launch_cohort WHERE launched_at>=?')
    .bind(now - 50 * HOUR).all<{world_id:string; milestone:Milestone; launched_at:number}>();
  const existing = new Set(available.results.map(row => `${row.world_id}:${row.milestone}:${row.launched_at}`));
  const next = candidates.find(candidate => !existing.has(`${candidate.worldId}:${candidate.milestone}:${candidate.launched}`));
  if (next) {
    await db.prepare(`INSERT OR IGNORE INTO launch_cohort(world_id,milestone,launched_at,enrolled_at,sweep_id)
      SELECT ?,?,?,?,? WHERE (SELECT count(*) FROM launch_cohort WHERE launched_at>=?)<26`)
      .bind(next.worldId,next.milestone,next.launched,now,sweepId,now-50*HOUR).run();
  }
  const rows = await db.prepare(`SELECT world_id,milestone,launched_at,enrolled_at FROM launch_cohort
    WHERE launched_at>=? ORDER BY launched_at,world_id LIMIT 26`).bind(now - 50 * HOUR)
    .all<{world_id:string; milestone:Milestone; launched_at:number; enrolled_at:number}>();
  const checkpoints = [7,14,28].flatMap(days => [now - days * DAY - 4 * HOUR, now - days * DAY]);
  const older = await db.prepare(`SELECT world_id,milestone,launched_at,enrolled_at FROM launch_cohort
    WHERE (launched_at BETWEEN ? AND ?) OR (launched_at BETWEEN ? AND ?) OR (launched_at BETWEEN ? AND ?)
    ORDER BY launched_at,world_id LIMIT 8`).bind(...checkpoints)
    .all<{world_id:string; milestone:Milestone; launched_at:number; enrolled_at:number}>();
  return [...rows.results, ...older.results].map(row => ({worldId:row.world_id,milestone:row.milestone,
    launchedAt:new Date(row.launched_at).toISOString(),enrolledAt:new Date(row.enrolled_at).toISOString()}));
}
