import { WorkflowEntrypoint } from 'cloudflare:workers';
import type { WorkflowEvent, WorkflowStep } from 'cloudflare:workers';
import { HttpError } from './core.ts';
import type { Env } from './core.ts';
import type { SweepParameters } from './world-history.ts';
import { worldResearchDetails, worldResearchPage } from './vrchat.ts';
import type { ResearchPage, WorldDetailSnapshot } from './vrchat.ts';
import { trackLaunches } from './launch-cohort.ts';

const POOLS = ['active', 'games', 'social', 'published', 'labs'];
type PageResult = { status: 'ok'; data: ResearchPage }
  | { status: 'failed' | 'skipped'; pool: string; page: number; reason: string; code?: number };

export function shouldFetchSecond(first: ResearchPage, audit: boolean) {
  if (audit) return true;
  if (first.worlds.length < first.limit) return false;
  const tail = first.worlds.slice(-20).map(world => world.occupants);
  return tail.some(count => count === null) || tail.filter(count => count !== null && count >= 20).length >= 10;
}

export class WorldResearchSweep extends WorkflowEntrypoint<Env, SweepParameters> {
  async run(event: WorkflowEvent<SweepParameters>, step: WorkflowStep) {
    const { slot, delaySeconds, audit } = event.payload;
    if (!Number.isFinite(slot) || !Number.isFinite(delaySeconds) || delaySeconds < 0 || delaySeconds > 540) throw new Error('Invalid sweep parameters.');
    if (delaySeconds) await step.sleep('offset sweep start', delaySeconds * 1000);
    const startedAt = await step.do('record sweep start', async () => new Date().toISOString());
    const results: PageResult[] = [];
    let halted = false;
    let attempted = false;
    for (const pool of POOLS) {
      let first: ResearchPage | undefined;
      for (const page of pool === 'published' || pool === 'labs' ? [1] : [1, 2]) {
        if (halted || (page === 2 && first && !shouldFetchSecond(first, audit))) {
          results.push({status:'skipped', pool, page, reason:halted ? 'Previous request failed' : 'First page met the stopping rule'});
          continue;
        }
        let result: PageResult;
        for (let attempt = 1; ; attempt++) {
          if (attempted) await step.sleep(`space ${pool} page ${page} attempt ${attempt}`, 65_000);
          attempted = true;
          result = await step.do(`fetch ${pool} page ${page} attempt ${attempt}`, {retries:{limit:0, delay:'1 second'}}, async (): Promise<PageResult> => {
            try { return {status:'ok', data:await worldResearchPage(this.env, pool, String(page))}; }
            catch (error) {
              return {status:'failed', pool, page, code:error instanceof HttpError ? error.status : 500,
                      reason:error instanceof HttpError ? error.message : 'World request failed'};
            }
          });
          // The shared limiter can be busy when the site just refreshed a profile.
          if (result.status === 'ok' || result.code !== 429 || attempt === 3) break;
        }
        results.push(result);
        if (result.status === 'ok') first = result.data;
        else halted = true;
      }
    }
    const batchId = event.instanceId;
    const pages = results.flatMap(result => result.status === 'ok' ? [result.data] : []);
    const enrollment = await step.do('select launches for follow-up', async () => trackLaunches(this.env, pages, slot, batchId));
    const observed = new Set(pages.flatMap(p => p.worlds.filter(world => world.occupants !== null).map(world => world.id)));
    const followUps: ({worldId:string; status:'ok'; observedAt:string; world:ResearchPage['worlds'][number]}
      | {worldId:string; status:'failed'|'skipped'; reason:string; code?:number})[] = [];
    for (const worldId of new Set(enrollment.map(item => item.worldId))) {
      if (observed.has(worldId)) continue;
      if (halted) {
        followUps.push({worldId, status:'skipped', reason:'Previous upstream request failed'});
        continue;
      }
      let result;
      for (let attempt = 1; ; attempt++) {
        await step.sleep('space launch ' + worldId + ' attempt ' + attempt, 65_000);
        result = await step.do('sample launch ' + worldId + ' attempt ' + attempt,
          {retries:{limit:0, delay:'1 second'}}, async () => {
          try {
            const snapshot: WorldDetailSnapshot = await worldResearchDetails(this.env, worldId);
            const {description, authorName, authorId, version, ...world} = snapshot.world;
            return {worldId, status:'ok' as const, observedAt:snapshot.observedAt, world};
          } catch (error) {
            return {worldId, status:'failed' as const, code:error instanceof HttpError ? error.status : 500,
              reason:error instanceof HttpError ? error.message : 'Launch lookup failed'};
          }
        });
        if (result.status === 'ok' || result.code !== 429 || attempt === 3) break;
      }
      followUps.push(result);
      if (result.status==='failed' && (result.code===429 || result.code===503)) halted = true;
    }
    const finishedAt = await step.do('record sweep finish', async () => new Date().toISOString());
    await step.do('publish completed sweep', async () => {
      const db = this.env.WORLD_RESEARCH_BUFFER;
      if (!db) throw new Error('World history buffer binding is missing');
      const payload = JSON.stringify({version:1, batchId, slot, startedAt, finishedAt, results, launchStudy:{enrollment,followUps}});
      await db.batch([
        db.prepare('INSERT OR IGNORE INTO world_batches(batch_id,payload) VALUES (?,?)').bind(batchId, payload),
        db.prepare('DELETE FROM world_batches WHERE acknowledged_at <= ?').bind(Date.now() - 7 * 86400000),
      ]);
    });
    return {batchId, pages:results.filter(result => result.status === 'ok').length, halted, launchEnrollments:enrollment.length, launchLookups:followUps.length};
  }
}
