import { HttpError, json, now } from './core.ts';
import type { Env } from './core.ts';

export type SweepParameters = { slot: number; delaySeconds: number; audit: boolean };

export async function queueWorldSweep(env: Env, scheduledTime: number) {
  const interval = 2 * 60 * 60 * 1000;
  if (env.WORLD_RESEARCH_SCHEDULE !== 'enabled' || !env.WORLD_RESEARCH_SWEEPS) return;
  const slot = Math.floor(scheduledTime / interval) * interval;
  const id = `sweep-${slot}`;
  const params = { slot, delaySeconds: 30 + Math.floor(Math.random() * 510), audit: slot % 86400000 === 0 };
  // Every quarter-hour tick can queue its two-hour window. Workflow IDs deduplicate retries.
  try { await env.WORLD_RESEARCH_SWEEPS.create({ id, params }); }
  catch (error) {
    if (!(error instanceof Error && /already exists|duplicate/i.test(error.message))) throw error;
  }
}

export async function nextWorldBatch(env: Env, after: string | null) {
  const db = env.WORLD_RESEARCH_BUFFER;
  if (!db) throw new HttpError(503, 'World history buffer is unavailable.');
  if (after !== null && (!/^\d{1,15}$/.test(after) || !Number.isSafeInteger(Number(after)))) throw new HttpError(400, 'Invalid batch cursor.');
  const row = after === null
    ? await db.prepare('SELECT sequence, payload FROM world_batches WHERE acknowledged_at IS NULL ORDER BY sequence LIMIT 1').first<{sequence:number; payload:string}>()
    : await db.prepare('SELECT sequence, payload FROM world_batches WHERE sequence > ? ORDER BY sequence LIMIT 1').bind(Number(after)).first<{sequence:number; payload:string}>();
  if (!row) return new Response(null, {status:204});
  return new Response(row.payload, {headers:{'Content-Type':'application/json; charset=utf-8', 'X-World-Batch-Sequence':String(row.sequence)}});
}

export async function acknowledgeWorldBatch(env: Env, sequence: string | null, batchId: string | null) {
  const db = env.WORLD_RESEARCH_BUFFER;
  if (!db) throw new HttpError(503, 'World history buffer is unavailable.');
  if (!sequence || !/^[1-9]\d{0,14}$/.test(sequence) || !batchId || !/^[A-Za-z0-9_-]{1,100}$/.test(batchId)) {
    throw new HttpError(400, 'Invalid world batch identity.');
  }
  // Preserve the original acknowledgement time on retries.
  const row = await db.prepare('UPDATE world_batches SET acknowledged_at = coalesce(acknowledged_at, ?) WHERE sequence = ? AND batch_id = ? RETURNING sequence')
    .bind(now(), Number(sequence), batchId).first<{sequence:number}>();
  if (!row) throw new HttpError(404, 'World batch was not found.');
  return json({acknowledged:true});
}
