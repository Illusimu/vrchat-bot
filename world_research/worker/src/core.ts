export interface Env {
  DB: D1Database;
  VRCHAT_AUTH_COOKIE?: string;
  VRCHAT_COOKIE_KEY?: string;
  VRCHAT_USER_AGENT: string;
  WORLD_RESEARCH_TOKEN?: string;
  WORLD_RESEARCH_BUFFER?: D1Database;
  WORLD_RESEARCH_SWEEPS?: Workflow<import('./world-history.ts').SweepParameters>;
  WORLD_RESEARCH_SCHEDULE?: string;
}

export class HttpError extends Error {
  status: number;
  constructor(status: number, message: string) { super(message); this.status = status; }
}
export const now = () => Date.now();
export async function hash(value: string) {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value)))].map(n => n.toString(16).padStart(2, '0')).join('');
}
export function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json; charset=utf-8' } });
}

export async function rateLimit(env: Env, key: string, max: number, periodMs: number) {
  const time = now();
  const result = await env.DB.prepare(`INSERT INTO rate_limits(key, count, expires_at) VALUES (?, 1, ?)
    ON CONFLICT(key) DO UPDATE SET count = CASE WHEN expires_at <= ? THEN 1 ELSE count + 1 END,
    expires_at = CASE WHEN expires_at <= ? THEN excluded.expires_at ELSE expires_at END RETURNING count`).bind(key, time + periodMs, time, time).first<{count: number}>();
  if (!result || result.count > max) throw new HttpError(429, 'Please wait a moment before trying again.');
}

export async function upstream(url: string, init: RequestInit = {}) {
  try {
    const response = await fetch(url, { ...init, redirect: 'manual', signal: AbortSignal.timeout(12_000) });
    // Never forward credentials to a redirected destination.
    if (response.status >= 300 && response.status < 400) throw new Error('Unexpected upstream redirect');
    return response;
  }
  catch (error) {
    // Log only a fixed classification, never URLs, headers, bodies, or credentials.
    const detail = error instanceof Error ? error.message : '';
    const category = /redirect/i.test(detail) ? 'redirect' : /timeout|abort/i.test(detail) ? 'timeout' : /signal/i.test(detail) ? 'signal' : /fetch|network|connection/i.test(detail) ? 'network' : 'runtime';
    console.error('Upstream request failed', category, error instanceof TypeError ? 'TypeError' : 'other');
    throw new HttpError(503, 'The connected service is unavailable. Please try again later.');
  }
}
