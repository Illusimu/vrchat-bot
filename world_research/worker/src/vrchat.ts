import { hash, HttpError, now, rateLimit, upstream } from './core.ts';
import type { Env } from './core.ts';

async function sessionCookie(env: Env) {
  const seed = await hash(env.VRCHAT_AUTH_COOKIE || '');
  const key = env.VRCHAT_COOKIE_KEY && /^[a-f0-9]{64}$/.test(env.VRCHAT_COOKIE_KEY)
    ? await crypto.subtle.importKey('raw', new Uint8Array(env.VRCHAT_COOKIE_KEY.match(/../g)!.map(v => parseInt(v,16))), 'AES-GCM', false, ['encrypt','decrypt']) : null;
  const saved = await env.DB.prepare("SELECT value FROM integration_state WHERE name = 'vrchat_cookie'").first<{value:string}>();
  let cookie = env.VRCHAT_AUTH_COOKIE!;
  if (saved && key) {
    try {
      const state = JSON.parse(saved.value);
      if (state.seed === seed) cookie = new TextDecoder().decode(await crypto.subtle.decrypt({name:'AES-GCM',iv:new Uint8Array(state.iv)}, key, new Uint8Array(state.data)));
    } catch { throw new HttpError(503, 'VRChat lookup needs to be reconnected by the Worker operator.'); }
  }
  return {cookie, seed, key};
}

async function vrchatGet(env: Env, path: string, shape: (raw: unknown) => unknown, cacheMs: number) {
  if (!env.VRCHAT_AUTH_COOKIE) throw new HttpError(503, 'VRChat research session has not been configured.');
  const cached = await env.DB.prepare('SELECT payload FROM vrchat_cache WHERE cache_key = ? AND expires_at > ?').bind(path, now()).first<{payload:string}>();
  if (cached) return JSON.parse(cached.payload);
  try { await rateLimit(env, 'vrchat:global', 1, 60_000); }
  catch { throw new HttpError(429, 'VRChat lookup is busy. Please try again in one minute.'); }
  const auth = await sessionCookie(env);
  const response = await upstream(`https://api.vrchat.cloud/api/1${path}`, {headers:{Cookie:auth.cookie, 'User-Agent':env.VRCHAT_USER_AGENT, Accept:'application/json'}});
  if (response.status === 404) throw new HttpError(404, 'The requested VRChat research resource was not found.');
  if ([401,403].includes(response.status)) {
    await env.DB.prepare("INSERT INTO integration_state(name,value,updated_at) VALUES ('vrchat_health','reconnect',?) ON CONFLICT(name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at").bind(now()).run();
    throw new HttpError(503, 'VRChat lookup needs to be reconnected by the Worker operator.');
  }
  if (response.status === 429) {
    const retryAfter = response.headers.get('Retry-After') || '';
    const seconds = /^\d+$/.test(retryAfter) ? Number(retryAfter) : (Date.parse(retryAfter)-now())/1000;
    const delay = Number.isFinite(seconds) ? Math.max(60,Math.min(seconds,86400)) : 300;
    await env.DB.prepare("UPDATE rate_limits SET expires_at=? WHERE key='vrchat:global'").bind(now()+delay*1000).run();
    throw new HttpError(503, 'VRChat is limiting lookups. Please try again later.');
  }
  if (!response.ok) throw new HttpError(503, 'VRChat lookup is unavailable. Please try again later.');
  const raw = await response.json<unknown>();
  const value = shape(raw);
  const renewed = response.headers.getSetCookie().map(v => v.split(';')[0]).filter(v => /^(auth|twoFactorAuth)=[A-Za-z0-9_-]+$/.test(v));
  if (renewed.length && auth.key) {
    const cookies = new Map(auth.cookie.split(';').map(v => { const [name,...parts] = v.trim().split('='); return [name,parts.join('=')]; }));
    for (const item of renewed) { const [name,value] = item.split('='); cookies.set(name,value); }
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const ciphertext = await crypto.subtle.encrypt({name:'AES-GCM',iv},auth.key,new TextEncoder().encode([...cookies].map(([k,v])=>`${k}=${v}`).join('; ')));
    await env.DB.prepare("INSERT INTO integration_state(name,value,updated_at) VALUES ('vrchat_cookie',?,?) ON CONFLICT(name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at")
      .bind(JSON.stringify({seed:auth.seed,iv:[...iv],data:[...new Uint8Array(ciphertext)]}),now()).run();
  }
  await env.DB.batch([
    env.DB.prepare('INSERT INTO vrchat_cache(cache_key,payload,expires_at) VALUES (?,?,?) ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload, expires_at=excluded.expires_at').bind(path,JSON.stringify(value),now()+cacheMs),
    env.DB.prepare("INSERT INTO integration_state(name,value,updated_at) VALUES ('vrchat_health','ok',?) ON CONFLICT(name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at").bind(now())
  ]);
  return value;
}
function researchWorlds(raw: unknown, limit: number) {
  if (!Array.isArray(raw) || raw.length > limit) throw new HttpError(502, 'VRChat returned an invalid world list.');
  return raw.map(value => {
    if (!value || typeof value.id !== 'string' || !/^wrld_[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/i.test(value.id)
        || typeof value.name !== 'string' || value.releaseStatus !== 'public') {
      throw new HttpError(502, 'VRChat returned an invalid public world.');
    }
    const count = (value: unknown) => typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : null;
    return {
      id: value.id,
      name: value.name,
      releaseStatus: 'public',
      labsPublicationDate: typeof value.labsPublicationDate === 'string' ? value.labsPublicationDate : null,
      creatorId: typeof value.authorId === 'string' ? value.authorId : null,
      featured: typeof value.featured === 'boolean' ? value.featured : null,
      publicationDate: typeof value.publicationDate === 'string' ? value.publicationDate : null,
    updatedAt: typeof value.updated_at === 'string' ? value.updated_at : null,
    capacity: count(value.capacity),
    platforms: Array.isArray(value.supportedPlatforms)
      ? value.supportedPlatforms.filter((item: unknown) => typeof item === 'string')
      : Array.isArray(value.unityPackages)
        ? [...new Set<string>(value.unityPackages.flatMap((item: {platform?: unknown}) =>
            typeof item?.platform === 'string' ? [item.platform] : []))]
        : null,
      tags: Array.isArray(value.tags) ? value.tags.filter((tag: unknown) => typeof tag === 'string') : null,
      occupants: count(value.occupants),
      visits: count(value.visits),
      favorites: count(value.favorites),
    };
  });
}

export async function activeWorldResearchSnapshot(env: Env) {
  return vrchatGet(env, '/worlds/active?releaseStatus=public&sort=popularity&n=10', raw => ({
    version:1, source:'active', observedAt:new Date(now()).toISOString(), worlds:researchWorlds(raw, 10),
  }), 60_000);
}

export type ResearchPage = {
  version: number; pool: string; sort: string; page: number; limit: number; offset: number;
  observedAt: string; worlds: ReturnType<typeof researchWorlds>;
};

export async function worldResearchPage(env: Env, pool: string | null, page: string | null) : Promise<ResearchPage> {
  const tags = new Map([['games', 'author_tag_game'], ['hangouts', 'author_tag_social,author_tag_hangout'], ['social', 'author_tag_social']]);
  if (!pool || (pool !== 'active' && pool !== 'published' && pool !== 'labs' && !tags.has(pool))) throw new HttpError(400, 'Unknown world pool.');
  const releaseFeed = pool === 'published' || pool === 'labs';
  if (page !== '1' && (page !== '2' || releaseFeed)) throw new HttpError(400, 'Invalid world page.');
  const limit = releaseFeed ? 50 : 100;
  const offset = (Number(page) - 1) * limit;
  const sort = pool === 'labs' ? 'labsPublicationDate' : pool === 'published' ? 'publicationDate' : 'popularity';
  const parameters = new URLSearchParams({releaseStatus:'public', sort, order:'descending', n:String(limit), offset:String(offset)});
  const tag = pool === 'labs' ? 'system_labs' : tags.get(pool);
  if (tag) parameters.set('tag', tag);
  const path = releaseFeed ? '/worlds' : '/worlds/active';
  return vrchatGet(env, `${path}?${parameters}`, raw => ({
    version:2, pool, sort, page:Number(page), limit, offset, observedAt:new Date(now()).toISOString(),
    worlds:researchWorlds(raw, limit),
  }), 60_000) as Promise<ResearchPage>;
}

export type WorldDetailSnapshot = {
  observedAt:string; world:ReturnType<typeof researchWorlds>[number] & {
    description:string|null; authorName:string|null; authorId:string|null; version:number|null;
  };
};
export async function worldResearchDetails(env: Env, id: string | null): Promise<WorldDetailSnapshot> {
  if (!id || !/^wrld_[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/i.test(id)) throw new HttpError(400, 'Invalid world ID.');
  const worldId = id.toLowerCase();
  return vrchatGet(env, '/worlds/' + worldId, raw => {
    const [world] = researchWorlds([raw], 1);
    if (world.id !== worldId) throw new HttpError(502, 'VRChat returned a different world.');
    const details = raw as Record<string, unknown>;
    return { observedAt:new Date(now()).toISOString(), world:{
      ...world,
      description:typeof details.description === 'string' ? details.description : null,
      authorName:typeof details.authorName === 'string' ? details.authorName : null,
      authorId:typeof details.authorId === 'string' ? details.authorId : null,
      version:typeof details.version === 'number' ? details.version : null,
    }};
  }, 60_000) as Promise<WorldDetailSnapshot>;
}
