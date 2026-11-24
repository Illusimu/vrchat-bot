import { HttpError, json } from './core.ts';
import type { Env } from './core.ts';
import { activeWorldResearchSnapshot, worldResearchDetails, worldResearchPage } from './vrchat.ts';
import { acknowledgeWorldBatch, nextWorldBatch, queueWorldSweep } from './world-history.ts';

export { WorldResearchSweep } from './world-sweep.ts';

const prefix = '/api/world-research';

function requireResearchToken(request: Request, env: Env) {
  const expected = env.WORLD_RESEARCH_TOKEN;
  if (!expected || !/^[a-f0-9]{64}$/.test(expected)) {
    throw new HttpError(503, 'World research is unavailable.');
  }
  const token = request.headers.get('Authorization')?.match(/^Bearer ([a-f0-9]{64})$/)?.[1];
  if (!token) throw new HttpError(401, 'Invalid world research token.');
  const encode = new TextEncoder();
  if (!crypto.subtle.timingSafeEqual(encode.encode(expected), encode.encode(token))) {
    throw new HttpError(401, 'Invalid world research token.');
  }
}

async function route(request: Request, env: Env): Promise<Response> {
  const { pathname, searchParams } = new URL(request.url);
  const method = request.method;
  if (method === 'GET' && pathname === '/health') return json({ ok: true });
  if (!pathname.startsWith(prefix + '/')) throw new HttpError(404, 'Not found.');
  requireResearchToken(request, env);

  if (method === 'POST' && pathname === prefix + '/active') return json(await activeWorldResearchSnapshot(env));
  if (method === 'POST' && pathname === prefix + '/page') {
    return json(await worldResearchPage(env, searchParams.get('pool'), searchParams.get('page')));
  }
  if (method === 'GET' && pathname === prefix + '/details') {
    return json(await worldResearchDetails(env, searchParams.get('id')));
  }
  if (method === 'GET' && pathname === prefix + '/sync/next') return nextWorldBatch(env, searchParams.get('after'));
  if (method === 'POST' && pathname === prefix + '/sync/ack') {
    return acknowledgeWorldBatch(env, searchParams.get('sequence'), searchParams.get('id'));
  }
  throw new HttpError(404, 'Not found.');
}

export default {
  async fetch(request, env) {
    const response = await route(request, env).catch(error => {
      if (error instanceof HttpError) return json({ error: error.message }, error.status);
      console.error('World research request failed');
      return json({ error: 'World research request failed.' }, 500);
    });
    const headers = new Headers(response.headers);
    headers.set('Cache-Control', 'no-store');
    headers.set('X-Content-Type-Options', 'nosniff');
    return new Response(response.body, { status: response.status, headers });
  },
  async scheduled(controller, env) {
    await queueWorldSweep(env, controller.scheduledTime);
  },
} satisfies ExportedHandler<Env>;
