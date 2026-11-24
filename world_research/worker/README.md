# Standalone VRChat World Research Worker

This Worker serves the authenticated research API and collects scheduled
VRChat world snapshots into a D1 buffer. The Windows importer writes long-term
history to SQLite. The dashboard reads that local SQLite file.

## Build

Run `npm ci`, `npm run check`, and `npm run build` in this directory.
The build is a Wrangler dry run. The checked-in config has placeholder D1 IDs,
with scheduled collection and public routes disabled.

## Restore and cut over

1. Authenticate Wrangler with the destination Cloudflare account. Create a
   runtime database and a research buffer database. Replace both placeholder
   IDs in `wrangler.jsonc`, then apply `runtime.sql` to the empty runtime
   database. Set `VRCHAT_USER_AGENT` to an application identifier with your
   contact information.
2. Prepare the latest research export for D1. Use the last assigned sequence
   from the export manifest, including when old acknowledged rows have expired.
   Run `python prepare_restore.py <export.sql> <restore.sql> --high-water <sequence>`
   and import the generated file into an empty research buffer. This preserves
   IDs, timestamps and payloads while keeping statements below D1's size limit.
3. Provision `VRCHAT_AUTH_COOKIE`, `VRCHAT_COOKIE_KEY`, and
   `WORLD_RESEARCH_TOKEN` as Worker secrets. Use a fresh authorized VRChat
   session and fresh 32-byte random keys in lowercase hex for the last two.
   Run `./connect-vrchat.ps1 -Config wrangler.deploy.jsonc` after the Worker
   is deployed to authorize a VRChat session interactively.
   Use the Windows DPAPI procedure in [MIGRATION.md](../../MIGRATION.md) to
   store the matching research token on this PC.
4. Enable the intended route and deploy with scheduling disabled. Verify
   capture, import, acknowledgment, and replay with a scratch buffer and a
   scratch SQLite copy. On the destination Windows user account, set
   `WORLD_RESEARCH_ORIGIN` to the new HTTPS origin. Both the scheduled importer
   and the manual scan scripts use it.
5. Reconcile batches after the last backup. Drain the old research Workflow
   at the agreed slot boundary, restore the final delta, and stop old research
   scheduling and syncing after verification. Then set
   `WORLD_RESEARCH_SCHEDULE` to `enabled`, configure the
   `*/15 * * * *` trigger, and register the destination Windows sync task.
   Record the last old and first new batch IDs. Verify that replay adds no
   duplicates.
