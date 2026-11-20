# Move the VRChat Bot to another Windows PC

Clone the repository on the destination PC and open its folder in Codex. The repository contains the SteamVR driver source, pilot, harness, research collector, dashboard, and hosted backend patches.

## Research database

1. Install Python 3.11 or newer and run `python -m pip install tzdata` to provide the Central Time zone on Windows.
2. Transfer `world-research.sqlite3` separately and put it at the repository root. Use a consistent SQLite backup or pause the old importer and let any active import finish before copying. Keep the old copy until the new import works.
3. Run `python world_research/explorer.py` and open [the local dashboard](http://127.0.0.1:8768/).
4. Provision the same dedicated Cloudflare Worker token through a secure channel. On the destination, run `Read-Host 'World research token' -AsSecureString | ConvertFrom-SecureString | Set-Content .world-research-token` from the repository root. The old `.world-research-token` is encrypted for the original Windows user and machine.
5. Run `python world_research/hosted_sync.py --replay` to import retained batches and verify the credential. Run `./world_research/register-sync-task.ps1` to set up ongoing sync. Keep one PC responsible for ongoing imports. Cloudflare continues collecting during the move.

## VR pilot

Install SteamVR, Python with Tk, and the Python pilot dependencies in your environment.

```powershell
python -m pip install numpy scipy Pillow opencv-python mcp python-osc
```

Follow the [SteamVR driver build and registration steps](README.md). To reuse the local model, copy the model weights, projector, and runtime from `build/agent-model` and `build/llama-cuda`. The current `start-vr-model.ps1` targets a Tesla V100S and CUDA 12.8. Adjust its GPU selection and runtime path for the destination. Copy any saved maps or motion assets you want from `build` and recreate machine-specific environments such as `build/kimodo-env`.

The live collector uses the separate EscapeThroughReality Website project. Bring that repository and Cloudflare access over if you need to change the hosted collector. The Codex chats are separate from this checkout.
