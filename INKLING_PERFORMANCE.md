# Inkling 2 Mini on this PC

The model runs locally on a 16 GB RTX 4060 Ti with 32 GB of installed system RAM. The performance goal is a reliable single-request vision and tool-use server for Desktop Body, targeting 70–80 tok/s with VRChat running and 100 tok/s as a stretch. Run throughput comparisons with the same prompt and output limit, and record whether VRChat was open. Decode speed, prompt processing speed, and time to a usable final answer measure different parts of the job.

The benchmark and its image replay accept `--thinking-budget` for controlled comparisons. The default remains 64. The requested cap is recorded on live Inkling trials along with reported output usage and timing.

In an isolated saved-frame replay with VRChat open, three hard decisions used the same frames, history, seed, and output limit under caps of 64, 384, and 768. Median decision time increased from 10.99 to 16.81 to 23.50 seconds. At 384, the reception task switched from locating the potted plant to checking the camera, but still did not finish. The other larger-budget actions kept searching, turned the avatar, or shifted the already-carried mug farther left. These runs issued no simulator controls and did not warrant changing the default. See `build/vr-benchmarks/20261123-102314-124535-inkling-action-budget-replay/events.jsonl`. [NInfer thinking-budget behavior](https://github.com/iamwavecut/ninfer-all/blob/master/docs/serving.md)

The runner now groups action names that share identical argument rules and lists the common required fields once. The permitted names and argument constraints stay equivalent. At full image size, this reduced the nod replay from 8,767 to 7,396 prompt tokens and reported prefill from 8.17 to 6.63 seconds. The menu-volume replay went from 8,533 to 7,096 prompt tokens. Both still selected the correct action. Resizing the same inputs to 384 pixels on the long edge saved 220 prompt tokens in the menu case and 380 in the nod case. The default image size remains 768 pixels. See `build/vr-benchmarks/20261123-102617-991567-inkling-capture-size-replay/events.jsonl` and `build/vr-benchmarks/20261123-102756-794062-inkling-image-and-schema-replay/events.jsonl`. [NInfer structured output](https://github.com/iamwavecut/ninfer-all/blob/master/docs/serving.md#structured-output)

The compact action schema passed a live `menu-05` regression after the starting scene matched the manual baseline. Inkling presented the Quick Menu, lined up Audio, opened the panel, and closed it in four decisions over 40.20 seconds. The Audio and closed-menu screenshots were reviewed separately. See `build/vr-benchmarks/20261123-103012-341175-menu-05/`.

Object trials now use instructions for picking up, carrying, and placing props. On two paired saved-frame decisions with the same action schema and history, the shorter instructions reduced input size from 7,391 to 4,518 tokens for pickup and from 8,483 to 5,610 for transfer. Total response times fell from 16.52 to 11.95 seconds and from 19.19 to 13.84 seconds. Both prompts selected the same tool in each comparison. Their requested forward distances differed by 5 cm in the transfer frame. See `build/vr-benchmarks/20261123-171845-845095-object04-compact-instructions-comparison/`.

Navigation trials now use instructions for visible floor, movement, and arrival checks. Three paired saved-frame decisions each used 2,887 fewer prompt tokens. Response times fell from 19.44 to 12.61 seconds, 16.47 to 12.03 seconds, and 16.17 to 10.95 seconds. Both prompt versions chose the same tool in each pair. At the verified hall desk, both chose to survey on the first frame and both still mistook the desk for the main bar in the replay after a failed locator call. The couch approach began with a locator call in both variants. These replays sent no VR controls. See `build/vr-benchmarks/20261123-184116-685010-nav05-verified-departure-cues-replay/`.

## Verified model identity

Thinking Machines Lab publishes Inkling 2 Mini as `thinkingmachines/Inkling-2-Mini`. The running NInfer server loads WaveCut's UD-Q2_K_XL conversion of that model. The file on this PC is 10,080,361,216 bytes and its SHA-256 matches WaveCut's published digest `fe0e309227b0ca6e42ecb5d6d10ced379eafeea1251492951c20c257c88b93b4`. [Thinking Machines model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [WaveCut artifact and conversion notes](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-Q2_K_XL-NInfer-v3)

On November 23, 2026, the live server on loopback port 8097 reported a loaded `vr-agent` model with a 32,768-token context and vision enabled. The running command uses GPU expert residency, MXFP8 KV, DSpark with two draft tokens, and a 2 GiB host prefix cache. The 1,048,576-token training context in its metadata describes the model's configuration. The server enforces its separate 32,768-token runtime limit.

With VRChat running in the private Black Cat instance, the completed booth transfer and doorway carry benchmarks each measured a median of about 69 decode tokens per second across 24 and 28 structured model decisions, respectively. Their end-to-end durations were 244.63 and 269.22 seconds, including prompt processing, visual checks, and controls. See `build/vr-benchmarks/20261123-210827-939853-object-03/events.jsonl` and `build/vr-benchmarks/20261123-213120-635154-object-05/events.jsonl`.

## Tested configurations

These text measurements used the same prompt, one request at a time, with VRChat open. The model was served locally and each request prefills the prompt again.

| Configuration | Context | Decode | Result |
| --- | ---: | ---: | --- |
| Mainline llama.cpp, IQ3_XXS, GPU | 4,096 | 14.3–14.6 tok/s | Read the Audio heading from a saved VRChat screenshot in 11.1 seconds. |
| Unsloth llama.cpp, IQ3_XXS, DSpark | 4,096 | 16.9–17.9 tok/s | Read the same heading in 8.3 seconds. |
| Unsloth llama.cpp, IQ3_XXS, DSpark | 16,384 | 17.5–18.6 tok/s | Read the same heading in 7.9 seconds. Windows measured 12.84 GiB of dedicated GPU memory and 0.11 GiB of shared GPU memory for the model process. |
| Ada-tuned llama.cpp, same mainline IQ3_XXS | 4,096 | 16.1–17.2 tok/s | The image request used its token allowance before returning a final answer. |
| Ada-tuned llama.cpp, mixB | 32,768 | 17.2–17.6 tok/s | Read the same heading in 6.6 seconds. |
| NInfer, UD-Q2_K_XL, all experts on GPU, DSpark | 32,768 | 24.0–24.6 tok/s | Read the same heading in 3.6 seconds, decoding at 31.0 tok/s on that image request. |
| NInfer, same model, DSpark depth 2 and smaller host context cache | 32,768 | 23.7–25.2 tok/s | Current profile. Read the heading in 3.5 seconds. Windows measured 12.5 GiB of dedicated GPU memory, 0.57 GiB of shared GPU memory, and a 1.96 GiB process working set after the run. |
| NInfer, same model, plain decode | 32,768 | 22.4–22.6 tok/s | Read the same heading in 4.5 seconds. |
| NInfer, same model, host experts with a 6 GiB GPU cache and DSpark | 32,768 | 16.7–17.2 tok/s | The split used 7.5 GiB of pinned RAM for expert sources. Read the heading in 4.8 seconds. |
| Mainline llama.cpp, 8 CPU MoE layers, Q4 KV | 262,144 | 11.1–12.6 tok/s | Loaded the configured context and processed an 8,452-token prompt in about 86 seconds before generation. |
| Mainline llama.cpp, 12 CPU MoE layers, Q4 KV | 262,144 | 9.2–9.5 tok/s | Same 892-token prompt used for the cache comparison. |
| Same 12-layer profile with a 256 MiB expert-cache budget | 262,144 | 4.6–4.9 tok/s | The runtime allocated about 235 MiB of cache. Leave it disabled for this profile. |
| Ada-tuned llama.cpp, 4 CPU MoE layers, Q4 KV | 262,144 | 0.75 tok/s | The first short decode slowed markedly with about 300 MiB of dedicated VRAM free. Stopped this configuration after the first result. |
| NInfer, same model, 4-bit KV, plain decode | 262,144 | 22.5–22.8 tok/s | Read the heading in 4.8 seconds. Prefilled 8,452 tokens in 15.2 seconds and then decoded at 21.1 tok/s. Startup left roughly 320 MiB of dedicated VRAM according to NVIDIA's system-wide reading. |

The qualified 256K result refers to the server's allocated context capacity. The largest prompt tested in the initial comparison was 8,452 tokens. The mainline 256K build left roughly 180 MiB of dedicated VRAM free with eight CPU MoE layers and VRChat in the tested scene.

The 16K DSpark profile processed the same 8,452-token prompt in 69 seconds. Its subsequent decode slowed to 9.8 tok/s with 19 of 79 draft tokens accepted. On the short screenshot prompt it accepted 48 of 78 and decoded at 20.5 tok/s.

NInfer's standalone benchmark with the UD-Q2_K_XL file and FP16 KV measured 22.3 tok/s for a 128-token decode and 20.3 tok/s after a 10,000-token prefill. It prefilled the 10,000 tokens at 554 tok/s. This run excludes the chat template, vision encoder, and HTTP request handling.

The active NInfer model file is 9.39 GiB on disk and includes the vision tower and drafter. Its installed Windows runtime adds 2.87 GiB. The mainline IQ3_XXS model is 10.86 GiB, and the F16 vision projector is 0.60 GiB. The Unsloth variant uses a separate 10.86 GiB copy of the weights in its own layout and a 0.375 GiB DSpark drafter. The original four checkpoints and runtimes total about 47.5 GiB. The UD-IQ4_XS comparison adds 15.36 GiB.

## Impact of running VRChat on the same GPU

These tests used the same GPU-resident UD-Q2_K_XL checkpoint, the same short text prompt, and a 128-token greedy decode. SteamVR stayed open throughout. Each row shows three runs.

| Game state | NInfer profile | Decode |
| --- | --- | ---: |
| VRChat Home, original launch | 32K MXFP8, DSpark depth 2 | 24.5–24.8 tok/s |
| VRChat closed | 32K MXFP8, DSpark depth 2 | 57.3–67.5 tok/s |
| VRChat closed, repeat | 32K MXFP8, DSpark depth 2 | 64.4–68.1 tok/s |
| VRChat closed | 32K MXFP8, DSpark depth 3 | 63.1–66.7 tok/s |
| VRChat closed | 32K MXFP8, DSpark depth 6 | 43.4–43.9 tok/s |
| VRChat Home launched with `--fps=30` | 32K MXFP8, DSpark depth 2 | 14.9–27.7 tok/s |
| Same 30 FPS launch | 8K RK4V4, DSpark depth 2 | 21.4–22.3 tok/s |
| 30 FPS launch, SteamVR at 50% per-app resolution and `framesToThrottle=3` | 8K RK4V4, DSpark depth 2 | 22.8, 59.2, 60.2 tok/s |
| Same reduced rendering, five runs | 32K MXFP8, DSpark depth 2 | 48.7–52.4 tok/s |
| Same reduced rendering | 32K RK8V4, DSpark depth 2 | 40.6–46.6 tok/s |

With VRChat closed, the 32K depth-2 profile decoded at 70.2 tok/s after a 3,412-token prompt. It correctly identified **Audio** in a saved VRChat frame in 1.75 seconds, with 84.4 tok/s generation on that response. Depth 3 identified the same heading in 1.67 seconds at 89.2 tok/s. Those image-request decode rates describe one short response each. The repeatable text tests favor depth 2.

Closing VRChat reduced system-wide dedicated GPU memory use from roughly 14,800 to 13,800 MiB immediately before the repeated inference tests. The FPS launch option was present in the VRChat log. Its effect on submitted frames was unverified in the first launch test. The reduced-render experiment reached roughly 14 submitted frames per second during a short inference sample. The 32K MXFP8 profile identified Audio in the saved screenshot in 2.22 seconds. After relaunching, Desktop Body opened the Quick Menu and Inkling correctly read **Launch Pad** from a fresh game-mirror frame in 1.41 seconds, with generation at 70.3 tok/s for that short response.

In the live `menu-04` benchmark, the 32K MXFP8 profile with DSpark depth 2 produced a structured action from an Audio screenshot in 4.75 seconds, including prefill. NInfer reported 86.6 tok/s for 150 generated tokens, including reasoning. The world-search controls then ran for 31.63 seconds. This is one run with VRChat open on the RTX 4060 Ti. The trace is `build/vr-benchmarks/20261122-165254-899419-menu-04/events.jsonl`.

In the matched three-landmark navigation trial, 41 structured decisions averaged 7.57 seconds of prompt processing and 2.30 seconds of decode at 69.6 tok/s. With the earlier 512 MiB host cache, two saved-frame tests reported zero reused tokens on repeat requests and on a changing view. See `build/vr-benchmarks/20261122-235116-499636-nav-05/events.jsonl`, `build/vr-benchmarks/20261123-000103-431587-inkling-nav-prefix-cache/events.jsonl`, and `build/vr-benchmarks/20261123-134240-996676-inkling-prefix-reuse-isolation/events.jsonl`.

At 2 GiB, the same navigation replay reused 6,656 prompt tokens. The matching-frame request took 2.88–2.95 seconds after a 9.58-second first request. Replacing the screenshot took 3.61 seconds and produced a description of the new view. This build continued to report reuse on the Chat Completions endpoint after the client set `cache_prompt` to false. The server's `--no-prefix-reuse` option verified zero reuse in the live control below. The harness still builds every request from the current trial's history and images. NInfer reports reuse only for an exact prompt prefix. See `build/vr-benchmarks/20261123-134437-724019-inkling-raised-host-cache/events.jsonl` and `build/vr-benchmarks/20261123-134541-082557-inkling-cold-control-after-warm/events.jsonl`. [NInfer serving documentation](https://github.com/iamwavecut/ninfer-all/blob/master/docs/serving.md)

Both live `menu-05` trials matched the same manual start and passed the Audio and closed-menu checks. VRChat stayed open in the same private instance. Both runs made the same four tool choices.

| NInfer startup | Full trial | Total inference | Reused tokens by decision |
| --- | ---: | ---: | --- |
| 2 GiB host prefix cache | 25.19 s | 17.75 s | 0, 6,151, 6,151, 6,151 |
| Prefix reuse disabled | 40.38 s | 33.06 s | 0, 0, 0, 0 |

The larger cache saved 15.19 seconds on that menu sequence. After the cache run, NInfer used about 3.59 GiB of system RAM, with about 11.73 GiB of free physical RAM across the machine. NInfer's 32K context and VRChat remained available. See `build/vr-benchmarks/20261123-134638-974182-menu-05/events.jsonl` and `build/vr-benchmarks/20261123-134821-939101-menu-05/events.jsonl`.

An observation-only placement replay compared 768 px and 512 px game-mirror inputs with the same prompt and VRChat open. Both sizes chose the same explicit grip release on both repetitions. The smaller frame reduced the prompt from 8,984 to 8,814 tokens and averaged 9.22 seconds per decision, compared with 9.34 seconds at 768 px. This single scene showed limited savings, so the benchmark still uses its existing frame size. See `build/inkling-object-feedback/placement-vision-size-replay.json`.

A separate saved-frame comparison reduced the schema from 31 action definitions to 12 while keeping the screenshots and task prompt fixed. Across two observations, mean prefill decreased from 7.24 to 5.39 seconds and mean time per decision decreased from 9.25 to 7.48 seconds. Both schema sizes chose to shift in the first frame and release in the second. The smaller schema changed the first target from x=.33 to x=.42, which would move right from the observed x=.33. The full action schema remains in use. See `build/inkling-object-feedback/placement-schema-size-replay.json`.

## Dynamic MoE expert cache

With VRChat closed, NInfer using the verified UD-IQ4_XS artifact, FP16 KV, a 10,240-token context and plain decoding measured 33.4–35.5 tok/s on the short text and 32.9 tok/s after 9,922 prompt tokens. It prefills that prompt at 1,190 tok/s. The engine allocated a 9,080 MiB GPU expert cache, pinned 10,935 MiB of expert sources in system RAM, and left 2,450 MiB of expert sources in the model file. The smaller all-GPU UD-Q2_K_XL had faster decode here and remains the default.

```powershell
.\start-inkling-ninfer.ps1 -Quantization UD-IQ4_XS -ExpertResidency host -KvType fp16 -Context 10240 -Speculation none
```

## Run VRChat with reduced rendering

The tested SteamVR settings applied only to VRChat: 50% per-application resolution and `framesToThrottle=3`. SteamVR retains those settings between launches. To return to the standard settings, open SteamVR's per-application video settings for VRChat and use **Reset to Default**. The per-launch FPS cap is applied by launching with:

```powershell
& 'C:\Program Files (x86)\Steam\steam.exe' -applaunch 438100 -vrmode openvr --fps=30
```

The pilot can request screenshots as needed. VRChat and the compositor still render between requests, so lowering screenshot frequency by itself does not reduce their rendering load.

## Reproduce inference

Start the server in a terminal and stop it with Ctrl+C.

```powershell
.\start-inkling-ninfer.ps1
.\.venv\Scripts\python.exe inkling_perf.py --repeats 3
```

For a cold-prefill control, launch with `-NoPrefixReuse`. To test a smaller host cache while retaining prefix reuse, launch with `-HostCacheMiB 512`. Inspect the returned `timings.cache_n` when interpreting prefill measurements.

The NInfer long-context configuration fits this smaller quant on the GPU by compressing the KV cache and disabling its drafter. The larger context leaves less room for VRChat. The mainline launcher also supports moving experts from selected layers to system RAM with `-CpuMoeLayers`.

```powershell
.\start-inkling-ninfer.ps1 -Context 262144 -KvType rk4v4 -PrefillChunk 512 -Speculation none
.\.venv\Scripts\python.exe inkling_perf.py --repeats 1 --prefix-lines 400
```

Machine-readable measurements are in `build/inkling-2-mini/`. Use a saved VRChat screenshot with `inkling_perf.py --image PATH` to measure whether the model returns a usable vision answer as well as how quickly it generates.

## Published reference points

- NInfer measured 53.8 tok/s on an RTX 4060 Ti 16 GB with a Ryzen 9 7900X and host-expert caching, or 47.2 tok/s after a 10,000-token prompt. On the same card, its essay went from 52.3 to 55.7 tok/s with DSpark. Its RTX 5090 reached 182.1 tok/s on a short decode with all experts on the GPU. The model, prompt lengths, runtime, and CPU are recorded with the measurements. [NInfer Inkling benchmark](https://github.com/iamwavecut/ninfer-all/blob/master/docs/inkling2.md)
- Petronella measured 158.2 tok/s for single-user prose on an RTX 5090 using the native NVFP4/MXFP8 checkpoint and vLLM. Strata measured about 177–199 tok/s on a 5090 with its SQ-20 package. [Petronella measurements](https://petronellatech.com/ai/llm-benchmarks/models/inkling-2-mini/) · [Strata results](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata)
- The Ada-tuned llama.cpp author measured 61–64 tok/s on an RTX 4070 Ti at 64K using the custom mixB checkpoint. The same author measured 51–53 tok/s deeper into a 22K prompt. The published speed depends on both the mixed quantization and the graph/CUDA changes. [Ada build and benchmark](https://github.com/hridaya423/inkling-2-mini-12gb)
- llama.cpp's dynamic MoE cache keeps frequently used CPU-resident experts on the GPU for decoding. The payoff depends on how many experts fit and the cost of cache misses. [llama.cpp expert caching](https://github.com/ggml-org/llama.cpp/blob/master/docs/moe-cache.md)
