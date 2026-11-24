"""Measure local Inkling inference with controlled text and optional VR images."""

import argparse
import base64
import json
from pathlib import Path
from time import monotonic

import httpx


PROMPT = (
    "Write an operations note for a virtual robot moving a blue cube from one table to another. "
    "Give concrete observations for safe pickup, carrying through a doorway, and verifying placement. "
    "Use specific visual cues and precise short sentences.\n\nField notes:\n"
)


def measure(client, route, payload):
    start = monotonic()
    response = client.post(route, json=payload)
    response.raise_for_status()
    data = response.json()
    return dict(wall_seconds=round(monotonic() - start, 3), timings=data.get("timings"),
                usage=data.get("usage"), choice=data.get("choices", [{}])[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--prefix-lines", type=int, default=0)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    prefix = "".join(
        f"Fixture {index:05d}. The fixture is idle with the green marker at station {index % 73:02d}.\n"
        for index in range(args.prefix_lines)
    )
    with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=1200, trust_env=False) as client:
        props = client.get("/props")
        props.raise_for_status()
        settings = props.json()["default_generation_settings"]
        print(json.dumps(dict(context=settings["n_ctx"])), flush=True)
        for index in range(args.repeats):
            result = measure(client, "/completion", dict(
                prompt=prefix + PROMPT, n_predict=args.tokens, temperature=0, seed=61,
                ignore_eos=True, cache_prompt=False, timings_per_token=True,
            ))
            print(json.dumps(dict(case="text", run=index + 1, **result)), flush=True)
        if args.image:
            url = "data:image/jpeg;base64," + base64.b64encode(args.image.read_bytes()).decode("ascii")
            result = measure(client, "/v1/chat/completions", dict(
                model="vr-agent", temperature=0, max_tokens=160, cache_prompt=False, timings_per_token=True,
                reasoning_effort="none", messages=[{"role": "user", "content": [
                    {"type": "text", "text": "What heading is visible on the VRChat panel? Answer briefly."},
                    {"type": "image_url", "image_url": {"url": url}},
                ]}],
            ))
            message = result.pop("choice").get("message", {})
            print(json.dumps(dict(case="vision", answer=message.get("content"), **result)), flush=True)


if __name__ == "__main__":
    main()
