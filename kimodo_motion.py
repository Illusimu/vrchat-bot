"""Generate a SOMA motion clip on the local Tesla V100S."""

import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--constraints", type=Path)
    args = parser.parse_args()
    if not args.prompt and not args.constraints:
        parser.error("Provide a motion prompt or a Kimodo constraints JSON file.")
    if not 0.5 <= args.duration <= 30:
        parser.error("Choose a duration between 0.5 and 30 seconds.")
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    gpu, = [row.split(",")[-1].strip() for row in devices if "V100S" in row]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CHECKPOINT_DIR"] = str(Path(__file__).resolve().parent / "build/kimodo-models")
    import numpy as np
    import torch
    from kimodo import load_model
    from kimodo.constraints import load_constraints_lst
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.model.load_model import TEXT_ENCODER_PRESETS
    from kimodo.model.llm2vec import LLM2VecEncoder

    if not torch.cuda.is_available() or "V100S" not in torch.cuda.get_device_name(0):
        raise RuntimeError("PyTorch cannot use the selected V100S.")
    if args.prompt:
        # Volta supports native FP16. Use the official encoder weights in FP16.
        settings = dict(TEXT_ENCODER_PRESETS["llm2vec"]["kwargs"], dtype="float16")
        encoder = LLM2VecEncoder(**settings)
    else:
        # Kimodo masks empty prompts. Only movement constraints condition this run.
        def encoder(texts):
            assert all(not text.strip() for text in texts)
            return torch.zeros(len(texts), 1, 4096), [0] * len(texts)

    model = load_model("Kimodo-SOMA-RP-v1.2", device="cuda:0", text_encoder=encoder)
    constraints = load_constraints_lst(str(args.constraints), model.skeleton) if args.constraints else []
    frames = round(args.duration * model.fps)
    with torch.inference_mode():
        sample = model([args.prompt or ""], [frames], constraint_lst=constraints,
                       num_samples=1, multi_prompt=True, num_denoising_steps=50,
                       post_processing=False, return_numpy=True)
    clip = {key: val[0] if isinstance(val, np.ndarray) and val.shape[:1] == (1,) else val
            for key, val in sample.items()}
    joints = clip["posed_joints"]
    if len(joints) != frames or not np.isfinite(joints).all() or np.ptp(joints, axis=0).max() < 0.001:
        raise RuntimeError("Generated clip failed the joint-motion check.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_kimodo_npz(str(args.output), clip)
    print(f"Saved {frames} frames at {model.fps} fps on {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
