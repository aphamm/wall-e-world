from utils import (
    rescale_bridge_action,
    discover_trials,
    predict,
    aggregate_model_results,
    print_results_table,
)
from world_model import WorldModel
import os
import jax
import numpy as np
from PIL import Image
import mediapy as media
from tqdm import tqdm
import torch
from pathlib import Path
from octo.model.octo_model import OctoModel

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def normalize_actions(unnorm_actions, statistics):
    action_low = np.array(statistics["mean"]-10*statistics["std"])
    action_high = np.array(statistics["mean"]+10*statistics["std"])
    mask = np.array(statistics.get("mask", np.ones_like(action_low)), dtype=bool)

    norm_actions = np.where(
        mask,
        2 * (unnorm_actions - action_low) / (action_high - action_low) - 1,
        unnorm_actions,  # leave unmasked dimensions as-is
    )
    return norm_actions

def evaluate_octo(wm, vla, trials, rollout_length=40, retries=1, save_video=False, video_out_dir=None):
    results = []
    if save_video and video_out_dir:
        Path(video_out_dir).mkdir(parents=True, exist_ok=True)
    for trial in tqdm(trials, desc="Octo trials"):
        start_frame = np.array(Image.open(trial["trial_png"]).resize((256, 256)))
        for r in range(retries):
            wm.reset(torch.from_numpy(start_frame).cuda().float() / 255.0)
            frames = [start_frame]
            for step in range(rollout_length):
                prompt = f"In: What action should the robot take to {trial['instruction']}?\nOut:"
                inputs = {
                    "image_primary": frames[-1][np.newaxis, np.newaxis, ...],
                    "timestep_pad_mask": np.array([[True]]),
                }
                task_spec = vla.create_tasks(texts=[prompt])
                actions = vla.sample_actions(
                    inputs,
                    task_spec,
                    unnormalization_statistics=vla.dataset_statistics["bridge_dataset"]["action"],
                    rng=jax.random.PRNGKey(0),
                )[0]
                seq_len = actions.shape[0]
                action_chunk = torch.zeros((seq_len, 10), device="cuda", dtype=torch.float32)
                for ai in range(seq_len):
                    raw = actions[ai]
                    raw = normalize_actions(raw, vla.dataset_statistics["bridge_dataset"]["action"])
                    a = torch.tensor(raw, device="cuda", dtype=torch.float32)
                    a = torch.cat([a, a.new_zeros(3)], dim=-1)  # pad to 10
                    a = rescale_bridge_action(a, wv_lo=-1, wv_hi=1, rd_lo=-1, rd_hi=1)
                    action_chunk[ai] = a
                if getattr(wm, "chunk_size", None) != seq_len:
                    wm.chunk_size = seq_len
                for _, x in wm.generate_chunk(action_chunk):
                    new_frame = x[0, 0].cpu().numpy()
                    new_frame = np.clip(new_frame * 255, 0, 255).astype(np.uint8)
                    frames.append(new_frame)
            rollout_video = np.stack(frames)
            if save_video and video_out_dir:
                vid_name = Path(trial["trial_png"]).stem
                media.write_video(str(Path(video_out_dir) / f"{vid_name}.mp4"), rollout_video, fps=20)
            score = predict(rollout_video, trial)
            results.append({
                "task_key": trial["task_key"],
                "category": trial["category"],
                "task_display": trial["task_display"],
                "score": float(score),
            })
    return results

CHECKPOINTS_TO_KWARGS = {
    "bridge_v2_ckpt.pt": {  # The demo model checkpoint from our original arxiv release.
        "use_pixel_rope": True,
    },
    "200k_20frame_cfg_bridgev2_ckpt.pt": {  # New in-progress model with CFG and EMA.
        "use_pixel_rope": False,
        "default_cfg": 3.0,
    },
}
FILESERVER_URL = "https://85daf289d906.ngrok.app"  # This might change.

ckpt_path = "200k_20frame_cfg_bridgev2_ckpt.pt"  # Take your pick from above.
if not Path(ckpt_path).exists():
    ckpt_url = FILESERVER_URL + "/" + ckpt_path
    print(f"{ckpt_url=}")
    os.system(f"wget {ckpt_url}")

wm = WorldModel(ckpt_path, **CHECKPOINTS_TO_KWARGS[ckpt_path])

MODEL_NAME = "octo-base-1.5"
model = OctoModel.load_pretrained("hf://rail-berkeley/"+MODEL_NAME)

ROOT_DIR = "/vast/as20482/data/bridge/openvla_evaluation"
trials = discover_trials(ROOT_DIR)
print(f"Discovered {len(trials)} trials.")

results = evaluate_octo(wm, model, trials, rollout_length=15, retries=1,
                        save_video=True, video_out_dir="/vast/as20482/data/bridge/rollouts/chunking/"+MODEL_NAME)

agg = aggregate_model_results(results)
print_results_table(agg)
