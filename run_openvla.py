from utils import (
    rescale_bridge_action,
    discover_trials,
    predict,
    aggregate_model_results,
    print_results_table,
)
from transformers import AutoModelForVision2Seq, AutoProcessor
from world_model import WorldModel
import os
import numpy as np
from PIL import Image
import mediapy as media
from tqdm import tqdm
import torch
from pathlib import Path


def evaluate_openvla(wm, vla, processor, trials, retries=1, rollout_length=40, save_video=False, video_out_dir=None):
    """
    Rollout an OpenVLA model on a list of tasks, and return the score on each task.
    Arguments:
        wm: WorldModel
        vla: An OpenVLA model from `transformers`
        tasks: A list of N tasks in loaded from a json. See "put_carrot_on_plate.json" for an example of the format.
    Returns:
        scores: A list of N scores from the VLM corresponding to each input task.
    """
    results = []
    if save_video and video_out_dir:
        Path(video_out_dir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for trial in tqdm(trials, desc="Openvla trials"):
            start_frame = np.array(Image.open(trial["trial_png"]).resize((256, 256)))
            for r in range(retries):
                wm.reset(torch.from_numpy(start_frame).cuda().float() / 255.0)
                frames = [start_frame]
                for step in tqdm(range(rollout_length)):
                    curr_frame = Image.fromarray(frames[-1])
                    prompt = f"In: What action should the robot take to {trial['instruction']}?\nOut:"
                    # system_prompt = (
                    #     "A chat between a curious user and an artificial intelligence assistant. "
                    #     "The assistant gives helpful, detailed, and polite answers to the user's questions."
                    # )
                    # prompt = f"{system_prompt} USER: What action should the robot take to {trial['instruction']}? ASSISTANT:"
                    inputs = processor(prompt, curr_frame).to(
                        device="cuda", dtype=torch.bfloat16
                    )
                    actions = vla.predict_action(
                        **inputs, unnorm_key="bridge_orig", do_sample=False
                    )
                    a = torch.tensor(actions).cuda()
                    # NOTE: OpenVLA outputs 7-dim actions, while the world model was trained with up to 10-dim actions.
                    a = torch.cat([a, a.new_zeros(3)], dim=-1)  # pad with zeros
                    # a = rescale_bridge_action(a, wv_lo = -0.25, wv_hi = +0.25, rd_lo = -0.25, rd_hi = +0.25)
                    a = rescale_bridge_action(a)
                    for i, x in wm.generate_chunk(a):
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

MODEL_NAME = "openvla-7b" #openvla-v01-7b

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/"+MODEL_NAME, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/"+MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).cuda()
vla.eval()

ROOT_DIR = "/vast/as20482/data/bridge/openvla_evaluation"
trials = discover_trials(ROOT_DIR)
print(f"Discovered {len(trials)} trials.")

results = evaluate_openvla(wm, vla, processor, trials, rollout_length=105, retries=1,
                        save_video=True, video_out_dir="/vast/as20482/data/bridge/rollouts/"+MODEL_NAME)

agg = aggregate_model_results(results)
print_results_table(agg)
