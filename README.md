# WorldGym: World Model as An Environment for Policy Evaluation [\[paper\]](https://arxiv.org/abs/2506.00613) [\[website\]](https://world-model-eval.github.io/abstract) [\[demo\]](https://world-model-eval.github.io/) 

<!-- GIF gallery -->
<div style="display: flex; gap: 10px;">
  <img src="media/sweep_z.gif" alt="sweep z" width="200"/>
  <img src="media/sweep_y.gif" alt="sweep y" width="200"/>
  <img src="media/sweep_x.gif" alt="sweep x" width="200"/>
  <img src="media/gripper.gif" alt="gripper" width="200"/>
</div>

[Julian Quevedo](https://julian-q.github.io/)<sup>1</sup>, [Ansh Kumar Sharma](https://www.linkedin.com/in/ansh-ks/)<sup>2</sup>, Yixiang Sun<sup>2</sup>, Varad Suryavanshi<sup>2</sup>, [Percy Liang](https://cs.stanford.edu/~pliang/)<sup>1</sup>, [Sherry Yang](https://sherryy.github.io/)<sup>1,2,3</sup>

Stanford University<sup>1</sup> &nbsp;&nbsp; New York University<sup>2</sup> &nbsp;&nbsp; Google DeepMind<sup>3</sup>

---

## Overview

This repository contains the evaluation harness used in *Evaluating Robot Policies in a World Model*. It bundles

- the pretrained diffusion world model,
- policy-specific runners for OpenVLA, Octo, SpatialVLA, and RT-1-X, and
- utilities for dataset conversion and automatic VLM scoring.

---

## Installation

Install the package in editable mode (optionally with extras for specific policies):

   ```bash
   pip install -e .[openvla,spatialvla,octo,rt1]
   ```

   Extras are additive—omit the ones you do not need. Some stacks have additional one-off steps:

  - **Octo** –
    1. install the dlimp library: `pip install git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972 --no-deps`
    2. edit the installed Octo package (typically under your Python site-packages) and update `octo/utils/typing.py` so that it defines `PRNGKey = jax.random.PRNGKey`.
   - **RT-1-X** – obtain the official JAX checkpoint from the [Open X-Embodiment release](https://github.com/google-deepmind/open_x_embodiment?tab=readme-ov-file#rt-1-x-jax-checkpoint).

---

## World-model checkpoint

The evaluation runners require a diffusion world-model checkpoint, e.g. `mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt` (≈9 GB).

1. Set the file-server URL and the checkpoint filename you want:

  ```python
  FILESERVER_URL = "https://85daf289d906.ngrok.app"  # This might change.
  ckpt_path = "mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt"  # Choose any of the hosted files.
  ```

2. Download the checkpoint if it is not already present alongside your workspace:

  ```python
  from pathlib import Path
  import os

  if not Path(ckpt_path).exists():
     ckpt_url = FILESERVER_URL + "/" + ckpt_path
     print(f"{ckpt_url=}")
     os.system(f"wget {ckpt_url}")
  ```

---

## Prepare evaluation data

Point every runner’s `--root-dir` flag at the directory whose subfolders contain `*.png` + metadata pairs. The helper `discover_trials` recursively discovers tasks from that root.

---

### OpenVLA

```bash
world-model-eval-openvla \
  --root-dir /path/to/tasks \
  --checkpoint-path ~/checkpoints/world-model/mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt \
  --model-name openvla-7b \
  --save-video --video-out-dir ./rollouts/openvla
```

### SpatialVLA

```bash
world-model-eval-spatialvla \
  --root-dir /path/to/tasks \
  --checkpoint-path ~/checkpoints/world-model/mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt \
  --model-name spatialvla-4b-224-pt
```

### Octo

```bash
world-model-eval-octo \
  --root-dir /path/to/tasks \
  --checkpoint-path ~/checkpoints/world-model/mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt \
  --model-name octo-base-1.5
```

### RT-1-X

The RT-1 runner uses Abseil flags:

```bash
world-model-eval-rt1 \
  --root_dir /path/to/tasks \
  --checkpoint_path /path/to/rt1x_checkpoint \
  --world_model_checkpoint ~/checkpoints/world-model/mixed_openx_9robots_20frames_0p1actiondropout_580ksteps.pt
```

Pass `--save_video` / `--video_out_dir` counterparts where available if you want MP4 rollouts.

---

## Training quick start

This is how you launch training. It will train on the tiny 10-example dataset in `sample_data/`.
```bash
# Replace N with the number of available GPUs
torchrun --nproc_per_node=N train.py
```

Checkpoints and generated GIF samples will be written to `outputs/<timestamp>/`.

## Train on Open X-Embodiment Datasets
To train on the Open X-Embodiment datasets we used in the paper:
```bash
# We'll need tensorflow datasets and tensorflow since this code is 
# based on the original Open X-Embodiment repo.
pip install tensorflow tensorflow_datasets
# For example, download just the RT-1 dataset:
python -m world_model_eval.download_data --dataset_name rt_1
# By default the data will be written to ./converted_datasets.
# To choose your own output directory:
python -m world_model_eval.download_data --dataset_name rt_1 --output_dir <your output dir>
```
See `world_model_eval/download_data.py` for more dataset names to choose from.


Then launch training with the correct dataset path:
```bash
torchrun --nproc_per_node=N -m world_model_eval.train --dataset_dir ./converted_datasets --subset_names rt_1
# Replace ./converted_datasets if your path is different.
```
You can enter a comma separated list for `subset_names` to train on a mixture of multiple datasets. For example, after downloading the `rt_1` and `bridge_v2` datasets, you can do `--subset_names rt_1,bridge_v2` to train on both the RT-1 and Bridge V2 datasets.

#### Training on Bridge V2
Since Bridge V2 was not included in the original Open X-Embodiment dataset, you'll need to first download the TFDS dataset to your machine like this:
```
wget -r -np -R "index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/
```
Then, convert the dataset to our format with `python -m world_model_eval.download_data --dataset_name bridge_v2`, changing `BRIDGE_V2_PATH` at the top of the script if necessary. Since Bridge V2 is a superset of Bridge V1, choose between either downloading `bridge` or `bridge_v2`.

---

## Citation

If you find this work useful, please cite:

```text
@misc{quevedo2025worldgymworldmodelenvironment,
      title={WorldGym: World Model as An Environment for Policy Evaluation}, 
      author={Julian Quevedo and Ansh Kumar Sharma and Yixiang Sun and Varad Suryavanshi and Percy Liang and Sherry Yang},
      year={2025},
      eprint={2506.00613},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.00613}, 
}
```

---

## Acknowledgements
- [Boyuan Chen](https://boyuan.space/) and [Kiwhan Song](https://kiwhan.dev/) for [Diffusion Forcing](https://github.com/buoyancy99/diffusion-forcing)
- [DiT](https://github.com/facebookresearch/DiT)
- [Oasis](https://github.com/etched-ai/open-oasis)
- [open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment)

