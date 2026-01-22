# Wall-E-World

<div align="center">
  <img src="docs/static/gifs/gen_000008000.gif" width="180"/>
  <img src="docs/static/gifs/gen_000016000.gif" width="180"/>
  <img src="docs/static/gifs/gen_000024000.gif" width="180"/>
  <img src="docs/static/gifs/gen_000028000.gif" width="180"/>
</div>

---

## Training on Modal

### Setup

```bash
cd modal
uv sync
uv run modal setup
uv run modal secret create huggingface-secret HF_TOKEN=<your-token>
```

### Download Data

```bash
uv run modal run modal_data.py::download --dataset-name bridge
```

### Train

```bash
uv run modal run modal_train.py::train --subset-names bridge
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--subset-names` | `bridge` | Dataset names |
| `--batch-size` | `4` | Batch size per GPU |
| `--max-train-steps` | `100000` | Total training steps |
| `--validate-every` | `2000` | Checkpoint frequency |

### Monitor

```bash
uv run modal app logs world-model-train
uv run modal volume ls world-model-eval checkpoints
```

<details>
<summary><strong>Troubleshooting DDP/NCCL Issues</strong></summary>

When running distributed training on Modal, you may encounter NCCL timeout errors. Here's how we handle them:

**Validation Desync**: All ranks must wait during validation, not just rank 0.
```python
if train_steps % validate_every == 0:
    if rank == 0:
        # validation work
    if distributed:
        dist.barrier()  # All ranks wait
```

**Logging Desync**: Add barriers after rank-0-only file I/O.
```python
if rank == 0:
    plt.savefig("loss.png")
if distributed:
    dist.barrier()
```

**NCCL Environment Variables** (set in `modal/modal_config.py`):
```python
"NCCL_TIMEOUT": "1800",           # 30 min timeout
"NCCL_ASYNC_ERROR_HANDLING": "1", # Better error recovery
"NCCL_IB_DISABLE": "1",           # Disable InfiniBand
```

**Auto-retry**: Training automatically retries on transient failures and resumes from the latest checkpoint.

</details>

---

## Citation

```bibtex
@article{pham2025walleworld,
    title     = {Wall-E-World: Evaluating Robot Policies via Large-Scale Human-Centric World Models},
    author    = {Austin Pham and Hao Gu and Hod Lipson and Yue Wang},
    journal   = {arXiv preprint},
    year      = {2025},
}
```

## Acknowledgements

- [WorldGym](https://arxiv.org/abs/2506.00613) - Quevedo et al.
- [Diffusion Forcing](https://github.com/buoyancy99/diffusion-forcing)
- [DiT](https://github.com/facebookresearch/DiT)
- [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment)
