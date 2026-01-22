"""Modal training for wall-e-world.

Usage:
    # Default: 4x A100-80GB training on Bridge data:
    modal run modal_train.py::train --subset-names bridge
"""

import os
import shutil
from pathlib import Path

from modal_config import VOLUME, with_modal


def copy_data_to_local(
    subset_names: str, source_dir: str, local_dir: str, num_processes: int = 16
) -> str:
    """Copy dataset from network volume to local SSD using parallel processes.

    Uses multiple processes to speed up copying large datasets.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    source = Path(source_dir)
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)

    subsets = [s.strip() for s in subset_names.split(",")]

    for subset in subsets:
        subset_source = source / subset
        subset_local = local / subset

        if not subset_source.exists():
            print(f"Warning: {subset_source} does not exist, skipping")
            continue

        if subset_local.exists():
            print(f"Local cache exists for {subset}, skipping copy")
            continue

        # Create directory structure first
        for split in ["train", "test"]:
            (subset_local / split).mkdir(parents=True, exist_ok=True)

        # Get list of all files to copy
        files_to_copy = []
        for split in ["train", "test"]:
            split_source = subset_source / split
            split_local = subset_local / split
            if split_source.exists():
                for f in split_source.iterdir():
                    files_to_copy.append((f, split_local / f.name))

        print(
            f"Copying {subset} ({len(files_to_copy)} files) using {num_processes} processes..."
        )

        def copy_file(args):
            src, dst = args
            shutil.copy2(src, dst)
            return src.name

        # Parallel copy
        completed = 0
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = {executor.submit(copy_file, f): f for f in files_to_copy}
            for future in as_completed(futures):
                completed += 1
                if completed % 1000 == 0:
                    print(f"  Progress: {completed}/{len(files_to_copy)} files copied")

        print(f"  Done: {completed} files copied")

    return local_dir


@with_modal("world-model-train", timeout=24, gpu="A100-80GB:4", cpu=16)
def train(
    subset_names: str = "bridge",
    batch_size: int = 4,
    max_train_steps: int = 100_000,
    lr: float = 8e-5,
    validate_every: int = 2_000,
    use_local_cache: bool = True,
    num_workers: int = 8,
    num_gpus: int = 4,
    max_retries: int = 5,
):
    """Run multi-GPU distributed training using torchrun.

    Args:
        subset_names: Comma-separated dataset names (e.g., "bridge,rt_1")
        batch_size: Training batch size PER GPU (total = batch_size * num_gpus)
        max_train_steps: Maximum training steps
        lr: Learning rate
        validate_every: Validation frequency
        use_local_cache: Copy data to local SSD first for faster I/O (default: True)
        num_workers: DataLoader workers per GPU
        num_gpus: Number of GPUs to use
        max_retries: Maximum number of retry attempts on NCCL/training failures
    """
    import subprocess
    import time

    import torch

    # Verify GPU count
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    if available_gpus < num_gpus:
        print(f"Warning: requested {num_gpus} GPUs but only {available_gpus} available")
        num_gpus = available_gpus

    data_dir = os.environ["DATA_DIR"]

    # Copy data to local SSD for faster I/O
    if use_local_cache:
        local_data_dir = "/tmp/data"
        print("Copying data to local SSD for faster I/O...")
        data_dir = copy_data_to_local(subset_names, data_dir, local_data_dir)
        print(f"Using local data directory: {data_dir}")

    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--standalone",
        "-m",
        "wall_e_world.train",
        f"--dataset_dir={data_dir}",
        f"--checkpoint_dir={os.environ['CHECKPOINT_DIR']}",
        f"--subset_names={subset_names}",
        f"--batch_size={batch_size}",
        f"--max_train_steps={max_train_steps}",
        f"--lr={lr}",
        f"--validate_every={validate_every}",
        f"--num_workers={num_workers}",
    ]

    print(f"Running: {' '.join(cmd)}")

    # Retry loop for transient NCCL failures
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, check=True)
            print("Training completed successfully!")
            break
        except subprocess.CalledProcessError as e:
            print(f"\n{'=' * 60}")
            print(f"Training failed on attempt {attempt + 1}/{max_retries}")
            print(f"Exit code: {e.returncode}")
            print(f"{'=' * 60}\n")

            # Save checkpoint progress before retry
            print("Committing volume to save checkpoint progress...")
            VOLUME.commit()

            if attempt + 1 < max_retries:
                wait_time = 30 * (attempt + 1)  # Increasing backoff
                print(
                    f"Waiting {wait_time}s before retry (will resume from checkpoint)..."
                )
                time.sleep(wait_time)
                print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
            else:
                print("Max retries exceeded. Training failed.")
                raise

    VOLUME.commit()
