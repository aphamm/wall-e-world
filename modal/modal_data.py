"""Download datasets to Modal volume.

Usage:
    # Download bridge dataset:
    modal run modal_data.py::download --dataset-name bridge

    # Download all datasets:
    modal run modal_data.py::download --dataset-name all

    # List available datasets:
    modal run modal_data.py::list_datasets
"""

import os

from modal_config import VOLUME, data_image, with_modal


@with_modal("world-model-data", timeout=6, cpu=16, img=data_image)
def download(dataset_name: str = "bridge", num_workers: int = 16):
    """Download and convert dataset to MP4+NPZ format.

    Args:
        dataset_name: Name of dataset to download (e.g., 'bridge', 'rt_1', or 'all')
        num_workers: Number of parallel workers for processing (default: 8)
    """
    from wall_e_world.download_data import main

    main(
        dataset_name=dataset_name,
        output_dir=os.environ["DATA_DIR"],
        num_workers=num_workers,
    )
    VOLUME.commit()


@with_modal("world-model-data", timeout=1, cpu=2, img=data_image)
def list_datasets():
    """List available datasets."""
    from wall_e_world.download_data import get_dataset_configs

    configs = get_dataset_configs("gs://gresearch/robotics")
    print("Available datasets:")
    for name in configs:
        print(f"  - {name}")
    return list(configs.keys())
