from pathlib import Path
import json
import fal_client
import fire
import requests
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io

def process_png(png, transform_prompt, new_instruction):
    # Get original image size
    original_img = Image.open(png)
    original_size = original_img.size
    original_img.close()

    resp = fal_client.subscribe(
        "fal-ai/nano-banana/edit",
        arguments={
            "prompt": transform_prompt,
            "image_urls": [fal_client.upload_file(png)]
        },
    )
    generation_url = resp["images"][0]["url"]

    # Download the generated image
    response = requests.get(generation_url)
    generated_img = Image.open(io.BytesIO(response.content))

    # Resize to match original resolution
    resized_img = generated_img.resize(original_size, Image.LANCZOS)

    # Save the resized image
    resized_img.save(png)

    task_dir = png.parent
    base = png.stem
    json_same = task_dir / f"{base}.json"
    meta_path = json_same if json_same.exists() else None
    meta = {"instruction": new_instruction}
    with meta_path.open("w") as f:
        json.dump(meta, f)
    print(f"saved {png}")

def create_ood_images(root_dir, new_dir, transform_prompt, new_instruction, max_images=None):
    root_path = Path(root_dir).resolve()
    new_path = Path(new_dir).resolve()
    os.system(f"rm -rf {new_path} && cp -r {root_path} {new_path}")

    png_files = sorted(list(new_path.rglob("*.png")))
    if max_images is not None:
        png_files = png_files[:max_images]

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_png, png, transform_prompt, new_instruction): png
                  for png in png_files}

        for future in tqdm(as_completed(futures), total=len(png_files), desc="Processing images"):
            future.result()

if __name__ == "__main__":
    fire.Fire(create_ood_images)


