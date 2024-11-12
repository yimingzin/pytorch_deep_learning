import os
import requests
import zipfile
from pathlib import Path

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

if image_path.is_dir():
    print(f"Directory: {image_path} exists.")
else:
    print(f"Directory: {image_path} no found, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
    print("Done.")

    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        print("Downloading pizza, steak, sushi data...")
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        f.write(request.content)
        print("Done")

    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)
        print("Done")

    os.remove(data_path / "pizza_steak_sushi.zip")

def walk_through_data(dir_path):
    for dirpaths, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} files in {dirpaths}")

# walk_through_data(image_path)