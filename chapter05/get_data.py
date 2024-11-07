
import os
import zipfile

from pathlib import Path
import requests

data_path = Path("data/")
image_path = data_path / "pizza_sushi_steak"

if image_path.is_dir():
    print(f"Directory: {image_path} exists.")
else:
    print(f"Directory: {image_path} no found, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
    
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)
        print("Done.")
    
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print(f"Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)
        print("Done.")
    os.remove(data_path / "pizza_steak_sushi.zip")
