import os
from bing_image_downloader import downloader

# -----------------------------------
# Dataset Structure
# -----------------------------------

plants = {
    "rice": [
        "Rice Bacterial Leaf Blast disease",
        "Rice Brown Spot disease",
        "Healthy Rice leaf",
        "Rice Leaf Blast disease",
        "Rice Leaf Scald disease",
        "Rice Narrow Brown Spot disease"
    ],
    "corn": [
        "Corn Common Rust disease",
        "Corn Gray Leaf Spot disease",
        "Corn Blight disease",
        "Healthy Corn leaf"
    ],
    "tomato": [
        "Tomato Early Blight disease",
        "Tomato Late Blight disease",
        "Tomato Leaf Mold disease",
        "Tomato Septoria Leaf Spot disease",
        "Tomato Target Spot disease",
        "Tomato Mosaic Virus disease",
        "Healthy Tomato leaf"
    ],
    "potato": [
        "Potato Early Blight disease",
        "Potato Late Blight disease",
        "Healthy Potato leaf"
    ],
    "wheat": [
        "Wheat Leaf Rust disease",
        "Wheat Stripe Rust disease",
        "Healthy Wheat leaf"
    ]
}

# -----------------------------------
# Download Images
# -----------------------------------

BASE_DIR = "data"

os.makedirs(BASE_DIR, exist_ok=True)

for crop, diseases in plants.items():

    crop_folder = os.path.join(BASE_DIR, crop)
    os.makedirs(crop_folder, exist_ok=True)

    for disease in diseases:
        print(f"Downloading images for: {disease}")

        downloader.download(
            disease,
            limit=3,  # change number if you want more images
            output_dir=crop_folder,
            adult_filter_off=True,
            force_replace=False,
            timeout=60
        )

print("Dataset download completed successfully!")
