
# BUSA6015 - Deep Learning
# Fruit Quality Classification
# Exploratory Data Analysis

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from PIL import Image


# Reproducibility

SEED = 42
tf.keras.utils.set_random_seed(SEED)


# Environment verification

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))


# Locate dataset

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

for folder in DATA_DIR.rglob("*"):
    if folder.is_dir() and folder.name in ["train", "test"]:
        print("Dataset directory:", folder)

# Configure dataset directories

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Identify classification categories

class_names = sorted([
    folder.name
    for folder in TRAIN_DIR.iterdir()
    if folder.is_dir()
])

print("\nClasses:", class_names)
print("Number of classes:", len(class_names))


# Examine dataset and collect image metadata

import hashlib

records = []
invalid_images = []

for split, directory in [
    ("train", TRAIN_DIR),
    ("test", TEST_DIR)
]:

    for class_folder in sorted(directory.iterdir()):

        if not class_folder.is_dir():
            continue

        for image_path in class_folder.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in [
                ".jpg", ".jpeg", ".png", ".bmp"
            ]:
                continue

            try:
                with Image.open(image_path) as img:

                    img.load()

                    width, height = img.size
                    mode = img.mode

                file_hash = hashlib.sha256(
                    image_path.read_bytes()
                ).hexdigest()

                records.append({
                    "filepath": str(image_path),
                    "split": split,
                    "class": class_folder.name,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "hash": file_hash
                })

            except Exception as error:
                invalid_images.append({
                    "filepath": str(image_path),
                    "error": str(error)
                })

df = pd.DataFrame(records)

print("\nDataset preview:")
print(df.head())

print("\nTotal valid images:", len(df))
print("Invalid images:", len(invalid_images))

# Dataset distribution

distribution = pd.crosstab(
    df["class"],
    df["split"]
)

distribution["Total"] = distribution.sum(axis=1)

print("\nClass distribution:")
print(distribution)

print("\nDataset split:")
print(df["split"].value_counts())

print("\nClass percentages:")
print(
    df["class"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# Visualise class distribution

plt.figure(figsize=(11, 5))

sns.countplot(
    data=df,
    x="class",
    hue="split",
    order=class_names
)

plt.title("Fruit Image Distribution by Class")
plt.xlabel("Fruit Category")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)
plt.legend(title="Dataset")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "class_distribution.png",
    dpi=300
)

plt.show()



# Examine image dimensions

print("\nImage dimension statistics:")
print(df[["width", "height"]].describe())

print("\nMost common image dimensions:")
print(
    df.groupby(["width", "height"])
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print("\nColour modes:")
print(df["mode"].value_counts())

# Display a training image from each class

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

train_df = df[df["split"] == "train"]

for ax, class_name in zip(
    axes.flatten(),
    class_names
):

    sample = train_df[
        train_df["class"] == class_name
    ].sample(1, random_state=SEED)

    image_path = sample["filepath"].iloc[0]

    with Image.open(image_path) as img:
        ax.imshow(img.convert("RGB"))

    ax.set_title(class_name)
    ax.axis("off")

plt.suptitle("Sample Fruit Images by Class")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "sample_images.png",
    dpi=300
)

plt.show()

# Check for exact duplicate files

duplicates = df[
    df.duplicated("hash", keep=False)
]

print(
    "\nImages involved in exact duplicates:",
    len(duplicates)
)

# Check duplicates across training and testing

train_hashes = set(
    df[df["split"] == "train"]["hash"]
)

test_hashes = set(
    df[df["split"] == "test"]["hash"]
)

overlapping_hashes = train_hashes.intersection(
    test_hashes
)

print(
    "Identical file hashes across train and test:",
    len(overlapping_hashes)
)


# Save dataset metadata for preprocessing

df.to_csv(
    OUTPUT_DIR / "image_metadata.csv",
    index=False
)

print("\nExploratory data analysis completed.")
print("Outputs saved to:", OUTPUT_DIR)