# BUSA6015 - Deep Learning
# Fruit Quality Classification
# Stage 3: Data Preprocessing

from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split


# ==================================================
# 1. Configuration
# ==================================================

SEED = 42
IMG_SIZE = (64, 64)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ==================================================
# 2. Load metadata and identify original images
# ==================================================

df = pd.read_csv(
    OUTPUT_DIR / "image_metadata.csv"
)

# Extract filenames from file paths

df["filename"] = df["filepath"].map(
    lambda path: PureWindowsPath(path).name
)

# Identify the underlying source image

df["source_image"] = df["filename"].str.extract(
    r"(Screen Shot.*)$",
    expand=False
)

assert df["source_image"].notna().all(), \
    "Unrecognised filenames"


# Investigate source-image leakage

source_overlap = (
    df.groupby("source_image")["split"]
    .nunique()
    .gt(1)
    .sum()
)


# Retain only original images

originals = df[
    df["filename"] == df["source_image"]
].copy()

assert originals["source_image"].is_unique, \
    "Duplicate original source filenames"

assert set(originals["source_image"]) == set(df["source_image"])


# Display findings

print("Total downloaded images:", len(df))
print("Overlapping source-image groups:", source_overlap)
print("Original images:", len(originals))
print("Augmented copies excluded:",
      len(df) - len(originals))


# ==================================================
# 3. Create training, validation and test datasets
# ==================================================

# Retain original images from the supplied test set

test_df = originals[
    originals["split"] == "test"
].copy()


# Identify original training images

train_pool = originals[
    originals["split"] == "train"
].copy()


# Create stratified training and validation sets

train_df, val_df = train_test_split(
    train_pool,
    test_size=0.20,
    random_state=SEED,
    stratify=train_pool["class"]
)


# Assign final partition labels

train_df = train_df.assign(partition="train")
val_df = val_df.assign(partition="validation")
test_df = test_df.assign(partition="test")


# Combine partition metadata

partitioned_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)


# Verify source-image independence

assert (
    partitioned_df.groupby("source_image")["partition"]
    .nunique()
    .max() == 1
)

assert len(partitioned_df) == len(originals)


# Display partition sizes

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

print("\nFinal class distribution:")

print(pd.crosstab(
    partitioned_df["class"],
    partitioned_df["partition"]
))


# ==================================================
# 4. Encode class labels
# ==================================================

class_names = sorted(
    originals["class"].unique()
)

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}

print("\nClass encoding:")

for name, index in class_to_index.items():
    print(index, ":", name)


# ==================================================
# 5. Define image preprocessing function
# ==================================================

def prepare_images(dataframe):

    images = []
    labels = []

    for _, row in dataframe.iterrows():

        # Construct image path

        image_path = (
            DATA_DIR
            / row["split"]
            / row["class"]
            / row["filename"]
        )

        with Image.open(image_path) as img:

            # Handle transparency using a white background

            img = img.convert("RGBA")

            background = Image.new(
                "RGBA",
                img.size,
                (255, 255, 255, 255)
            )

            img = Image.alpha_composite(
                background, img
            ).convert("RGB")


            # Resize image

            img = img.resize(
                IMG_SIZE,
                Image.Resampling.BILINEAR
            )


            # Convert to numerical array and normalise

            img_array = (
                np.asarray(img, dtype=np.float32)
                / 255.0
            )

            images.append(img_array)


        # Assign numerical class label

        labels.append(
            class_to_index[row["class"]]
        )


    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int32)
    )


# ==================================================
# 6. Prepare training, validation and test arrays
# ==================================================

print("\nPreparing training images...")

X_train, y_train = prepare_images(train_df)

print("Preparing validation images...")

X_val, y_val = prepare_images(val_df)

print("Preparing testing images...")

X_test, y_test = prepare_images(test_df)


# Display tensor shapes

print("\nTraining:", X_train.shape, y_train.shape)

print("Validation:", X_val.shape, y_val.shape)

print("Testing:", X_test.shape, y_test.shape)


# Verify normalisation

print("\nMinimum training pixel value:", X_train.min())

print("Maximum training pixel value:", X_train.max())


# ==================================================
# 7. Visualise preprocessed training images
# ==================================================

fig, axes = plt.subplots(
    2, 3, figsize=(10, 7)
)

for index, ax in enumerate(axes.flatten()):

    image_index = np.where(
        y_train == index
    )[0][0]

    ax.imshow(X_train[image_index])

    ax.set_title(class_names[index])

    ax.axis("off")


fig.suptitle(
    "Preprocessed Fruit Images (64 x 64 RGB)"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "preprocessed_images.png",
    dpi=300
)


# ==================================================
# 8. Save preprocessed datasets
# ==================================================

np.savez_compressed(
    OUTPUT_DIR / "processed_data.npz",

    X_train=X_train,
    y_train=y_train,

    X_val=X_val,
    y_val=y_val,

    X_test=X_test,
    y_test=y_test,

    class_names=np.array(class_names)
)


# Save partition metadata for reproducibility

partitioned_df.to_csv(
    OUTPUT_DIR / "final_partitions.csv",
    index=False
)


print("\nPreprocessing completed successfully.")

print("Outputs saved to:", OUTPUT_DIR)


# Display the plot after saving all outputs

plt.show()