# BUSA6015 - Deep Learning
# Fruit Quality Classification
# Stage 5: Regularised CNN Development

from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import f1_score


# ==================================================
# 1. Configuration
# ==================================================

SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
PATIENCE = 5

tf.keras.utils.set_random_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==================================================
# 2. Load preprocessed data
# ==================================================

# Keep the test set untouched until final evaluation.
with np.load(OUTPUT_DIR / "processed_data.npz") as data:
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    class_names = data["class_names"]

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Classes:", class_names)


# ==================================================
# 3. Build regularised CNN
# ==================================================

model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),

    # Training-only data augmentation (inactive during validation/test).
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.05, fill_mode="reflect", seed=SEED + 1),

    # First convolutional block
    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=1,
        padding="valid",
        activation="relu"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Second convolutional block
    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=1,
        padding="valid",
        activation="relu"
    ),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Classification layers
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(DROPOUT_RATE, seed=SEED + 2),
    layers.Dense(len(class_names), activation="softmax")

], name="Regularised_CNN")

model.summary()


# ==================================================
# 4. Compile model
# ==================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==================================================
# 5. Train model
# ==================================================

# Save the model with the lowest validation loss.
# Stop when validation loss no longer improves.
checkpoint = callbacks.ModelCheckpoint(
    filepath=str(OUTPUT_DIR / "regularised_cnn.keras"),
    monitor="val_loss",
    mode="min",
    save_best_only=True
)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=PATIENCE,
    restore_best_weights=True
)

start_time = time.perf_counter()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

training_time = time.perf_counter() - start_time
print(f"\nTraining duration: {training_time:.2f} seconds")


# ==================================================
# 6. Save training history
# ==================================================

history_df = pd.DataFrame(history.history)
history_df.index += 1
history_df.index.name = "epoch"

history_df.to_csv(OUTPUT_DIR / "regularised_history.csv")


# ==================================================
# 7. Generate learning curves
# ==================================================

best_epoch = int(np.argmin(history_df["val_loss"].values)) + 1

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy curves
axes[0].plot(history_df.index, history_df["accuracy"],
             label="Training")
axes[0].plot(history_df.index, history_df["val_accuracy"],
             label="Validation")
axes[0].set_title("Regularised CNN Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Loss curves
axes[1].plot(history_df.index, history_df["loss"],
             label="Training")
axes[1].plot(history_df.index, history_df["val_loss"],
             label="Validation")
axes[1].set_title("Regularised CNN Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Cross-Entropy Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

# Identify the selected checkpoint on both graphs.
for ax in axes:
    ax.axvline(best_epoch, color="gray", linestyle="--",
               alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax.legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "regularised_learning_curves.png", dpi=300)


# ==================================================
# 8. Evaluate best model on validation set
# ==================================================

best_model = models.load_model(OUTPUT_DIR / "regularised_cnn.keras")

val_loss, val_accuracy = best_model.evaluate(
    X_val, y_val, verbose=0
)

# Generate class probabilities and predicted labels.
y_prob = best_model.predict(X_val, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

val_f1 = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)

# Training accuracy is measured with augmentation and dropout active.
train_accuracy = history_df.loc[best_epoch, "accuracy"]

print("\n===== REGULARISED CNN RESULTS =====")
print("Epochs completed:", len(history_df))
print("Best epoch:", best_epoch)
print(f"Training accuracy at best epoch: {train_accuracy:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation macro F1: {val_f1:.4f}")
print(f"Model parameters: {best_model.count_params():,}")


# ==================================================
# 9. Save performance metrics
# ==================================================

results = pd.DataFrame([{
    "model": "Regularised CNN",
    "best_epoch": best_epoch,
    "epochs_completed": len(history_df),
    "training_accuracy": train_accuracy,
    "validation_accuracy": val_accuracy,
    "validation_loss": val_loss,
    "validation_macro_f1": val_f1,
    "training_time_seconds": training_time,
    "parameters": best_model.count_params()
}])

results.to_csv(OUTPUT_DIR / "regularised_results.csv", index=False)

print("\nRegularised CNN training completed.")
print("Outputs saved to:", OUTPUT_DIR)

# Display graphs after saving all outputs.
plt.show()
