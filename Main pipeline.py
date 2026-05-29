# ============================================================
# Sports Action Recognition with Temporal Saliency
# and Squeeze-Excitation InceptionV3
# ============================================================

import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "UCF101_Selected_Classes"

OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 299
FRAMES_PER_VIDEO = 16
BATCH_SIZE = 8
EPOCHS = 50
RANDOM_STATE = 42

CLASSES = [
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CricketShot",
    "FloorGymnastics",
    "LongJump"
]

NUM_CLASSES = len(CLASSES)

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frames(video_path, frames_per_video=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    if total_frames <= 0:
        cap.release()
        return np.zeros((frames_per_video, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    frame_indices = np.linspace(0, total_frames - 1, frames_per_video).astype(int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

            # Median filtering
            frame = cv2.medianBlur(frame, 3)

        # Normalization
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()

    return np.array(frames, dtype=np.float32)


# ============================================================
# TEMPORAL SALIENCY EXTRACTION
# ============================================================

def compute_temporal_saliency(frames):
    saliency_frames = []

    prev_gray = None

    for frame in frames:
        gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        if prev_gray is None:
            saliency = np.zeros_like(gray, dtype=np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            saliency = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

        saliency = cv2.resize(saliency, (IMG_SIZE, IMG_SIZE))
        saliency = np.expand_dims(saliency, axis=-1)

        saliency_frame = frame * saliency
        saliency_frames.append(saliency_frame)

        prev_gray = gray

    return np.array(saliency_frames, dtype=np.float32)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    X = []
    y = []

    for class_index, class_name in enumerate(CLASSES):
        class_path = os.path.join(DATASET_DIR, class_name)

        if not os.path.exists(class_path):
            print(f"Missing folder: {class_path}")
            continue

        video_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".avi", ".mp4", ".mov", ".mkv"))
        ]

        for video_file in video_files:
            video_path = os.path.join(class_path, video_file)

            frames = extract_frames(video_path, FRAMES_PER_VIDEO)
            saliency_frames = compute_temporal_saliency(frames)

            video_feature = np.mean(saliency_frames, axis=0)

            X.append(video_feature)
            y.append(class_index)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    return X, y


# ============================================================
# SQUEEZE-EXCITATION BLOCK
# ============================================================

def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]

    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // ratio, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, filters))(se)

    output = layers.Multiply()([input_tensor, se])
    return output


# ============================================================
# BUILD SE-INCEPTIONV3 MODEL
# ============================================================

def build_model():
    base_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    for layer in base_model.layers[:-40]:
        layer.trainable = False

    x = base_model.output
    x = squeeze_excite_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# PLOT TRAINING CURVES
# ============================================================

def plot_training_curves(history):
    epochs = range(1, len(history.history["accuracy"]) + 1)

    plt.figure(figsize=(10, 7))
    plt.plot(epochs, history.history["accuracy"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], marker="s", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Figure_Model_Accuracy.png"), dpi=1000)
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(epochs, history.history["loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history.history["val_loss"], marker="s", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Loss")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Figure_Model_Loss.png"), dpi=1000)
    plt.show()


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
        os.path.join(TABLE_DIR, "Confusion_Matrix.csv")
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, CLASSES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASSES)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Figure_Confusion_Matrix.png"), dpi=1000)
    plt.show()


# ============================================================
# ROC CURVE
# ============================================================

def plot_roc_curve(y_true, y_prob):
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    plt.figure(figsize=(10, 7))

    roc_data = []

    for i, class_name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, linewidth=2, label=f"{class_name} AUC={roc_auc:.4f}")

        roc_data.append({
            "Class": class_name,
            "AUC": roc_auc
        })

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(fontsize=8)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Figure_ROC_Curve.png"), dpi=1000)
    plt.show()

    pd.DataFrame(roc_data).to_csv(os.path.join(TABLE_DIR, "ROC_AUC_Table.csv"), index=False)


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

def plot_precision_recall_curve(y_true, y_prob):
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    plt.figure(figsize=(10, 7))

    pr_data = []

    for i, class_name in enumerate(CLASSES):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, linewidth=2, label=f"{class_name} AUC={pr_auc:.4f}")

        pr_data.append({
            "Class": class_name,
            "PR_AUC": pr_auc
        })

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(fontsize=8)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Figure_Precision_Recall_Curve.png"), dpi=1000)
    plt.show()

    pd.DataFrame(pr_data).to_csv(os.path.join(TABLE_DIR, "Precision_Recall_AUC_Table.csv"), index=False)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def save_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy, precision, recall, f1],
        "Percentage": [accuracy * 100, precision * 100, recall * 100, f1 * 100]
    })

    metrics_df.to_csv(os.path.join(TABLE_DIR, "Performance_Metrics.csv"), index=False)

    print("\nPerformance Metrics")
    print(metrics_df)

    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(TABLE_DIR, "Classification_Report.csv"))

    plt.figure(figsize=(10, 7))
    plt.bar(metrics_df["Metric"], metrics_df["Percentage"])
    plt.xlabel("Metric")
    plt.ylabel("Score (%)")
    plt.title("Performance Metrics")

    for i, v in enumerate(metrics_df["Percentage"]):
        plt.text(i, v + 0.2, f"{v:.2f}%", ha="center", fontweight="bold")

    plt.ylim(90, 100)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Figure_Performance_Metrics.png"), dpi=1000)
    plt.show()


# ============================================================
# FPR AND FNR
# ============================================================

def save_fpr_fnr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fpr_list = []
    fnr_list = []

    for i in range(NUM_CLASSES):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - TP - FN - FP

        FPR = FP / (FP + TN + 1e-8)
        FNR = FN / (FN + TP + 1e-8)

        fpr_list.append(FPR)
        fnr_list.append(FNR)

    mean_fpr = np.mean(fpr_list)
    mean_fnr = np.mean(fnr_list)

    fpr_fnr_df = pd.DataFrame({
        "Metric": ["FPR", "FNR"],
        "Value": [mean_fpr, mean_fnr]
    })

    fpr_fnr_df.to_csv(os.path.join(TABLE_DIR, "FPR_FNR_Table.csv"), index=False)

    plt.figure(figsize=(8, 6))
    plt.bar(fpr_fnr_df["Metric"], fpr_fnr_df["Value"])
    plt.xlabel("Metric")
    plt.ylabel("Rate")
    plt.title("FPR vs FNR")

    for i, v in enumerate(fpr_fnr_df["Value"]):
        plt.text(i, v + 0.001, f"{v:.4f}", ha="center", fontweight="bold")

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "Figure_FPR_FNR.png"), dpi=1000)
    plt.show()


# ============================================================
# SAVE SOFTWARE VERSIONS
# ============================================================

def save_software_versions():
    versions = pd.DataFrame({
        "Software/Library": [
            "Python",
            "TensorFlow",
            "OpenCV",
            "NumPy",
            "Pandas",
            "Matplotlib",
            "Scikit-learn"
        ],
        "Version": [
            "3.x",
            tf.__version__,
            cv2.__version__,
            np.__version__,
            pd.__version__,
            plt.matplotlib.__version__,
            "Installed sklearn version"
        ]
    })

    versions.to_csv(os.path.join(TABLE_DIR, "Software_Configuration.csv"), index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading dataset...")
    X, y = load_dataset()

    print("Dataset shape:", X.shape)
    print("Labels shape:", y.shape)

    dataset_df = pd.DataFrame({
        "Class": [CLASSES[i] for i in y]
    })
    dataset_df["Class"].value_counts().to_csv(
        os.path.join(TABLE_DIR, "Dataset_Class_Distribution.csv")
    )

    y_cat = to_categorical(y, NUM_CLASSES)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_cat,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    split_df = pd.DataFrame({
        "Split": ["Training", "Testing"],
        "Samples": [len(X_train), len(X_test)]
    })
    split_df.to_csv(os.path.join(TABLE_DIR, "Train_Test_Split.csv"), index=False)

    print("Building model...")
    model = build_model()
    model.summary()

    model_path = os.path.join(MODEL_DIR, "SE_InceptionV3_Temporal_Saliency.h5")

    callbacks = [
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, mode="max"),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-7)
    ]

    print("Training model...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(TABLE_DIR, "Training_History.csv"), index=False)

    plot_training_curves(history)

    print("Evaluating model...")
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    save_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    save_fpr_fnr(y_true, y_pred)
    plot_roc_curve(y_true, y_prob)
    plot_precision_recall_curve(y_true, y_prob)
    save_software_versions()

    model.save(os.path.join(MODEL_DIR, "Final_SE_InceptionV3_Model.h5"))

    print("\nAll outputs saved successfully.")
    print("Figures saved in:", FIG_DIR)
    print("Tables saved in:", TABLE_DIR)
    print("Model saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()