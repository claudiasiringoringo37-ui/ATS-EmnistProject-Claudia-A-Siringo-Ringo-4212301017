# main.py
"""
EMNIST Letters Classification (HOG + SVM)
Evaluasi: 13.000 data (26 kelas × 500 sample per kelas)
Metode: Leave-One-Out Cross Validation (LOOCV)
Output: Folder 'result' + GUI Window
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage import exposure
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


# =========================================================
# 1. Load & Sampling Data (26 kelas × 500 sample)
# =========================================================
def load_and_sample_data(filepath, n_per_class=500, seed=42):
    print("Loading dataset...")
    df = pd.read_csv(filepath)

    # Pastikan kolom pertama adalah label
    df.columns = ["label"] + [f"pixel{i}" for i in range(1, len(df.columns))]

    np.random.seed(seed)
    # Ambil 500 sample per kelas agar seimbang
    sampled_df = (
        df.groupby("label", group_keys=False)
          .apply(lambda x: x.sample(n=n_per_class, random_state=seed))
          .reset_index(drop=True)
    )

    labels = sampled_df["label"].values
    pixels = sampled_df.iloc[:, 1:].values.reshape(-1, 28, 28)

    print(f"Total samples: {len(labels)} (26 classes × {n_per_class} = {26 * n_per_class})")
    print(f"Distribusi kelas:\n{sampled_df['label'].value_counts().sort_index()}")
    return pixels, labels


# =========================================================
# 2. Ekstraksi Fitur HOG
# =========================================================
def extract_hog_features(images):
    print("\nExtracting HOG features...")
    features = []
    for img in tqdm(images):
        img_norm = img / 255.0
        fd = hog(img_norm, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False, channel_axis=None)
        features.append(fd)
    return np.array(features)


def visualize_hog(image, out_path="result/hog_visualization.png"):
    img_norm = image / 255.0
    fd, hog_img = hog(img_norm, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, channel_axis=None)
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, hog_img.max()))
    os.makedirs("result", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(hog_img_rescaled, cmap='gray')
    plt.axis('off')
    plt.title("HOG Visualization")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =========================================================
# 3. Evaluasi Model (Leave-One-Out Cross Validation)
# =========================================================
def loocv_evaluation(X, y, svm_params):
    print(f"\nStarting Leave-One-Out Cross Validation (Total {len(X)} samples)...")
    loo = LeaveOneOut()
    y_true, y_pred = [], []

    start = time.time()

    for train_idx, test_idx in tqdm(loo.split(X), total=len(X), desc="LOOCV Progress", ncols=80):
        model = SVC(**svm_params)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        y_true.append(y[test_idx][0])
        y_pred.append(pred[0])

    elapsed = time.time() - start
    print(f"\nLOOCV completed in {elapsed/60:.2f} minutes.")
    return np.array(y_true), np.array(y_pred)


# =========================================================
# 4. Simpan & Tampilkan Hasil
# =========================================================
def save_results(y_true, y_pred, out_dir="result"):
    os.makedirs(out_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    pd.DataFrame([{"Accuracy": acc, "Precision": prec, "F1_Score": f1}]).to_csv(
        os.path.join(out_dir, "result_summary.csv"), index=False
    )
    pd.DataFrame({"True_Label": y_true, "Predicted_Label": y_pred}).to_csv(
        os.path.join(out_dir, "result_detail.csv"), index=False
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix - EMNIST (HOG + SVM, 13.000 samples, LOOCV)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    conf_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(conf_path, dpi=150)
    plt.close()

    print(f"\nAccuracy : {acc:.4f}\nPrecision: {prec:.4f}\nF1-Score : {f1:.4f}")
    return acc, prec, f1, conf_path


def show_results_window(acc, prec, f1, conf_img_path):
    root = tk.Tk()
    root.title("Hasil Evaluasi EMNIST Letters (13.000 Sample - LOOCV)")
    root.geometry("700x800")

    ttk.Label(root, text="HASIL KLASIFIKASI EMNIST LETTERS (13.000 Sample - LOOCV)",
              font=("Arial", 14, "bold")).pack(pady=10)

    ttk.Label(root, text=f"Accuracy : {acc:.4f}", font=("Arial", 12)).pack(pady=5)
    ttk.Label(root, text=f"Precision: {prec:.4f}", font=("Arial", 12)).pack(pady=5)
    ttk.Label(root, text=f"F1-Score : {f1:.4f}", font=("Arial", 12)).pack(pady=5)
    ttk.Separator(root, orient="horizontal").pack(fill="x", pady=10)

    img = Image.open(conf_img_path).resize((650, 650))
    img_tk = ImageTk.PhotoImage(img)
    lbl = ttk.Label(root, image=img_tk)
    lbl.image = img_tk
    lbl.pack(pady=10)

    ttk.Button(root, text="Tutup", command=root.destroy).pack(pady=10)
    root.mainloop()


# =========================================================
# 5. Main Program
# =========================================================
def main():
    csv_path = "emnist-letters-train.csv"

    print("="*70)
    print("EMNIST LETTERS CLASSIFICATION - 13.000 DATA (LOOCV)")
    print("Metode: HOG Feature + SVM (RBF) + Leave-One-Out Cross Validation")
    print("="*70)

    # 1. Load data seimbang (26 kelas × 500)
    X, y = load_and_sample_data(csv_path, n_per_class=500)

    # 2. Visualisasi contoh huruf
    os.makedirs("result", exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i], cmap="gray")
        ax.set_title(chr(64 + y[i]) if 1 <= y[i] <= 26 else str(y[i]))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("result/sample_images.png", dpi=150)
    plt.close()

    # 3. Visualisasi HOG satu contoh  
    visualize_hog(X[0], "result/hog_visualization.pn    n       g")

    # 4. Ekstraksi fitur HOG
    X_feat = extract_hog_features(X)
    X_feat = StandardScaler().fit_transform(X_feat)
  
    # 5. Evaluasi model dengan LOOCV     
    svm_params = {"kernel": "rbf", "C": 1.0, "gamma": "scale"}
    y_true, y_pred = loocv_evaluation(X_feat, y, svm_params) 

    # 6. Simpan hasil & tampilkan GUI
    acc, prec, f1, conf_path = save_results(y_true, y_pred)
    show_results_window(acc, prec, f1, conf_path)


if __name__ == "__main__":
    main()
