import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc
import matplotlib.pyplot as plt



# create histograms of latent space for background and signal, for each latent dimension
# pooling is applied over the track dimension, so one latent vector per jet
def plot_latent_histograms_multiple_batches(loader, model,max_batches=None):

    model.eval()
    all_pooled = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            pooled = model(batch["x"].to("cuda"),batch["mask"].to("cuda"))
            all_pooled.append(pooled.cpu())

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    # Concatenate all batches
    pooled_all = torch.cat(all_pooled, dim=0)
    latent_bkg = pooled_all[::2]
    latent_sgn = pooled_all[1::2]
    latent_dim = pooled_all.shape[1]

    # Plot histograms
    n_cols = 8
    n_rows = (latent_dim + n_cols - 1) // n_cols
    _, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*2.5))
    axes = axes.flatten()

    for i in range(latent_dim):
        axes[i].hist(latent_bkg[:, i], bins=50, alpha=0.5, label="Bkg", color="blue", density=False)
        axes[i].hist(latent_sgn[:, i], bins=50, alpha=0.5, label="Sgn", color="red", density=False)
        if i == 0:
            axes[i].legend()

    # Hide extra axes
    for j in range(latent_dim, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()



# Plot ROC curve given FPR and TPR CSV files
# FPR and TPR obtained using fpr, tpr, thresholds = metrics.roc_curve(all_labels_int, all_pred)
def plot_roc(id: int, base_dir: str = "/home/morandsam/Desktop/internship_25/ROC/data"):

    # Load CSVs
    fpr = np.loadtxt(f"{base_dir}/fpr/fpr_{id}.csv", delimiter=",")
    tpr = np.loadtxt(f"{base_dir}/tpr/tpr_{id}.csv", delimiter=",")

    # Compute AUC
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random guess")

    # Labels and style
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save + show
    plt.tight_layout()
    plt.show()