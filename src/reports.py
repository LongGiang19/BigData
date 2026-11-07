"""Save plots without depending on notebook display()."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_ae_loss(ae, out):
    plt.figure(figsize=(5.5, 3.2))
    plt.plot(ae.loss_curve_)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (0.5*MSE)")
    plt.title("Autoencoder training loss")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_metrics(mdf, target, out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name in mdf.model.unique():
        s = mdf[(mdf.model == name) & (mdf.day != "ALL")]
        for axis, metric in zip(ax, ["pr_auc", "recall", "fpr"]):
            axis.plot(s.day, s[metric], marker="o", label=name)
    ax[0].set_title("PR-AUC by day")
    ax[1].set_title("Recall by day")
    ax[2].set_title("FPR by day")
    ax[2].axhline(target, ls="--", c="k", label="Target")
    for axis in ax:
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=.3)
    ax[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
