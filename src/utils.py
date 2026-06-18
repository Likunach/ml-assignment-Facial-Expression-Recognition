import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from sklearn.metrics import accuracy_score, confusion_matrix

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def get_device(prefer_cuda=True):
    """Return a CUDA device if available and working, else CPU."""
    if not prefer_cuda or not torch.cuda.is_available():
        return "cpu"

    try:
        torch.zeros(1, device="cuda").add_(1)
        return "cuda"
    except Exception as exc:
        print(
            "CUDA is available but not compatible with this PyTorch build.\n"
            f"Falling back to CPU. ({exc})\n"
            "For RTX 50-series GPUs, reinstall PyTorch with CUDA 12.8:\n"
            "  pip uninstall torch torchvision -y\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        )
        return "cpu"


def generate_run_name(arch, optimizer, lr, batch_size, aug, dropout, weight_decay=0, extra=""):
    wd_str = f"_wd{weight_decay}" if weight_decay > 0 else ""
    extra_str = f"_{extra}" if extra else ""
    return f"{arch}_{optimizer}_{lr}_bs{batch_size}_{aug}_do{dropout}{wd_str}{extra_str}"


def log_confusion_matrix(all_labels, all_preds, run_name):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {run_name}")
    plt.tight_layout()
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0


WANDB_ENTITY = "lchit22-free-university-of-tbilisi-"
WANDB_PROJECT = "fer2013-expression-recognition"


def load_wandb_results(group=None, run_prefix=None, only_finished=True):
    """Load finished run summaries from Wandb."""

    api = wandb.Api()

    filters = {}

    if group is not None:
        filters["group"] = group

    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters=filters
    )

    all_results = []

    for run in runs:
        if only_finished and run.state != "finished":
            continue

        if run_prefix is not None and not run.name.startswith(run_prefix):
            continue

        best_val_acc = run.summary.get(
            "best_val_acc",
            run.summary.get("val_acc", 0)
        )

        all_results.append({
            "run": run.name,
            "state": run.state,
            "group": run.group,
            "best_val_acc": best_val_acc
        })

    all_results = sorted(
        all_results,
        key=lambda x: x["best_val_acc"],
        reverse=True
    )

    print(f"Loaded {len(all_results)} runs from Wandb")

    for r in all_results:
        print(
            f'{r["run"]} | group={r["group"]} | '
            f'state={r["state"]} | best_val_acc={r["best_val_acc"]:.4f}'
        )

    return all_results


def _download_checkpoint_from_wandb(run_name, checkpoint_dir):
    """Try to download a saved checkpoint from a finished Wandb run."""

    api = wandb.Api()
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"display_name": run_name},
    )

    checkpoint_name = f"{run_name}_best.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for run in runs:
        if run.name != run_name:
            continue

        for run_file in run.files():
            if run_file.name.endswith(checkpoint_name) or run_file.name.endswith("_best.pt"):
                print(f"Downloading checkpoint from Wandb run: {run.name}")
                run_file.download(root=checkpoint_dir, replace=True)
                if os.path.exists(checkpoint_path):
                    return checkpoint_path

                downloaded_path = os.path.join(checkpoint_dir, run_file.name)
                if os.path.exists(downloaded_path) and downloaded_path != checkpoint_path:
                    os.replace(downloaded_path, checkpoint_path)
                    return checkpoint_path

    return checkpoint_path


def log_tinycnn_confusion_matrix(
    model_class,
    get_dataloaders_fn,
    all_results,
    checkpoint_dir="checkpoints",
    data_dir="data",
    wandb_dir=None,
    group="tiny",
    device=None,
):
    """Log TinyCNN confusion matrix charts to Wandb."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if wandb_dir is None:
        wandb_dir = os.path.join(os.path.expanduser("~"), "wandb_logs", "fer2013")
        os.makedirs(wandb_dir, exist_ok=True)

    os.makedirs(checkpoint_dir, exist_ok=True)

    results_df = pd.DataFrame(all_results).sort_values(
        "best_val_acc",
        ascending=False
    )

    if results_df.empty:
        raise ValueError("all_results is empty. Load Wandb results first.")

    best_run_name = results_df.iloc[0]["run"]
    best_val_acc = results_df.iloc[0]["best_val_acc"]

    print("Best TinyCNN run:", best_run_name)
    print("Best val_acc from Wandb:", best_val_acc)

    if "do0.5" in best_run_name:
        dropout = 0.5
    elif "do0.25" in best_run_name:
        dropout = 0.25
    else:
        dropout = 0.0

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{best_run_name}_best.pt"
    )

    print("Checkpoint path:", checkpoint_path)

    if not os.path.exists(checkpoint_path):
        checkpoint_path = _download_checkpoint_from_wandb(
            best_run_name,
            checkpoint_dir,
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}\n\n"
            "Chart 6 needs the trained model weights (.pt file).\n"
            "Run the training cell first, or place the checkpoint in checkpoints/:\n"
            f"  {best_run_name}_best.pt"
        )

    _, val_loader, _ = get_dataloaders_fn(
        data_dir=data_dir,
        aug_mode="none",
        batch_size=128
    )

    model = model_class(dropout=dropout).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    computed_val_accuracy = accuracy_score(all_labels, all_preds)

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(7))
    )

    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_percent = np.nan_to_num(cm_percent)

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(cm_percent, interpolation="nearest")

    ax.set_title(
        f"TinyCNN — Confusion Matrix\nBest run: {best_run_name}",
        fontsize=13,
        pad=15
    )

    ax.set_xlabel("Predicted Emotion", fontsize=11)
    ax.set_ylabel("Actual Emotion", fontsize=11)

    ax.set_xticks(np.arange(len(EMOTION_LABELS)))
    ax.set_yticks(np.arange(len(EMOTION_LABELS)))

    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(EMOTION_LABELS)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percent = cm_percent[i, j] * 100
            count = cm[i, j]

            text_color = "white" if cm_percent[i, j] > 0.45 else "black"

            ax.text(
                j,
                i,
                f"{percent:.1f}%\n({count})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Percentage within actual class")

    fig.text(
        0.5,
        0.01,
        f"Validation accuracy: {computed_val_accuracy:.4f} | Each cell shows percentage and raw count",
        ha="center",
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    save_path = os.path.join(
        checkpoint_dir,
        "tinycnn_confusion_matrix_chart.png"
    )

    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    wandb.log({
        "best_run_name": best_run_name,
        "best_val_acc": best_val_acc,
        "computed_val_accuracy": computed_val_accuracy,
        "tinycnn_confusion_matrix_chart": wandb.Image(fig)
    })

    plt.show()

    print("Confusion matrix chart saved to:", save_path)

    return save_path
