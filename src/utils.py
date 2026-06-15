import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import confusion_matrix

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


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
