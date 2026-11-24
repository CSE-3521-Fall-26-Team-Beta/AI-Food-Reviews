"""Recreates the report figure. Run from the repository root with "python -m scripts.p7_plot_distilbert_loss"."""

from pathlib import Path

import matplotlib.pyplot as plt

# Epoch-level weighted cross-entropy from the timed three-epoch benchmark.
# Keep the measurements here so a fresh checkout can reproduce the figure.
EPOCHS = [1, 2, 3]
TRAIN_LOSS = [0.4367828014693514, 0.3004200039649159, 0.20563389205886637]
VAL_LOSS = [0.4028204496797125, 0.39416932374485, 0.4432000800590681]
OUTPUT = Path(__file__).resolve().parents[1] / "results" / "distilbert_loss_curves.png"

def main():
  fig, ax = plt.subplots(figsize=(6.8, 3.8), layout="constrained")
  for losses, label, color, marker in [
    (TRAIN_LOSS, "Training", "#236396", "o"),
    (VAL_LOSS, "Validation", "#a44713", "s"),
  ]:
    ax.plot(EPOCHS, losses, label=label, color=color, marker=marker, linewidth=2.3)
    for epoch, loss in zip(EPOCHS, losses):
      offset = -16 if label == "Validation" and epoch == 1 else 9
      ax.annotate(f"{loss:.3f}", (epoch, loss), xytext=(0, offset),
                  textcoords="offset points", ha="center", color=color)
  
  ax.set(xticks=EPOCHS, xlim=(0.8, 3.2), ylim=(0.15, 0.5), xlabel="Epoch",
          ylabel="Weighted cross-entropy", title="DistilBERT + MLP training and validation loss")
  ax.spines[["top", "right"]].set_visible(False)
  ax.grid(axis="y", alpha=0.2)
  ax.legend(loc="lower left", frameon=False)
  OUTPUT.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(OUTPUT, dpi=200)
  plt.close(fig)
  print(OUTPUT)

if __name__ == "__main__":
  main()
