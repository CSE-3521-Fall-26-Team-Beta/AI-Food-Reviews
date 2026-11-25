'''Recreate the report figure with python scripts/plot_distilbert_loss.py.'''

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT = Path(__file__).resolve().parents[1] / 'results' / 'distilbert_loss_curves.png'
HISTORY = OUTPUT.with_name('distilbert_loss_history.csv')

def main():
  if not HISTORY.is_file():
    raise SystemExit('Generate loss history with python -m scripts.p6_distilbert --retrain')
  history = pd.read_csv(HISTORY)
  epochs = history['epoch']
  fig, ax = plt.subplots(figsize = (6.8, 3.8), layout = 'constrained')
  for losses, label, color, marker in [
    (history['training_loss'], 'Training', '#236396', 'o'),
    (history['validation_loss'], 'Validation', '#a44713', 's'),
  ]:
    ax.plot(epochs, losses, label = label, color = color, marker = marker, linewidth = 2.3)
  
  ax.set(
    xticks = epochs,
    xlabel = 'Epoch',
    ylabel = 'Weighted cross-entropy',
    title = 'DistilBERT + MLP training and validation loss'
  )
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis = 'y', alpha = 0.2)
  ax.legend(loc = 'lower left', frameon = False)
  OUTPUT.parent.mkdir(parents = True, exist_ok = True)
  fig.savefig(OUTPUT, dpi=200)
  plt.close(fig)
  print(OUTPUT)

if __name__ == '__main__':
  main()
