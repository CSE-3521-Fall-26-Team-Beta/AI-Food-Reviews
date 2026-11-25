'''Recreate the report figure with python scripts/plot_distilbert_loss.py.'''

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lib.constants import DISTILBERT_LOSS_CURVES_PATH, DISTILBERT_LOSS_HISTORY_PATH

_OUTPUT = Path(DISTILBERT_LOSS_CURVES_PATH)
_HISTORY = Path(DISTILBERT_LOSS_HISTORY_PATH)

def create_loss_plot():
  if _OUTPUT.is_file() and (
    not _HISTORY.is_file()
    or _OUTPUT.stat().st_mtime_ns >= _HISTORY.stat().st_mtime_ns
  ):
    raise RuntimeError('DistilBERT loss curves chart already created')
  
  if not _HISTORY.is_file():
    raise RuntimeError('Generate loss history with python -m scripts.p6_distilbert --retrain')
  
  history = pd.read_csv(_HISTORY)
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
  _OUTPUT.parent.mkdir(parents = True, exist_ok = True)
  fig.savefig(_OUTPUT, dpi=200)
  plt.close(fig)
  print(_OUTPUT)

if __name__ == '__main__':
  create_loss_plot()
