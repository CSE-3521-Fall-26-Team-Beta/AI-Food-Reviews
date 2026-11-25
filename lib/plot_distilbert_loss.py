'''Recreates the report figure.'''

#from os.path import exists
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lib.constants import DISTILBERT_LOSS_CURVES_PATH, DISTILBERT_LOSS_HISTORY_PATH

_OUTPUT = Path(DISTILBERT_LOSS_CURVES_PATH)
_HISTORY = Path(DISTILBERT_LOSS_HISTORY_PATH)

# Epoch-level weighted cross-entropy from the timed three-epoch benchmark.
# Keep the measurements here so a fresh checkout can reproduce the figure.
#EPOCHS = [1, 2, 3]
#TRAIN_LOSS = [0.4367828014693514, 0.3004200039649159, 0.20563389205886637]
#VAL_LOSS = [0.4028204496797125, 0.39416932374485, 0.4432000800590681]

def ensure_loss_plot_created():
  if _OUTPUT.is_file() and (
    not _HISTORY.is_file()
    or _OUTPUT.stat().st_mtime_ns >= _HISTORY.stat().st_mtime_ns
  ):
    return False
  
  if not _HISTORY.is_file():
    raise RuntimeError('Generate loss history with python -m scripts.p6_distilbert --retrain')
  
  print('Generating DistilBERT loss curves chart...')
  
  history = pd.read_csv(_HISTORY)
  epochs = history['epoch']
  fig, ax = plt.subplots(figsize = (6.8, 3.8), layout = 'constrained')
  for losses, label, color, marker in [
    (history['training_loss'], 'Training', '#236396', 'o'),
    (history['validation_loss'], 'Validation', '#a44713', 's'),
  ]:
    ax.plot(epochs, losses, label = label, color = color, marker = marker, linewidth = 2.3)
    for epoch, loss in zip(epochs, losses):
      offset = -16 if label == 'Validation' and epoch == 1 else 9
      ax.annotate(
        f'{loss:.3f}',
        (epoch, loss),
        xytext = (0, offset),
        textcoords = 'offset points',
        ha = 'center',
        color = color,
      )
  
  ax.set(
    xticks = epochs,
    xlim = (0.8, 3.2),
    ylim = (0.15, 0.5),
    xlabel = 'Epoch',
    ylabel = 'Weighted cross-entropy',
    title = 'DistilBERT + MLP training and validation loss',
  )
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis = 'y', alpha = 0.2)
  ax.legend(loc = 'lower left', frameon = False)
  _OUTPUT.parent.mkdir(parents = True, exist_ok = True)
  fig.savefig(_OUTPUT, dpi = 200)
  plt.close(fig)
  
  print(f'DistilBERT loss curves chart saved to {DISTILBERT_LOSS_CURVES_PATH}')
