'''Ensure the DistilBERT loss graph exists when running the project pipeline by recreating the report figure from loss csv data. Run from the repository root with "python -m scripts.p6b_plot_distilbert_loss".'''

from lib.plot_distilbert_loss import ensure_loss_plot_created

if __name__ == '__main__':
  if not ensure_loss_plot_created():
    print('DistilBERT loss curves chart already created')
