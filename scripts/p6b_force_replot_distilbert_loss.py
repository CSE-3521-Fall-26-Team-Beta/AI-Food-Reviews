'''Ensure the DistilBERT loss graph exists when running the project pipeline by recreating the report figure from loss csv data. Run from the repository root with "python -m scripts.p6b_force_replot_distilbert_loss".'''

from lib.plot_distilbert_loss import create_loss_plot

if __name__ == '__main__':
  create_loss_plot()
