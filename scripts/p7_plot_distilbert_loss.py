"""Ensure the DistilBERT loss graph exists when running the project pipeline."""

from scripts import plot_distilbert_loss as plot


def main():
    if plot.OUTPUT.is_file() and (
        not plot.HISTORY.is_file()
        or plot.OUTPUT.stat().st_mtime_ns >= plot.HISTORY.stat().st_mtime_ns
    ):
        print("DistilBERT loss curves chart already created")
        return
    plot.main()


if __name__ == "__main__":
    main()
