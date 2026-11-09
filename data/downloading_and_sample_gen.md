# How to download the dataset

First, ensure the required packages are installed: run `pip install requests>=2.34.2` in a terminal.

From the root directory of this GitHub repo, run `python -m data.download_data` in a terminal to download the `Reviews.csv` file. It will appear at `data-bin/Reviews.csv` (a gitignored location).

# How to generate group sample

First, ensure the required packages are installed: run `pip install requests>=2.34.2` and `pip install pandas>=3.1.1` in a terminal.

From the root directory of this GitHub repo, run `python -m data.generate_group_sample` in a terminal to generate the group sample (you do not have to download the dataset first, it will happen automatically). It will appear at `data-bin/group_sample_seed43.csv` (a gitignored location).

**NOTE: If sample constants are changed (like the sample size or seed), the file `data-bin/group_sample_seed43.csv` must be manually deleted so the sample is regenerated with new constants.**
