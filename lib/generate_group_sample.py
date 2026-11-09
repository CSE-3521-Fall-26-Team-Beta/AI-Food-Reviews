from pandas import read_csv

from lib.dataset_download import ensure_results_csv_exists
from lib.constants import REVIEWS_CSV_PATH, SAMPLE_CSV_PATH, TRAINING_SAMPLE_SIZE, TRAINING_SEED

# Adapted from script in Canvas announcrements

ensure_results_csv_exists()
df = read_csv(REVIEWS_CSV_PATH)
sample = df.sample(n = TRAINING_SAMPLE_SIZE, random_state = TRAINING_SEED)

sample.to_csv(SAMPLE_CSV_PATH)

print(f'Saved {len(sample)} rows to {SAMPLE_CSV_PATH}')
print(sample[['Score', 'Text']].head(3))
