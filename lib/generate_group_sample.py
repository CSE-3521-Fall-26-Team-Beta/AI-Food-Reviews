from os.path import exists

from pandas import read_csv

from .constants import REVIEWS_CSV_PATH, SAMPLE_CSV_NAME, SAMPLE_CSV_PATH, TRAINING_SAMPLE_SIZE, TRAINING_SEED
from .dataset_download import ensure_results_csv_exists

# Adapted from script in Canvas announcements

def _generate_group_sample():
  ensure_results_csv_exists()
  
  print(f'Generating group sample {SAMPLE_CSV_NAME}...')
  
  df = read_csv(REVIEWS_CSV_PATH)
  sample = df.sample(n = TRAINING_SAMPLE_SIZE, random_state = TRAINING_SEED)
  
  sample.to_csv(SAMPLE_CSV_PATH)
  
  print(f'Saved {len(sample)} rows to {SAMPLE_CSV_PATH}')
  print(sample[['Score', 'Text']].head(3))

def ensure_sample_csv_exists():
  '''
  Returns True if sample needed to be downloaded by this function call,
  and returns False if sample already present.
  '''
  
  if not exists(SAMPLE_CSV_PATH):
    _generate_group_sample()
    return True
  else:
    return False
