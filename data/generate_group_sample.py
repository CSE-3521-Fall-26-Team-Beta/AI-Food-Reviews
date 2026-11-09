from pandas import read_csv

from lib.constants import REVIEWS_CSV_PATH, TRAINING_SAMPLE_SIZE, TRAINING_SEED

df = read_csv(REVIEWS_CSV_PATH)
sample = df.sample(n = TRAINING_SAMPLE_SIZE, random_state = TRAINING_SEED)

