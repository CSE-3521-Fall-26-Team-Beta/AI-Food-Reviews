from os.path import dirname

# Internal constants
_script_dir = dirname(__file__)

# File path constants
DATA_BIN_DIR = f'{_script_dir}/../data-bin'
ARCHIVE_ZIP_NAME = 'archive.zip'
REVIEWS_CSV_NAME = 'Reviews.csv'
REVIEWS_CSV_PATH = f'{DATA_BIN_DIR}/{REVIEWS_CSV_NAME}'

# Seeding constants
TRAINING_SEED = 43
TRAINING_SAMPLE_SIZE = 50_000

# More file path constants
SAMPLE_CSV_NAME = f'group_sample_seed{TRAINING_SEED}.csv'
SAMPLE_CSV_PATH = f'{DATA_BIN_DIR}/{SAMPLE_CSV_NAME}'
