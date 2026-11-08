from os.path import dirname

# Internal constants
_script_dir = dirname(__file__)

# File path constants
DATA_BIN_DIR = f'{_script_dir}/../data-bin'
ARCHIVE_ZIP_NAME = 'archive.zip'
REVIEWS_CSV_NAME = 'Reviews.csv'
REVIEWS_CSV_PATH = f'{DATA_BIN_DIR}/{ARCHIVE_ZIP_NAME}'
