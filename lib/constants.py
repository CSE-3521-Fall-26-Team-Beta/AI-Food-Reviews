from os.path import dirname

# Internal constants

_script_dir = dirname(__file__)

# File path constants

DATA_BIN_DIR = f'{_script_dir}/../data-bin'
ARCHIVE_ZIP_NAME = 'archive.zip'
REVIEWS_CSV_NAME = 'Reviews.csv'
REVIEWS_CSV_PATH = f'{DATA_BIN_DIR}/{REVIEWS_CSV_NAME}'
GLOVE_PATH = f'{DATA_BIN_DIR}/glove.6B.100d.txt'

RESULTS_DIR = f'{_script_dir}/../results'
DATASET_ANALYSIS_PATH = f'{RESULTS_DIR}/dataset_analysis.txt'
SCORE_DISTRIBUTION_PATH = f'{RESULTS_DIR}/score_distribution.png'
FEATURE_ENGINEERING_REPORT_PATH = f'{RESULTS_DIR}/feature_engineering_report.txt'

WORD_EMBEDDINGS_DIR = f'{RESULTS_DIR}/word_embeddings'
WORD_EMBEDDINGS_METRICS_PATH = f'{WORD_EMBEDDINGS_DIR}/metrics.csv'
WORD_EMBEDDINGS_CLASSIFICATION_REPORT_PATH = f'{WORD_EMBEDDINGS_DIR}/classification_report.csv'
WORD_EMBEDDINGS_CONFUSION_MATRIX_DATA_PATH = f'{WORD_EMBEDDINGS_DIR}/confusion_matrix.csv'
WORD_EMBEDDINGS_CONFUSION_MATRIX_IMAGE_PATH = f'{WORD_EMBEDDINGS_DIR}/confusion_matrix.png'
WORD_EMBEDDINGS_LABELED_WORDS_DIR = f'{WORD_EMBEDDINGS_DIR}/labeled_words'
WORD_EMBEDDINGS_WORD_CONTRIBUTIONS_IMAGE_PATH = f'{WORD_EMBEDDINGS_DIR}/word_contributions.png'
WORD_EMBEDDINGS_MISCLASSIFIED_REVIEWS_PATH = f'{WORD_EMBEDDINGS_DIR}/misclassified_reviews.csv'

# Seeding constants

TRAINING_SEED = 43
TRAINING_SAMPLE_SIZE = 50_000

# More file path constants

SAMPLE_CSV_NAME = f'group_sample_seed{TRAINING_SEED}.csv'
SAMPLE_CSV_PATH = f'{DATA_BIN_DIR}/{SAMPLE_CSV_NAME}'
