from os.path import dirname

# Internal constants

# https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory
# Answer: __file__ gives the path to the script file
# Can be found here: https://docs.python.org/3/reference/datamodel.html#module.__file__
_script_dir = dirname(__file__)

# File path constants

REPO_ROOT_DIR = dirname(_script_dir)

DATA_BIN_DIR = f'{REPO_ROOT_DIR}/data-bin'
DATASET_ARCHIVE_ZIP_NAME = 'dataset-archive.zip'
REVIEWS_CSV_NAME = 'Reviews.csv'
REVIEWS_CSV_PATH = f'{DATA_BIN_DIR}/{REVIEWS_CSV_NAME}'
GLOVE_ARCHIVE_ZIP_NAME = 'glove-archive.zip'
GLOVE_NAME = 'glove.6B.100d.txt'
GLOVE_PATH = f'{DATA_BIN_DIR}/{GLOVE_NAME}'

RESULTS_DIR = f'{REPO_ROOT_DIR}/results'
DATASET_ANALYSIS_PATH = f'{RESULTS_DIR}/dataset_analysis.txt'
SCORE_DISTRIBUTION_PATH = f'{RESULTS_DIR}/score_distribution.png'
FEATURE_ENGINEERING_REPORT_PATH = f'{RESULTS_DIR}/feature_engineering_report.txt'
WORD_EMBEDDINGS_REPORT_PATH = f'{RESULTS_DIR}/word_embedding_report.txt'
LOGISTIC_REGRESSION_REPORT_PATH = f'{RESULTS_DIR}/logistic_regression_report.txt'
VERIFICATION_HASH_NAME = f'verification_hashes.txt'
VERIFICATION_HASH_PATH = f'{RESULTS_DIR}/{VERIFICATION_HASH_NAME}'

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
