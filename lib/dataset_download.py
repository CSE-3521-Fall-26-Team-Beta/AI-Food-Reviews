from os.path import exists

from .constants import DATASET_ARCHIVE_ZIP_NAME, REVIEWS_CSV_NAME, REVIEWS_CSV_PATH
from .kagglehub_helpers import kaggle_download_file

def ensure_results_csv_exists():
  '''
  Returns True if data needed to be downloaded by this function call,
  and returns False if data already present.
  '''
  
  if not exists(REVIEWS_CSV_PATH):
    print(f'{REVIEWS_CSV_NAME} not found, downloading...')
    
    #_kaggle_python_package_download_file(
    #  dataset_path = 'danielwillgeorge/glove6B100dtxt',
    #  archive_sub_file_name = REVIEWS_CSV_NAME,
    #  output_file_path = REVIEWS_CSV_PATH,
    #)
    
    kaggle_download_file(
      owner = 'snap',
      dataset_name = 'amazon-fine-food-reviews',
      dataset_version = 2,
      archive_zip_name = DATASET_ARCHIVE_ZIP_NAME,
      archive_sub_file_name = REVIEWS_CSV_NAME,
      output_file_path = REVIEWS_CSV_PATH,
    )
    
    return True
  else:
    return False
