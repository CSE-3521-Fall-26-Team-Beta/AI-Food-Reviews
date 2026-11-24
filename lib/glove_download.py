from os.path import exists

from .constants import GLOVE_ARCHIVE_ZIP_NAME, GLOVE_NAME, GLOVE_PATH
from .kagglehub_helpers import kaggle_download_file

def ensure_results_csv_exists():
  '''
  Returns True if data needed to be downloaded by this function call,
  and returns False if data already present.
  '''
  
  if not exists(GLOVE_PATH):
    print(f'{GLOVE_NAME} not found, downloading...')
    
    #_kaggle_python_package_download_file(
    #  dataset_path = 'danielwillgeorge/glove6B100dtxt',
    #  archive_sub_file_name = GLOVE_NAME,
    #  output_file_path = GLOVE_PATH,
    #)
    
    kaggle_download_file(
      owner = 'danielwillgeorge',
      dataset_name = 'glove6B100dtxt',
      dataset_version = 1,
      archive_zip_name = GLOVE_ARCHIVE_ZIP_NAME,
      archive_sub_file_name = GLOVE_NAME,
      output_file_path = GLOVE_PATH,
    )
    
    return True
  else:
    return False
