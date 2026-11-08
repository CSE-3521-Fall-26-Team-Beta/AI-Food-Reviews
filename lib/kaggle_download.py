# https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory
# Answer: __file__ gives the path to the script file
# Can be found here: https://docs.python.org/3/reference/datamodel.html#module.__file__

from os import makedirs
from os.path import exists
from zipfile import ZipFile

from constants import ARCHIVE_ZIP_NAME, DATA_BIN_DIR, REVIEWS_CSV_NAME, REVIEWS_CSV_PATH

def _ensure_data_bin_exists():
  makedirs(DATA_BIN_DIR, exist_ok = True)

def _extract_reviews_from_archive():
  print(f'Extracting {REVIEWS_CSV_NAME} from downloaded {ARCHIVE_ZIP_NAME}...')
  
  with ZipFile(f'{DATA_BIN_DIR}/{ARCHIVE_ZIP_NAME}') as zip_archive:
    with zip_archive.open(REVIEWS_CSV_NAME, 'r') as input_f:
      with open(f'{DATA_BIN_DIR}/{REVIEWS_CSV_NAME}', 'wb') as output_f:
        output_f.write(input_f.read())

def _kaggle_automatic_download_reviews():
  print(f'Downloading {REVIEWS_CSV_NAME} using kagglehub Python package...')
  
  import kagglehub
  from shutil import copyfile
  
  download_path = kagglehub.dataset_download('snap/amazon-fine-food-reviews')
  
  _ensure_data_bin_exists()
  
  copyfile(f'{download_path}/{REVIEWS_CSV_NAME}', f'{DATA_BIN_DIR}/{REVIEWS_CSV_NAME}')

def _kaggle_manual_download_archive():
  print(f'Downloading {ARCHIVE_ZIP_NAME} using kagglehub raw API...')
  
  import requests
  
  _ensure_data_bin_exists()
  
  response = requests.post(
    'https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset',
    json = {
      "ownerSlug": "snap",
      "datasetSlug": "amazon-fine-food-reviews",
      "fileName": None,
      "datasetVersionNumber": 2,
      "raw": False,
      "hashLink": None,
    },
    headers = {
      # User agent header is required, otherwise api will return a 404
      "User-Agent": "kaggle-api/v1.7.0",
      "Content-Type": "application/json",
    }
  )
  
  if response.status_code != 200:
    raise Exception(f'Download failed with status code: {response.status_code}')
  
  with open(f'{DATA_BIN_DIR}/{ARCHIVE_ZIP_NAME}', 'wb') as f:
    f.write(response.content)

def _kaggle_manual_download_reviews():
  try:
    _extract_reviews_from_archive()
  except FileNotFoundError:
    print(f'{ARCHIVE_ZIP_NAME} not found, downloading...')
    
    _kaggle_manual_download_archive()
    _extract_reviews_from_archive()

def ensure_results_csv_exists():
  if not exists(REVIEWS_CSV_PATH):
    print(f'{REVIEWS_CSV_NAME} not found, downloading...')
    
    #_kaggle_automatic_download_reviews()
    _kaggle_manual_download_reviews()

ensure_results_csv_exists()
