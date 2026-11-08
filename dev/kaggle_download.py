# https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory
# Answer: __file__ gives the path to the script file
# Can be found here: https://docs.python.org/3/reference/datamodel.html#module.__file__

from os import makedirs
from os.path import dirname
from zipfile import ZipFile

_script_dir = dirname(__file__)
_download_dir = f'{_script_dir}/../data-bin'
ARCHIVE_ZIP_NAME = 'archive.zip'
REVIEWS_CSV_NAME = 'Reviews.csv'

def ensure_data_bin_exists():
  makedirs(_download_dir, exist_ok = True)

def extract_reviews_from_archive():
  print(f'Extracting {REVIEWS_CSV_NAME} from downloaded {ARCHIVE_ZIP_NAME}...')
  
  with ZipFile(f'{_download_dir}/{ARCHIVE_ZIP_NAME}') as zip_archive:
    with zip_archive.open(REVIEWS_CSV_NAME, 'rb') as input_f:
      with open(f'{_download_dir}/{REVIEWS_CSV_NAME}', 'wb') as output_f:
        output_f.write(input_f.read())

def kaggle_automatic_download_reviews():
  print(f'Downloading {REVIEWS_CSV_NAME} using kagglehub Python package...')
  
  import kagglehub
  from shutil import copyfile
  
  download_path = kagglehub.dataset_download('snap/amazon-fine-food-reviews')
  
  ensure_data_bin_exists()
  
  copyfile(f'{download_path}/{REVIEWS_CSV_NAME}', f'{_download_dir}/{REVIEWS_CSV_NAME}')

def kaggle_manual_download_archive():
  print(f'Downloading {ARCHIVE_ZIP_NAME} using kagglehub raw API...')
  
  import requests
  
  ensure_data_bin_exists()
  
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
  
  with open(f'{_download_dir}/{ARCHIVE_ZIP_NAME}', 'wb') as f:
    f.write(response.content)

def kaggle_manual_download_reviews():
  try:
    extract_reviews_from_archive()
  except FileNotFoundError:
    print(f'{ARCHIVE_ZIP_NAME} not found, downloading...')
    kaggle_manual_download_archive()
    extract_reviews_from_archive()

def ensure_results_csv_exists():
  #kaggle_automatic_download_reviews()
  kaggle_manual_download_reviews()

ensure_results_csv_exists()
