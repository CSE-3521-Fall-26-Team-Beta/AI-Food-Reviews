# https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory
# Answer: __file__ gives the path to the script file
# Can be found here: https://docs.python.org/3/reference/datamodel.html#module.__file__

from io import StringIO
from os import makedirs
from os.path import exists
from traceback import print_exception

from .constants import DATA_BIN_DIR, GLOVE_NAME, GLOVE_PATH

def _ensure_data_bin_exists():
  makedirs(DATA_BIN_DIR, exist_ok = True)

def _kaggle_python_package_download_glove():
  print(f'Downloading {GLOVE_NAME} using kagglehub Python package...')
  
  from shutil import copyfile
  
  import kagglehub
  
  download_path = kagglehub.dataset_download('danielwillgeorge/glove6B100dtxt')
  
  _ensure_data_bin_exists()
  
  copyfile(f'{download_path}/{GLOVE_NAME}', GLOVE_PATH)

def _kaggle_raw_api_download_glove():
  print(f'Downloading {GLOVE_NAME} using kagglehub raw API...')
  
  import requests
  
  _ensure_data_bin_exists()
  
  response = requests.post(
    'https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset',
    json = {
      'ownerSlug': 'danielwillgeorge',
      'datasetSlug': 'glove6B100dtxt',
      'fileName': None,
      'datasetVersionNumber': 1,
      'raw': False,
      'hashLink': None,
    },
    headers = {
      # User agent header is required, otherwise api will return a 404
      'User-Agent': 'kaggle-api/v1.7.0',
      'Content-Type': 'application/json',
    }
  )
  
  if response.status_code != 200:
    raise Exception(f'Download failed with status code: {response.status_code}')
  
  with open(f'{GLOVE_PATH}', 'wb') as f:
    f.write(response.content)

def _kaggle_download_glove():
  try:
    _kaggle_raw_api_download_glove()
  except Exception as exc:
    error_file = StringIO()
    
    print_exception(exc, file = error_file)
    
    print(f'Downloading {GLOVE_NAME} using kagglehub raw API failed with exception:')
    
    print(error_file.getvalue())
    print()
    
    print(f'Attempting kagglehub Python package download of {GLOVE_NAME} instead...')
    _kaggle_python_package_download_glove()
    return

def ensure_glove_exists():
  '''
  Returns True if data needed to be downloaded by this function call,
  and returns False if data already present.
  '''
  
  if not exists(GLOVE_PATH):
    print(f'{GLOVE_NAME} not found, downloading...')
    
    #_kaggle_python_package_download_glove()
    _kaggle_download_glove()
    return True
  else:
    return False
