from io import StringIO
from os import makedirs
from traceback import print_exception
from zipfile import ZipFile

from .constants import DATA_BIN_DIR

def _ensure_data_bin_exists():
  makedirs(DATA_BIN_DIR, exist_ok = True)

def _extract_file_from_archive(archive_zip_name, archive_sub_file_name, output_file_path):
  print(f'Extracting {archive_sub_file_name} from downloaded {archive_zip_name}...')
  
  with ZipFile(f'{DATA_BIN_DIR}/{archive_zip_name}') as zip_archive:
    with zip_archive.open(archive_sub_file_name, 'r') as input_f:
      with open(output_file_path, 'wb') as output_f:
        output_f.write(input_f.read())

def _kaggle_python_package_download_file(dataset_path, archive_sub_file_name, output_file_path):
  print(f'Downloading {archive_sub_file_name} using kagglehub Python package...')
  
  from shutil import copyfile
  
  import kagglehub
  
  download_path = kagglehub.dataset_download(dataset_path)
  
  _ensure_data_bin_exists()
  
  copyfile(f'{download_path}/{archive_sub_file_name}', output_file_path)

def _kaggle_raw_api_download_archive(archive_zip_name, owner, dataset_name, dataset_version):
  print(f'Downloading {archive_zip_name} using kagglehub raw API...')
  
  import requests
  
  _ensure_data_bin_exists()
  
  response = requests.post(
    'https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset',
    json = {
      'ownerSlug': owner,
      'datasetSlug': dataset_name,
      'fileName': None,
      'datasetVersionNumber': dataset_version,
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
  
  with open(f'{DATA_BIN_DIR}/{archive_zip_name}', 'wb') as f:
    f.write(response.content)

def kaggle_download_file(owner, dataset_name, dataset_version, archive_zip_name, archive_sub_file_name, output_file_path):
  try:
    _extract_file_from_archive(archive_zip_name, archive_sub_file_name, output_file_path)
  except FileNotFoundError:
    print(f'{archive_zip_name} not found, downloading...')
    
    try:
      _kaggle_raw_api_download_archive(archive_zip_name, owner, dataset_name, dataset_version)
    except Exception as exc:
      error_file = StringIO()
      
      print_exception(exc, file = error_file)
      
      print(f'Downloading {archive_zip_name} using kagglehub raw API failed with exception:')
      
      print(error_file.getvalue())
      print()
      
      print(f'Attempting kagglehub Python package download of {archive_sub_file_name} instead...')
      _kaggle_python_package_download_file(f'{owner}/{dataset_name}', archive_sub_file_name, output_file_path)
      return
    
    _extract_file_from_archive(archive_zip_name, archive_sub_file_name, output_file_path)
