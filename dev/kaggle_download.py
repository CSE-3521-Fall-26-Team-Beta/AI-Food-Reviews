# https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory
# Answer: __file__ gives the path to the script file
# Can be found here: https://docs.python.org/3/reference/datamodel.html#module.__file__

from os.path import dirname
from os import makedirs

script_dir = dirname(__file__)
output_dir = f'{script_dir}/../data-bin'

def ensure_data_bin_exists():
  makedirs(output_dir, exist_ok = True)

def kaggle_default_download():
  import kagglehub
  from shutil import copyfile
  
  download_path = kagglehub.dataset_download('snap/amazon-fine-food-reviews')
  
  ensure_data_bin_exists()
  
  copyfile(f'{download_path}/Reviews.csv', f'{output_dir}/Reviews.csv')

def kaggle_manual_download():
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
  
  with open(f'{output_dir}/archive.zip', 'wb') as f:
    f.write(response.content)
  
  

#kaggle_default_download()
kaggle_manual_download()
