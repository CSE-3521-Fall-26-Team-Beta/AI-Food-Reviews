# https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory
# Answer: __file__ gives the path to the script file
# Can be found here: https://docs.python.org/3/reference/datamodel.html#module.__file__

def kaggle_default_download():
  import kagglehub
  from shutil import copyfile
  from os.path import dirname
  
  download_path = kagglehub.dataset_download('snap/amazon-fine-food-reviews')

  copyfile(f'{download_path}/Reviews.csv', f'{dirname(__file__)}/../Reviews.csv')

def kaggle_manual_download():
  import requests
  
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
      "User-Agent": "kaggle-api/v1.7.0",
      "Content-Type": "application/json",
    }
  )
  print(response.status_code)
  return
  print(response.text[0:40])

#kaggle_default_download()
kaggle_manual_download()
