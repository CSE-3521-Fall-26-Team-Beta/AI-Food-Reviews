# https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory
# Answer: __file__ gives the path to the script file

def kaggle_default_download():
  import kagglehub
  from shutil import copyfile
  from os.path import dirname
  
  download_path = kagglehub.dataset_download('snap/amazon-fine-food-reviews')

  copyfile(f'{download_path}/Reviews.csv', f'{dirname(__file__)}/../Reviews.csv')

kaggle_default_download()
