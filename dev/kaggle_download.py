# https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory
# Answer: __file__ gives the path to the script file
print(__file__)

def kaggle_default_download():
  import kagglehub
  from shutil import copyfile
  from os.path import dirname
  
  download_path = kagglehub.dataset_download('snap/amazon-fine-food-reviews')

  copyfile(f'{download_path}/Reviews.csv', '')



#kaggle_default_download()
