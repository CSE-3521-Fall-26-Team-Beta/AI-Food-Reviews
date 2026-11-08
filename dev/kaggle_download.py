def kaggle_default_download():
  import kagglehub
  
  download_path = kagglehub.dataset_download('snap/amazon-fine-food-reviews')
  
  print(download_path)

kaggle_default_download()
