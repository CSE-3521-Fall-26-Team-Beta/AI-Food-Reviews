from lib.constants import REVIEWS_CSV_NAME
from lib.dataset_download import ensure_results_csv_exists

if not ensure_results_csv_exists():
  print(f'{REVIEWS_CSV_NAME} already downloaded')
