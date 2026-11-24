'Downloads the review data. Run from the repository root with "python -m scripts.p4b_force_glove_download".'

from lib.constants import GLOVE_NAME
from lib.dataset_download import ensure_glove_exists

if not ensure_glove_exists():
  print(f'{GLOVE_NAME} already downloaded')
