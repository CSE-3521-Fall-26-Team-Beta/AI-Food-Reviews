'Generates the group sample. Run from the repository root with "python -m scripts.p1_generate_group_sample".'

from lib.constants import SAMPLE_CSV_NAME
from lib.generate_group_sample import ensure_sample_csv_exists

if not ensure_sample_csv_exists():
  print(f'Sample {SAMPLE_CSV_NAME} already generated')
