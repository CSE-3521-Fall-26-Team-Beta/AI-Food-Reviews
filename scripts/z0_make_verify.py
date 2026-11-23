from os import walk
from os.path import exists

from lib.constants import REPO_ROOT_DIR, VERIFICATION_HASH_FILE

if not exists(VERIFICATION_HASH_FILE):
  print('Creating verification hashes...')
  
  for dir_path, dir_names, file_names in walk(REPO_ROOT_DIR):
    if dir_path == REPO_ROOT_DIR:
      dir_names[:] = ['data-bin', 'results']
    print(dir_path)
  
  exit()
  with open(VERIFICATION_HASH_FILE, 'w') as f:
    pass
  
  print(f'Verification hashes saved to {VERIFICATION_HASH_FILE}')
else:
  print('Verification hashes already created')
