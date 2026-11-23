from base64 import b64encode
from hashlib import sha256
from os import walk
from os.path import exists

from lib.constants import REPO_ROOT_DIR, VERIFICATION_HASH_FILE

if not exists(VERIFICATION_HASH_FILE):
  print('Creating verification hashes...')
  
  with open(VERIFICATION_HASH_FILE, 'w') as verification_file:
    for dir_path, dir_names, file_names in walk(REPO_ROOT_DIR):
      if dir_path == REPO_ROOT_DIR:
        dir_names[:] = ['data-bin', 'results']
      else:
        for file_name in file_names:
          full_file_path = f'{dir_path}/{file_name}'
          root_relative_file_path = f'{full_file_path}'[len(REPO_ROOT_DIR) + 1 :].replace('\\', '/')
          
          with open(full_file_path, 'rb') as input_file:
            file_hash = b64encode(sha256(input_file.read()).digest()).decode('cp1252')
            print(f'{root_relative_file_path}: {file_hash}', file = verification_file)
  
  print(f'Verification hashes saved to {VERIFICATION_HASH_FILE}')
else:
  print('Verification hashes already created')
