from os.path import exists

from lib.constants import VERIFICATION_HASH_PATH
from lib.generated_file_walker import walk_auto_generated_files_with_hash

if not exists(VERIFICATION_HASH_PATH):
  print('Creating verification hashes...')
  
  with open(VERIFICATION_HASH_PATH, 'w', newline = '') as verification_file:
    for _, root_relative_file_path, file_hash in walk_auto_generated_files_with_hash():
      print(f'{root_relative_file_path}: {file_hash}', file = verification_file)
  
  print(f'Verification hashes saved to {VERIFICATION_HASH_PATH}')
else:
  print('Verification hashes already created')
