from base64 import b64encode
from hashlib import sha256
from os.path import exists

from lib.constants import VERIFICATION_HASH_FILE
from lib.generated_file_walker import walk_auto_generated_files

if not exists(VERIFICATION_HASH_FILE):
  print('Creating verification hashes...')
  
  with open(VERIFICATION_HASH_FILE, 'w') as verification_file:
    for full_file_path, root_relative_file_path in walk_auto_generated_files():
      with open(full_file_path, 'rb') as input_file:
        file_hash = b64encode(sha256(input_file.read()).digest()).decode('cp1252')
        print(f'{root_relative_file_path}: {file_hash}', file = verification_file)
  
  print(f'Verification hashes saved to {VERIFICATION_HASH_FILE}')
else:
  print('Verification hashes already created')
