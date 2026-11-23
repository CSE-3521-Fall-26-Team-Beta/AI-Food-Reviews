from lib.constants import VERIFICATION_HASH_FILE
from lib.generated_file_walker import walk_auto_generated_files_with_hash

with open(VERIFICATION_HASH_FILE, 'r') as f:
  f.read().split('\n')
  raise NotImplementedError()

for _, root_relative_file_path, file_hash in walk_auto_generated_files_with_hash():
  raise NotImplementedError()
  print(f'{root_relative_file_path}: {file_hash}', file = verification_file)

print(f'Verification hashes saved to {VERIFICATION_HASH_FILE}')
