'Checks all downloaded input files and result output files to make sure their hashes match the reference hash file results/verification_hashes.txt. Run from the repository root with "python -m scripts.p7_verify_hashes".'

from lib.constants import VERIFICATION_HASH_PATH
from lib.generated_file_walker import walk_auto_generated_files_with_hash

with open(VERIFICATION_HASH_PATH, 'r') as f:
  file_hashes = dict(map(lambda line: line.split(': '), f.read().strip().split('\n')))

error = False

for _, root_relative_file_path, file_hash in walk_auto_generated_files_with_hash():
  if root_relative_file_path not in file_hashes:
    print(f'ERROR: {root_relative_file_path}: file not present in verification list')
    error = True
  else:
    expected_hash = file_hashes[root_relative_file_path]
    
    if file_hash == expected_hash:
      print(f'Correct: {root_relative_file_path}: correct hash of {file_hash}')
    else:
      print(f'ERROR: {root_relative_file_path}: expected hash {expected_hash}, saw hash {file_hash}')
      error = True

if error:
  print('Errors in file verification')
  raise RuntimeError('Errors in file verification')
else:
  print('File verification successful')
