from base64 import b64encode
from hashlib import sha256
from os import walk

from .constants import REPO_ROOT_DIR, RESULTS_DIR, VERIFICATION_HASH_NAME

def walk_auto_generated_files():
  for dir_path, dir_names, file_names in walk(REPO_ROOT_DIR):
    # Trim root dir walk to only contiain data-bin and results directories
    if dir_path == REPO_ROOT_DIR:
      dir_names[:] = ['data-bin', 'results']
    else:
      # Trim results dir walk to not contain verification hash file
      if dir_path.replace('\\', '/') == RESULTS_DIR.replace('\\', '/'):
        del dir_names[dir_names.index(VERIFICATION_HASH_NAME)]
      
      for file_name in file_names:
        full_file_path = f'{dir_path}/{file_name}'
        root_relative_file_path = f'{full_file_path}'[len(REPO_ROOT_DIR) + 1 :].replace('\\', '/')
        
        yield full_file_path, root_relative_file_path

def walk_auto_generated_files_with_hash():
  for full_file_path, root_relative_file_path in walk_auto_generated_files():
    with open(full_file_path, 'rb') as input_file:
      file_hash = b64encode(sha256(input_file.read()).digest()).decode('cp1252')
      yield full_file_path, root_relative_file_path, file_hash
