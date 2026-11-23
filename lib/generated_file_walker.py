from os import walk

from .constants import REPO_ROOT_DIR

def walk_auto_generated_files():
  for dir_path, dir_names, file_names in walk(REPO_ROOT_DIR):
    if dir_path == REPO_ROOT_DIR:
      dir_names[:] = ['data-bin', 'results']
    else:
      for file_name in file_names:
        full_file_path = f'{dir_path}/{file_name}'
        root_relative_file_path = f'{full_file_path}'[len(REPO_ROOT_DIR) + 1 :].replace('\\', '/')
        
        yield full_file_path, root_relative_file_path
