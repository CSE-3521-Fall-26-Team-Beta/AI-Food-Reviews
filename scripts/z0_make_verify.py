from os.path import exists

from lib.constants import VERIFICATION_HASH_FILE

if not exists(VERIFICATION_HASH_FILE):
  print('Creating verification hashes...')
  
  exit()
  with open(VERIFICATION_HASH_FILE, 'w') as f:
    pass
  
  print(f'Verification hashes saved to {VERIFICATION_HASH_FILE}')
else:
  print('Verification hashes already created')
