from os import environ
from os.path import exists

OPT_OUT_MATCH_CONTENTS = '[browser]\ngatherUsageStats = false'
OPT_OUT_FULL_CONTENTS = f'{OPT_OUT_MATCH_CONTENTS}\n'

def ensure_streamlit_opted_out():
  if 'USERPROFILE' not in environ:
    return
  
  config_path = f'{environ['USERPROFILE']}/.streamlit/config.toml'
  
  if not exists(config_path):
    with open(config_path, 'w', newline = '') as f:
      f.write(OPT_OUT_FULL_CONTENTS)
  else:
    with open(config_path, 'r') as f:
      config_contents = f.read()
      opt_out_write_needed = OPT_OUT_MATCH_CONTENTS not in config_contents
    
    if opt_out_write_needed:
      with open(config_path, 'a') as f:
        f.write('\n' + OPT_OUT_FULL_CONTENTS)
