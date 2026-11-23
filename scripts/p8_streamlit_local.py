from subprocess import run

from lib.streamlit_opt_out import ensure_streamlit_opted_out

ensure_streamlit_opted_out()

run(['streamlit', 'run', 'app.py'], check = True, shell = True)
