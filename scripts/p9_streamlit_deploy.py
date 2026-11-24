'Deploys the Streamlit configuration. Run from the repository root with "python -m scripts.p9_streamlit_deploy" (not intended to be run by people other than the repo contributors).'

from lib.streamlit_opt_out import ensure_streamlit_opted_out

ensure_streamlit_opted_out()

# TODO
