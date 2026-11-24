'Deploys the Streamlit configuration. Run from the repository root with "python -m scripts.p9_streamlit_deploy". This is intended to be run by repo contributors and the resultant Streamlit app made public.'

from lib.streamlit_opt_out import ensure_streamlit_opted_out

ensure_streamlit_opted_out()

# TODO
