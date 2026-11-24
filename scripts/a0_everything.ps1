# Runs all main non-Streamlit scripts to ensure all output files are created. Run from the repository root with "./scripts/a0_everything.ps1".

Set-Location "$PSScriptRoot/.."

python -m scripts.p0_download_data
python -m scripts.p1_generate_group_sample
python -m scripts.p2_analyze_dataset
python -m scripts.p3_feature_engineering
python -m scripts.p4_word_embeddings
python -m scripts.p5_logistic_regression
python -m scripts.p7_verify_hashes
