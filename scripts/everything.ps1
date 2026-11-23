Set-Location "$PSScriptRoot/.."

python -m scripts.p0_download_data
python -m scripts.p1_generate_group_sample
python -m scripts.p2_analyze_dataset
