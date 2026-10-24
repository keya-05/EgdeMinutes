# Start EdgeMinutes with no internet access. Models must already be downloaded
# (run the app online once). Extra arguments are passed to streamlit, e.g.:
#   .\run_offline.ps1 --server.address 0.0.0.0
$env:HF_HUB_OFFLINE = "1"         # use only the local Hugging Face cache
$env:TRANSFORMERS_OFFLINE = "1"
Set-Location $PSScriptRoot
& .\venv\Scripts\python.exe -m streamlit run app.py @args
