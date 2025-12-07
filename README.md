# Telco CSV Processor

This small project provides utilities to load, clean, summarize, and prepare the Telco customer churn CSV for modeling.

Quick start

1. Create a virtual environment and install requirements:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run a summary:

```powershell
python cli.py summary "c:\Users\sreev\OneDrive\Attachments\Telco_Cusomer_Churn.csv"
```

3. Produce a cleaned CSV:

```powershell
python cli.py clean "c:\Users\sreev\OneDrive\Attachments\Telco_Cusomer_Churn.csv" --out cleaned_telco.csv
```

4. Inspect features:

```powershell
python cli.py features cleaned_telco.csv
```

Files

- `process_csv.py` — core functions (load, summarize, clean, prepare_features)
- `cli.py` — command line interface
- `requirements.txt` — Python dependencies
- `tests/` — basic unit tests

# run stream lit app

streamlit.run.app.py

open
