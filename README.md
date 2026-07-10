# JedgeNet Streamlit Demo

Four-class jujube surface-defect classification with the existing JedgeNet
PyTorch checkpoint. The app includes all 519 images from the held-out test
split and also supports user-uploaded images.

## Local run

Use Python 3.11 or 3.12 from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Run the inference tests with:

```powershell
python -m unittest discover -s tests -v
```

## Streamlit Community Cloud

1. Push this directory as the root of a GitHub repository.
2. In Streamlit Community Cloud, select that repository and branch.
3. Set the entrypoint to `app.py` and Python to 3.11 or 3.12.
4. Deploy. Future pushes to the selected branch trigger a rebuild.

