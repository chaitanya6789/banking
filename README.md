# 🛡️ Fraud Drift Guard

Streamlit deployment of **"An Event-Driven Feature-Partitioned Heterogeneous Sub-Ensemble
Framework for Selective Concept Drift Mitigation in Financial Fraud Detection"**
(SRM IST · CSE-AIML · Team *Fraud Detection Banking*).

Three sub-models (Random Forest · XGBoost · Decision Tree) each see one feature partition,
vote with weights 0.30 / 0.45 / 0.25, and a PSI monitor retrains **only the sub-model whose
partition drifted**.

## App tabs
| Tab | What you can do |
|---|---|
| Overview | Idea + architecture diagram |
| Data | Class imbalance, amount distribution, 60/40 split |
| Models | Per-model settings, train time, baseline metrics, feature importance |
| Predict | Score one transaction (editable what-if) or upload a CSV and download results |
| Drift Monitor | Inject drift (scale / shift / noise), PSI per partition and per feature |
| Strategies | A static vs B global vs C selective: metrics, time saved, confusion matrices |
| Live Stream | Windowed stream simulation with PSI alerts |
| Conclusion | Auto-generated summary of your run |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset (important: GitHub 100 MB limit)
`creditcard.csv` (Kaggle "Credit Card Fraud Detection") is ~144 MB, too big for GitHub.
Pick one:
1. **Zip it** (≈ 65 MB) and commit `creditcard.zip` next to `app.py` - auto-detected.
2. **Don't commit it**: choose *Upload CSV / ZIP* in the sidebar at run time.

The notebook used the **first 186,283 rows**; the sidebar default reproduces that
(F1 0.8684, PSI(P3) 3.252). Choose *Full dataset* to use all 284,807 rows.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo.
2. https://share.streamlit.io → *New app* → select repo, branch, main file `app.py`.
3. Deploy. First training takes ~15-30 s (the RF is the slowest part).

## Files
- `app.py` - UI
- `fraud_core.py` - all ML logic (mirrors `Project_SRM_final.ipynb`)
- `requirements.txt`, `.streamlit/config.toml`, `.gitignore`
