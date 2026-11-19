# Credit Scoring Model

CodeAlpha Machine Learning Internship · Task 1

Predict next-month credit-card default from six months of repayment status, balances and payments. Compare Logistic Regression and Random Forest against a class-prior baseline. This is an educational model for historical credit-card behavior.

## Run in Google Colab

Upload `Credit_Scoring_Colab.ipynb` to [Google Colab](https://colab.research.google.com/), then select **Runtime → Run all**. The notebook contains all source code, fetches the public data and exports trained outputs.

## Run with Python

```bash
python -m pip install -r requirements.txt
python train.py
python predict.py demo_profiles.csv
```

The first run downloads the original UCI workbook and verifies its SHA-256 digest. Use `--data /path/to/workbook.xls` to supply an existing workbook. The demo CSV uses two invented records, with money amounts in NT dollars and months ordered from most recent to oldest.

## Method and results

- Target label 1 means default next month and label 0 means no default.
- Features include late-month count, maximum delay, mean bill, mean payment, recent utilization and payments relative to total positive bills. The model excludes customer ID and demographics. The dataset has no income field.
- Use a stratified 60/20/20 split. Fit preprocessing and parameters on the training set, choose the model by validation ROC-AUC and choose each learned model's threshold by validation F1. Evaluate the frozen choices on the test set.
- `results/metrics.csv` compares accuracy, precision, recall, F1 and ROC-AUC. `summary.json`, `confusion_matrix.png`, `roc_curves.png`, and `model.joblib` contain the selected model, charts, and metrics.
- The supplied source bundle ships before a benchmark run. Run training to generate evidence. The model score is uncalibrated. Real lending decisions require validation beyond this educational project.

## Dataset

Yeh, I.-C. (2009). Default of Credit Card Clients. UCI Machine Learning Repository. [Dataset and attribution](https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients) · [DOI](https://doi.org/10.24432/C55S3H). The feature engineering in this project adapts the dataset for training.
