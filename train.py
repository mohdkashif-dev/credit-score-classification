"""Compare classifiers on the UCI credit-card default dataset."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SOURCE = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
          "00350/default%20of%20credit%20card%20clients.xls")
SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
TARGET = "default payment next month"
STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]
COLUMNS = ["LIMIT_BAL", *STATUS, *BILLS, *PAYMENTS]

def load_data(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        urlretrieve(SOURCE, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            temporary.unlink()
            raise ValueError("Downloaded workbook failed checksum validation.")
        temporary.replace(path)
    return pd.read_excel(path, sheet_name="Data", header=1, engine="xlrd")

def features(frame):
    data = frame[COLUMNS].apply(pd.to_numeric, errors="raise")
    bills = data[BILLS]
    payments = data[PAYMENTS]
    status = data[STATUS]
    mean_bill = bills.mean(axis=1)
    total_bill = bills.clip(lower=0).sum(axis=1, min_count=6).replace(0, np.nan)
    limit = data["LIMIT_BAL"].where(data["LIMIT_BAL"] > 0)
    return data.assign(
        late_months=status.gt(0).sum(axis=1).where(status.notna().all(axis=1)),
        max_delay=status.max(axis=1),
        mean_bill=mean_bill,
        mean_payment=payments.mean(axis=1),
        recent_utilization=bills["BILL_AMT1"].clip(lower=0).div(limit),
        payment_to_bill=payments.sum(axis=1, min_count=6).div(total_bill),
    )

def metrics(y, score, threshold):
    pred = score >= threshold
    return dict(accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                roc_auc=roc_auc_score(y, score))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/credit.xls"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = load_data(args.data)
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Expected a binary target with both labels 0 and 1.")
    x, y = features(frame), frame[TARGET].astype(int)
    x_build, x_test, y_build, y_test = train_test_split(
        x, y, stratify=y, test_size=.20, random_state=SEED)
    x_train, x_val, y_train, y_val = train_test_split(
        x_build, y_build, stratify=y_build, test_size=.25, random_state=SEED)
    models = {
        "Prior baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED)),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"), RandomForestClassifier(
                n_estimators=250, min_samples_leaf=10, n_jobs=-1, random_state=SEED)),
    }
    scores = []
    thresholds = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        val_scores = model.predict_proba(x_val)[:, 1]
        threshold = .5
        if name != "Prior baseline":
            precision, recall, options = precision_recall_curve(y_val, val_scores)
            pr_sum = precision[:-1] + recall[:-1]
            f1 = np.divide(2 * precision[:-1] * recall[:-1], pr_sum,
                           out=np.zeros_like(pr_sum), where=pr_sum > 0)
            threshold = float(options[f1.argmax()])
        thresholds[name] = threshold
        scores.append(dict(model=name, split="validation", threshold=threshold,
                           **metrics(y_val, val_scores, threshold)))

    selected = max((row for row in scores if row["model"] != "Prior baseline"),
                   key=lambda row: row["roc_auc"])["model"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        test_scores = model.predict_proba(x_test)[:, 1]
        threshold = thresholds[name]
        scores.append(dict(model=name, split="test", threshold=threshold,
                           **metrics(y_test, test_scores, threshold)))
        RocCurveDisplay.from_predictions(y_test, test_scores, name=name, ax=ax)
        if name == selected:
            cm = ConfusionMatrixDisplay.from_predictions(
                y_test, test_scores >= threshold, labels=[0, 1],
                display_labels=["No default", "Default"], cmap="Blues")
            cm.ax_.set_title(f"{selected} · test set")
            cm.figure_.tight_layout()
            cm.figure_.savefig(args.out / "confusion_matrix.png", dpi=160)
            plt.close(cm.figure_)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_title("Next-month default · test ROC curves")
    fig.tight_layout()
    fig.savefig(args.out / "roc_curves.png", dpi=160)
    plt.close(fig)
    table = pd.DataFrame(scores)
    table.to_csv(args.out / "metrics.csv", index=False)
    summary = dict(selected_model=selected, threshold=thresholds[selected],
                   dataset=str(args.data), seed=SEED,
                   train_rows=len(y_train), validation_rows=len(y_val),
                   test_rows=len(y_test), test_default_rate=float(y_test.mean()),
                   test_metrics=next(row for row in scores
                                     if row["model"] == selected and row["split"] == "test"))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    joblib.dump({"name": selected, "pipeline": models[selected],
                 "threshold": thresholds[selected]}, args.out / "model.joblib")
    print(table.round(4).to_string(index=False))
    print(f"\nSelected on validation ROC-AUC: {selected}")

if __name__ == "__main__":
    main()
