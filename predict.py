"""Score a CSV with the six months of credit-card history."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from train import COLUMNS, features

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--model", type=Path, default=Path("results/model.joblib"))
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    frame = pd.read_csv(args.csv)
    missing = sorted(set(COLUMNS) - set(frame))
    if missing:
        parser.error("Missing fields: " + ", ".join(missing))
    if frame.empty:
        parser.error("Include at least one record.")
    probabilities = bundle["pipeline"].predict_proba(features(frame))[:, 1]
    result = [
        {"row": index + 1, "default_score": round(float(probability), 6),
         "above_demo_threshold": bool(probability >= bundle["threshold"])}
        for index, probability in enumerate(probabilities)
    ]
    print(json.dumps({"model": bundle["name"], "threshold": bundle["threshold"],
                      "predictions": result}, indent=2))

if __name__ == "__main__":
    main()
