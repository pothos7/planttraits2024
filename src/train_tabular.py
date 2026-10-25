"""Tabular-only baseline for PlantTraits2024 (FGVC11).

Uses only the climate/soil/satellite columns in train.csv (no images).
Fits one gradient-boosted model per trait on log10 targets, reports
5-fold CV R2 per trait and the mean, and writes a submission CSV.

Usage:
    python src/train_tabular.py --data /scratch/$USER/data/planttraits2024
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--q-low", type=float, default=0.001,
                   help="drop rows below this quantile of any trait")
    p.add_argument("--q-high", type=float, default=0.98,
                   help="drop rows above this quantile of any trait")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data / "train.csv")
    test = pd.read_csv(args.data / "test.csv")
    sub = pd.read_csv(args.data / "sample_submission.csv")

    # Submission columns are trait names like X4, X11, ...; train has X4_mean, X4_sd, ...
    traits = [c for c in sub.columns if c != "id"]
    target_cols = [f"{t}_mean" for t in traits]
    missing = [c for c in target_cols if c not in train.columns]
    if missing:
        raise SystemExit(f"Expected target columns not in train.csv: {missing}")

    # Features = columns shared by train and test, minus id and any trait columns.
    trait_pattern = re.compile(r"^X\d+_(mean|sd)$")
    features = [
        c for c in test.columns
        if c != "id" and c in train.columns and not trait_pattern.match(c)
    ]
    print(f"{len(train)} train rows, {len(test)} test rows, {len(features)} features")
    print(f"Traits: {traits}")

    # Drop rows with non-positive targets so log10 is defined.
    y_raw = train[target_cols]
    keep = (y_raw > 0).all(axis=1)
    if (~keep).sum():
        print(f"Dropping {(~keep).sum()} rows with non-positive targets")
    train = train[keep].reset_index(drop=True)

    # Drop extreme outliers: rows outside [q_low, q_high] for any trait.
    lo = train[target_cols].quantile(args.q_low)
    hi = train[target_cols].quantile(args.q_high)
    inside = ((train[target_cols] >= lo) & (train[target_cols] <= hi)).all(axis=1)
    print(f"Outlier filter [{args.q_low}, {args.q_high}]: "
          f"keeping {inside.sum()} of {len(train)} rows")
    train = train[inside].reset_index(drop=True)
    X = train[features].to_numpy(dtype=np.float32)
    X_test = test[features].to_numpy(dtype=np.float32)
    Y = np.log10(train[target_cols].to_numpy())

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros_like(Y)
    test_pred = np.zeros((len(test), len(traits)))

    for j, trait in enumerate(traits):
        for fold, (tr, va) in enumerate(kf.split(X)):
            model = HistGradientBoostingRegressor(
                max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
                l2_regularization=1.0, early_stopping=True,
                random_state=args.seed + fold,
            )
            model.fit(X[tr], Y[tr, j])
            oof[va, j] = model.predict(X[va])
            test_pred[:, j] += model.predict(X_test) / args.folds
        # Raw-unit R2 matches the competition metric; log R2 is less outlier-sensitive.
        r2 = r2_score(10 ** Y[:, j], 10 ** oof[:, j])
        r2_log = r2_score(Y[:, j], oof[:, j])
        print(f"{trait:>6}  CV R2 = {r2:.4f}   (log-scale R2 = {r2_log:.4f})")

    mean_r2 = np.mean([
        r2_score(10 ** Y[:, j], 10 ** oof[:, j]) for j in range(len(traits))
    ])
    mean_r2_log = np.mean([r2_score(Y[:, j], oof[:, j]) for j in range(len(traits))])
    print(f"Mean CV R2 = {mean_r2:.4f}   (log-scale {mean_r2_log:.4f})")

    out = pd.DataFrame(10 ** test_pred, columns=traits)
    out.insert(0, "id", test["id"].values)
    path = args.out / "submission_tabular.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
