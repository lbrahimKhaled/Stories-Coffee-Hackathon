import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.data_utils import (  # noqa: E402
    aggregate_margin,
    build_branch_department_features,
    extract_total_scope_rows,
    load_and_clean_eda14_dataframe,
    split_total_rows,
)
from utils.ridge_utils import fit_ridge_with_loocv, ridge_predict, standardize_matrix  # noqa: E402


@dataclass
class EDA14TotersRolloutModel:
    data_path: str = "datasets/cleaned_14.csv"
    alphas: tuple = (0.01, 0.1, 1, 10, 100)

    feature_cols = [
        "Sales_TABLE",
        "GM_pct_TABLE",
        "Qty_TABLE",
        "Sales_TAKE AWAY",
        "GM_pct_TAKE AWAY",
        "Qty_TAKE AWAY",
        "Total_Sales",
        "TA_Share",
        "TA_OrderValue",
    ]

    def __post_init__(self):
        self.df = None
        self.df_items = None
        self.df_totals = None
        self.division_total = None
        self.category_total = None
        self.department_total = None
        self.category_margin = None
        self.department_margin = None
        self.branch_department_agg = None
        self.features = None
        self.scaler_mu = None
        self.scaler_sigma = None
        self.model_artifacts = {}
        self.predictions = None

    def prepare(self):
        self.df = load_and_clean_eda14_dataframe(self.data_path)
        self.df_items, self.df_totals = split_total_rows(self.df)

        self.division_total = extract_total_scope_rows(self.df_totals, "Division")
        self.category_total = extract_total_scope_rows(self.df_totals, "Category")
        self.department_total = extract_total_scope_rows(self.df_totals, "Department")

        self.category_margin = aggregate_margin(self.category_total, ["Branch", "Category"])
        self.department_margin = aggregate_margin(self.department_total, ["Branch", "Department"])

        feat, self.branch_department_agg = build_branch_department_features(self.df_items)

        feat["Total_Sales"] = feat["Sales_TABLE"].fillna(0) + feat["Sales_TAKE AWAY"].fillna(0)
        feat["TA_Share"] = np.where(feat["Total_Sales"] > 0, feat["Sales_TAKE AWAY"] / feat["Total_Sales"], 0.0)
        feat["Table_Share"] = np.where(feat["Total_Sales"] > 0, feat["Sales_TABLE"] / feat["Total_Sales"], 0.0)
        feat["TA_OrderValue"] = np.where(feat["Qty_TAKE AWAY"] > 0, feat["Sales_TAKE AWAY"] / feat["Qty_TAKE AWAY"], 0.0)
        feat["Table_OrderValue"] = np.where(feat["Qty_TABLE"] > 0, feat["Sales_TABLE"] / feat["Qty_TABLE"], 0.0)
        feat["has_toters"] = feat["Sales_Toters"].fillna(0) > 0

        self.features = feat
        return self

    def fit(self):
        if self.features is None:
            self.prepare()

        train = self.features[self.features["has_toters"]].copy()
        if len(train) == 0:
            raise ValueError("No branches with Toters found. Cannot train model.")

        X_train = train[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
        X_train, self.scaler_mu, self.scaler_sigma = standardize_matrix(X_train)

        for target in ["Sales_Toters", "GM_pct_Toters"]:
            y_train = train[target].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
            beta, best_alpha, rmse = fit_ridge_with_loocv(X_train, y_train, self.alphas)
            self.model_artifacts[target] = {
                "beta": beta,
                "alpha": best_alpha,
                "loocv_rmse": rmse,
            }

        X_all = self.features[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
        X_all = (X_all - self.scaler_mu) / self.scaler_sigma

        pred_sales = ridge_predict(X_all, self.model_artifacts["Sales_Toters"]["beta"]).clip(min=0)
        pred_gm = ridge_predict(X_all, self.model_artifacts["GM_pct_Toters"]["beta"]).clip(0, 1)

        preds = self.features[["Branch", "has_toters"]].copy()
        preds["pred_Sales_Toters"] = pred_sales
        preds["pred_GM_pct_Toters"] = pred_gm
        preds["pred_Profit_Toters"] = pred_sales * pred_gm

        self.predictions = preds
        return self

    def model_metrics(self):
        if not self.model_artifacts:
            raise ValueError("Model is not trained. Run fit() first.")

        rows = []
        for target, artifact in self.model_artifacts.items():
            rows.append(
                {
                    "target": target,
                    "best_alpha": artifact["alpha"],
                    "loocv_rmse": artifact["loocv_rmse"],
                }
            )
        return pd.DataFrame(rows)

    def recommend(self, top_n=10, only_missing_toters=True):
        if self.predictions is None:
            raise ValueError("Model is not trained. Run fit() first.")

        out = self.predictions.copy()
        if only_missing_toters:
            out = out[~out["has_toters"]]
        out = out.sort_values("pred_Profit_Toters", ascending=False).reset_index(drop=True)
        return out.head(top_n)


def run_cli():
    parser = argparse.ArgumentParser(description="EDA-14 Toters rollout model")
    parser.add_argument("--data-path", default="datasets/cleaned_14.csv", help="Path to cleaned_14.csv")
    parser.add_argument("--top-n", type=int, default=10, help="How many branch recommendations to print")
    parser.add_argument(
        "--save-path",
        default="datasets/toters_rollout_recommendations.csv",
        help="CSV path for recommendations output",
    )
    args = parser.parse_args()

    model = EDA14TotersRolloutModel(data_path=args.data_path)
    model.prepare().fit()

    print("Model metrics:")
    print(model.model_metrics().to_string(index=False))
    print()
    print("Top branch recommendations for Toters rollout:")
    recommendations = model.recommend(top_n=args.top_n, only_missing_toters=True)
    print(recommendations.to_string(index=False))

    if args.save_path:
        recommendations.to_csv(args.save_path, index=False)
        print()
        print(f"Saved recommendations to: {args.save_path}")


if __name__ == "__main__":
    run_cli()
