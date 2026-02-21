import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.data_utils import extract_total_scope_rows, load_and_clean_eda14_dataframe, split_total_rows  # noqa: E402


def safe_div(num, den, default=0.0):
    return np.where(den > 0, num / den, default)


def loocv_rmse_model(model, X, y):
    loo = LeaveOneOut()
    preds = []
    trues = []
    for train_idx, test_idx in loo.split(X):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds.append(float(m.predict(X.iloc[test_idx])[0]))
        trues.append(float(y.iloc[test_idx].iloc[0]))
    return float(np.sqrt(mean_squared_error(trues, preds)))


def loocv_mean_baseline_rmse(y):
    y_arr = np.array(y, dtype=float)
    preds = []
    for i in range(len(y_arr)):
        preds.append(float(np.delete(y_arr, i).mean()))
    return float(np.sqrt(mean_squared_error(y_arr, preds)))


def pick_best_feature_model(feature_sets, candidates, train_df, target_col):
    y = train_df[target_col].astype(float)
    best = None
    for feature_set_name, feature_cols in feature_sets.items():
        X = train_df[feature_cols]
        for model_name, model in candidates:
            rmse = loocv_rmse_model(model, X, y)
            if (best is None) or (rmse < best["rmse"]):
                best = {
                    "feature_set": feature_set_name,
                    "feature_cols": feature_cols,
                    "model_name": model_name,
                    "model": clone(model),
                    "rmse": rmse,
                }
    return best


def main():
    # Apply EDA-14 preprocessing before learning.
    df = load_and_clean_eda14_dataframe("datasets/cleaned_14.csv")
    _, df_totals = split_total_rows(df)
    department_total = extract_total_scope_rows(df_totals, "Department")
    if department_total.empty:
        raise ValueError("No 'Total By Department' rows found after EDA preprocessing.")

    agg = (
        department_total.groupby(["Branch", "Department"], dropna=False)
        .agg(
            Sales=("Total Price", "sum"),
            Cost=("Total Cost", "sum"),
            Profit=("Total Profit", "sum"),
            Qty=("Qty", "sum"),
        )
        .reset_index()
    )
    agg["GM_pct"] = safe_div(agg["Profit"], agg["Sales"], default=np.nan)

    feat = agg.pivot(index="Branch", columns="Department", values=["Sales", "Cost", "Profit", "Qty", "GM_pct"])
    feat.columns = [f"{metric}_{dept}" for metric, dept in feat.columns]
    feat = feat.reset_index()

    for dep in ["TABLE", "TAKE AWAY", "Toters"]:
        for metric in ["Sales", "Cost", "Profit", "Qty", "GM_pct"]:
            col = f"{metric}_{dep}"
            if col not in feat.columns:
                feat[col] = np.nan

    # Feature engineering.
    feat["Current_Sales"] = feat[["Sales_TABLE", "Sales_Toters"]].fillna(0).sum(axis=1)
    feat["Current_Profit"] = feat[["Profit_TABLE", "Profit_Toters"]].fillna(0).sum(axis=1)
    feat["Current_Qty"] = feat[["Qty_TABLE", "Qty_Toters"]].fillna(0).sum(axis=1)
    feat["Current_GM_pct"] = safe_div(feat["Current_Profit"], feat["Current_Sales"], default=np.nan)
    feat["Table_AOV"] = safe_div(feat["Sales_TABLE"], feat["Qty_TABLE"], default=0.0)
    feat["Current_AOV"] = safe_div(feat["Current_Sales"], feat["Current_Qty"], default=0.0)
    feat["Toters_Share"] = safe_div(feat["Sales_Toters"].fillna(0), feat["Current_Sales"], default=0.0)
    feat["has_table"] = (feat["Sales_TABLE"].fillna(0) > 0).astype(int)
    feat["has_toters"] = (feat["Sales_Toters"].fillna(0) > 0).astype(int)
    feat["log_Current_Sales"] = np.log1p(feat["Current_Sales"].clip(lower=0))
    feat["log_Current_Qty"] = np.log1p(feat["Current_Qty"].clip(lower=0))
    feat["log_Table_Sales"] = np.log1p(feat["Sales_TABLE"].fillna(0).clip(lower=0))
    feat["log_Toters_Sales"] = np.log1p(feat["Sales_Toters"].fillna(0).clip(lower=0))

    all_features = [
        "Sales_TABLE",
        "Profit_TABLE",
        "GM_pct_TABLE",
        "Qty_TABLE",
        "Table_AOV",
        "Sales_Toters",
        "Profit_Toters",
        "GM_pct_Toters",
        "Qty_Toters",
        "Toters_Share",
        "Current_Sales",
        "Current_Profit",
        "Current_GM_pct",
        "Current_AOV",
        "has_table",
        "has_toters",
        "log_Current_Sales",
        "log_Current_Qty",
        "log_Table_Sales",
        "log_Toters_Sales",
    ]

    feature_sets = {
        "core": [
            "Sales_TABLE",
            "Profit_TABLE",
            "GM_pct_TABLE",
            "Qty_TABLE",
            "Table_AOV",
            "Sales_Toters",
            "Profit_Toters",
            "GM_pct_Toters",
            "Qty_Toters",
            "Toters_Share",
            "Current_Sales",
            "Current_Profit",
            "Current_GM_pct",
            "Current_AOV",
        ],
        "core_plus_flags": [
            "Sales_TABLE",
            "Profit_TABLE",
            "GM_pct_TABLE",
            "Qty_TABLE",
            "Table_AOV",
            "Sales_Toters",
            "Profit_Toters",
            "GM_pct_Toters",
            "Qty_Toters",
            "Toters_Share",
            "Current_Sales",
            "Current_Profit",
            "Current_GM_pct",
            "Current_AOV",
            "has_table",
            "has_toters",
        ],
        "all": all_features,
    }

    train = feat[feat["Sales_TAKE AWAY"].fillna(0) > 0].copy()
    if len(train) < 5:
        raise ValueError("Not enough TAKE AWAY training branches.")

    sales_candidates = [
        (
            "extra_trees",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=600,
                            random_state=42,
                            min_samples_leaf=2,
                            max_features="sqrt",
                        ),
                    ),
                ]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=500,
                            random_state=42,
                            min_samples_leaf=2,
                            max_features="sqrt",
                        ),
                    ),
                ]
            ),
        ),
        (
            "ridge_log",
            TransformedTargetRegressor(
                regressor=Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("ridge", Ridge(alpha=10.0)),
                    ]
                ),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
        ),
    ]

    gm_candidates = [
        (
            "extra_trees",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=600,
                            random_state=42,
                            min_samples_leaf=2,
                            max_features="sqrt",
                        ),
                    ),
                ]
            ),
        ),
        (
            "ridge_strong",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=1000.0)),
                ]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=500,
                            random_state=42,
                            min_samples_leaf=2,
                            max_features="sqrt",
                        ),
                    ),
                ]
            ),
        ),
    ]

    best_sales = pick_best_feature_model(feature_sets, sales_candidates, train, "Sales_TAKE AWAY")
    best_gm = pick_best_feature_model(feature_sets, gm_candidates, train, "GM_pct_TAKE AWAY")

    y_sales = train["Sales_TAKE AWAY"].astype(float)
    y_gm = train["GM_pct_TAKE AWAY"].astype(float)
    sales_baseline_rmse = loocv_mean_baseline_rmse(y_sales)
    gm_baseline_rmse = loocv_mean_baseline_rmse(y_gm)

    best_sales["model"].fit(train[best_sales["feature_cols"]], y_sales)
    best_gm["model"].fit(train[best_gm["feature_cols"]], y_gm)

    feat["pred_Sales_TAKE AWAY"] = best_sales["model"].predict(feat[best_sales["feature_cols"]]).clip(min=0)
    feat["pred_GM_pct_TAKE AWAY"] = best_gm["model"].predict(feat[best_gm["feature_cols"]]).clip(0, 1)
    feat["pred_Profit_TAKE AWAY"] = feat["pred_Sales_TAKE AWAY"] * feat["pred_GM_pct_TAKE AWAY"]

    branch_key = feat["Branch"].str.strip().str.lower()
    if "stories alay" not in set(branch_key):
        raise ValueError("Branch 'Stories alay' not found in dataset.")
    alay = feat[branch_key == "stories alay"].iloc[0]

    current_sales = float(alay["Current_Sales"])
    current_profit = float(alay["Current_Profit"])
    current_gm = float(alay["Current_GM_pct"])
    ta_sales = float(alay["pred_Sales_TAKE AWAY"])
    ta_gm = float(alay["pred_GM_pct_TAKE AWAY"])
    ta_profit = float(alay["pred_Profit_TAKE AWAY"])

    print("Preprocessing applied before learning:")
    print("- EDA-14 cleaning (numeric parsing, total-price fix, zero-sales removal)")
    print("- Department total extraction from 'Total By Department:' rows")
    print("- Missingness flags, log features, median imputation; scaling for linear models")
    print()
    print("Model diagnostics:")
    print(
        f"- Sales model: {best_sales['model_name']} | feature set: {best_sales['feature_set']} "
        f"| LOOCV RMSE={best_sales['rmse']:,.2f}"
    )
    print(f"- Sales baseline (mean), LOOCV RMSE={sales_baseline_rmse:,.2f}")
    print(
        f"- Sales RMSE improvement vs baseline: {((sales_baseline_rmse - best_sales['rmse']) / sales_baseline_rmse):.2%}"
    )
    print(
        f"- GM% model: {best_gm['model_name']} | feature set: {best_gm['feature_set']} "
        f"| LOOCV RMSE={best_gm['rmse']:.4f}"
    )
    print(f"- GM% baseline (mean), LOOCV RMSE={gm_baseline_rmse:.4f}")
    print(f"- GM% RMSE improvement vs baseline: {((gm_baseline_rmse - best_gm['rmse']) / gm_baseline_rmse):.2%}")
    print()
    print("Stories alay baseline:")
    print(f"- Current sales: {current_sales:,.2f}")
    print(f"- Current profit: {current_profit:,.2f}")
    print(f"- Current GM%: {current_gm:.2%}")
    print()
    print("Predicted TAKE AWAY (before cannibalization adjustment):")
    print(f"- TAKE AWAY sales: {ta_sales:,.2f}")
    print(f"- TAKE AWAY profit: {ta_profit:,.2f}")
    print(f"- TAKE AWAY GM%: {ta_gm:.2%}")
    print()

    scenarios = []
    for cannibalization_rate in [0.0, 0.25, 0.50]:
        inc_sales = ta_sales * (1 - cannibalization_rate)
        inc_profit = ta_profit - (cannibalization_rate * ta_sales * current_gm)
        post_sales = current_sales + inc_sales
        post_profit = current_profit + inc_profit
        post_gm = post_profit / post_sales if post_sales > 0 else np.nan
        scenarios.append(
            {
                "Cannibalization": f"{int(cannibalization_rate * 100)}%",
                "Incremental Sales": inc_sales,
                "Incremental Profit": inc_profit,
                "Profit Uplift %": (inc_profit / current_profit) if current_profit > 0 else np.nan,
                "Post-launch GM%": post_gm,
            }
        )

    scenarios_df = pd.DataFrame(scenarios)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("Impact scenarios for Stories alay:")
    print(scenarios_df.to_string(index=False))


if __name__ == "__main__":
    main()
