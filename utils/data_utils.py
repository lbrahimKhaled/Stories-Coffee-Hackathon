import re
import numpy as np
import pandas as pd


DEFAULT_NUMERIC_COLUMNS = [
    "Qty",
    "Total Price",
    "Total Cost",
    "Total Cost %",
    "Total Profit",
    "Total Profit %",
]

DEFAULT_DEPARTMENTS = ("TABLE", "TAKE AWAY", "Toters")
DEFAULT_METRICS = ("Sales", "Cost", "Profit", "Qty", "GM_pct")


def to_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(",", "")
    if re.match(r"^\(.*\)$", s):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except Exception:
        return np.nan


def clean_numeric_columns(df, numeric_columns=None):
    out = df.copy()
    numeric_columns = numeric_columns or DEFAULT_NUMERIC_COLUMNS
    for col in numeric_columns:
        if col in out.columns:
            out[col] = out[col].apply(to_num)
    return out


def load_and_clean_eda14_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    if "all_nan" in df.columns:
        df = df.drop(columns=["all_nan"])

    df = clean_numeric_columns(df)

    if {"Total Cost", "Total Profit"}.issubset(df.columns):
        # Matches the fix used in EDA-14.ipynb
        df["Total Price"] = df["Total Cost"].fillna(0) + df["Total Profit"].fillna(0)

    if "Total Price" in df.columns:
        df = df[df["Total Price"] != 0].copy()

    return df.reset_index(drop=True)


def split_total_rows(df, desc_col="Product Desc"):
    is_total = df[desc_col].astype(str).str.strip().str.match(r"(?i)^total\s+by\s+")
    df_totals = df[is_total].copy().reset_index(drop=True)
    df_items = df[~is_total].copy().reset_index(drop=True)
    return df_items, df_totals


def extract_total_scope_rows(df_totals, scope, desc_col="Product Desc"):
    target = f"Total By {scope}:"
    scoped = df_totals[df_totals[desc_col].astype(str).str.contains(target, case=False, na=False)].copy()
    if desc_col in scoped.columns:
        scoped = scoped.drop(columns=[desc_col])
    return scoped.reset_index(drop=True)


def aggregate_margin(df, group_cols):
    group_cols = list(group_cols)
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["Sales", "COGS", "GM", "Qty", "GM_pct"])

    g = (
        df.groupby(group_cols, dropna=False)
        .agg(
            Sales=("Total Price", "sum"),
            COGS=("Total Cost", "sum"),
            GM=("Total Profit", "sum"),
            Qty=("Qty", "sum"),
        )
        .reset_index()
    )
    g["GM_pct"] = np.where(g["Sales"] != 0, g["GM"] / g["Sales"], np.nan)
    return g


def build_branch_department_features(
    df_items,
    departments=DEFAULT_DEPARTMENTS,
    metrics=DEFAULT_METRICS,
):
    agg = (
        df_items.groupby(["Branch", "Department"], dropna=False)
        .agg(
            Sales=("Total Price", "sum"),
            Cost=("Total Cost", "sum"),
            Profit=("Total Profit", "sum"),
            Qty=("Qty", "sum"),
        )
        .reset_index()
    )
    agg["GM_pct"] = np.where(agg["Sales"] != 0, agg["Profit"] / agg["Sales"], np.nan)

    feat = agg.pivot(index="Branch", columns="Department", values=["Sales", "Cost", "Profit", "Qty", "GM_pct"])
    feat.columns = [f"{metric}_{dept}" for metric, dept in feat.columns]
    feat = feat.reset_index()

    for dept in departments:
        for metric in metrics:
            col = f"{metric}_{dept}"
            if col not in feat.columns:
                feat[col] = np.nan

    return feat, agg
