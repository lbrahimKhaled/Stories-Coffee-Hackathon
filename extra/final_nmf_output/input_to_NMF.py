# models/input_to_NMF.py
# Reads your cleaned CSV, builds branch-level normalized profiles (the NMF input matrix),
# and saves: datasets/branch_profile_group_for_nmf.csv

import re
import pandas as pd

# =========================
# CONFIG: set your input here
# =========================
INPUT_CSV = "datasets/Cleaned_191.csv"   # change to "datasets/cleaned_14.csv" if needed
OUTPUT_CSV = "datasets/branch_profile_group_for_nmf.csv"

# If your dataset has these exact headers (as in your sample), keep them.
COL_DESCRIPTION = "Description"
COL_QTY = "Qty"
COL_TOTAL_AMOUNT = "Total Amount"
COL_BRANCH = "Branch"
COL_GROUP = "Group"
COL_DIVISION = "Division"

TOTAL_ROW_PATTERN = re.compile(r"^\s*Total\s+by\s+", flags=re.IGNORECASE)


def parse_money(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace('"', '', regex=False).str.strip()
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def main():
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.astype(str).str.strip()

    # Basic column check
    required = [COL_DESCRIPTION, COL_QTY, COL_TOTAL_AMOUNT, COL_BRANCH, COL_GROUP, COL_DIVISION]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    # Drop all_nan rows if present
    if "all_nan" in df.columns:
        df = df.loc[~df["all_nan"].fillna(False)].copy()

    # Remove "Total by ..." rows (critical to avoid double-counting)
    desc = df[COL_DESCRIPTION].astype(str)
    df = df.loc[~desc.str.match(TOTAL_ROW_PATTERN)].copy()

    # Parse numeric columns
    df["TotalAmount"] = parse_money(df[COL_TOTAL_AMOUNT])
    df["Qty"] = pd.to_numeric(df[COL_QTY], errors="coerce")

    # Keep only rows with a branch and at least one metric
    df = df.dropna(subset=[COL_BRANCH])
    df = df.loc[~(df["TotalAmount"].isna() & df["Qty"].isna())].copy()

    # =========================
    # Build NMF input using Group revenue shares
    # =========================
    grp = (
        df.groupby([COL_BRANCH, COL_GROUP], dropna=False)["TotalAmount"]
        .sum()
        .reset_index()
    )

    # Total revenue per branch
    totals = grp.groupby(COL_BRANCH)["TotalAmount"].sum().rename("branch_total").reset_index()
    grp = grp.merge(totals, on=COL_BRANCH, how="left")

    # Revenue share per group within branch
    grp["rev_share"] = grp["TotalAmount"] / grp["branch_total"]

    # Wide matrix: rows=Branch, columns=Group, values=rev_share
    X = grp.pivot_table(
        index=COL_BRANCH,
        columns=COL_GROUP,
        values="rev_share",
        fill_value=0.0,
        aggfunc="sum",
    ).reset_index()

    # Optional: make column names explicit (safe for downstream)
    X.columns = [COL_BRANCH] + [f"revshare_group__{str(c).strip()}" for c in X.columns[1:]]

    X.to_csv(OUTPUT_CSV, index=False)

    print("Saved NMF input to:", OUTPUT_CSV)
    print("Shape:", X.shape)


if __name__ == "__main__":
    main()