"""
this will output the following:

They’re the three outputs of your clustering pipeline, each at a different level:

product_cluster_k_scores_14.csv: model selection diagnostics for each tested k (3..8 here).
Contains silhouette, inertia, cluster size balance, and selected=True for the chosen k.

product_cluster_assignments_14.csv: product-level results (1 row per product).
Includes each product’s metrics (Sales, MarginPct, channel shares, etc.), assigned Cluster, plus ClusterLabel and SuggestedAction.

product_cluster_profiles_14.csv: cluster-level summary (1 row per cluster).
Aggregates totals/medians per cluster, cluster sales/profit share, label/action, and TopProducts for quick interpretation.

So: first file chooses the cluster count, second gives detailed assignments, third gives executive summary profiles.



"""


import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.data_utils import load_and_clean_eda14_dataframe, split_total_rows  # noqa: E402


DEPARTMENTS = ("TABLE", "TAKE AWAY", "Toters")
FEATURE_COLUMNS = [
    "log_Sales",
    "MarginPct",
    "log_AvgPrice",
    "BranchPenetration",
    "TakeAwayShare",
    "TableShare",
    "TotersShare",
]


def mode_or_first(values: pd.Series):
    mode_vals = values.mode(dropna=True)
    if not mode_vals.empty:
        return mode_vals.iloc[0]
    return values.iloc[0] if len(values) else np.nan


def resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def build_product_table(data_path: Path) -> pd.DataFrame:
    df = load_and_clean_eda14_dataframe(str(data_path))
    item_rows, _ = split_total_rows(df)

    if item_rows.empty:
        raise ValueError("No item-level rows found after preprocessing.")

    n_branches = int(item_rows["Branch"].nunique())
    if n_branches == 0:
        raise ValueError("No branches found in dataset.")

    product_table = (
        item_rows.groupby("Product Desc", dropna=False)
        .agg(
            Category=("Category", mode_or_first),
            Division=("Division", mode_or_first),
            Sales=("Total Price", "sum"),
            Cost=("Total Cost", "sum"),
            Profit=("Total Profit", "sum"),
            Qty=("Qty", "sum"),
            Branches=("Branch", "nunique"),
        )
        .reset_index()
    )

    dept_sales = (
        item_rows.groupby(["Product Desc", "Department"], dropna=False)["Total Price"]
        .sum()
        .unstack(fill_value=0.0)
    )
    for dept in DEPARTMENTS:
        if dept not in dept_sales.columns:
            dept_sales[dept] = 0.0
    dept_sales = dept_sales[list(DEPARTMENTS)].reset_index()

    product_table = product_table.merge(dept_sales, on="Product Desc", how="left")

    product_table["MarginPct"] = np.where(
        product_table["Sales"] > 0,
        product_table["Profit"] / product_table["Sales"],
        np.nan,
    )
    product_table["AvgPrice"] = np.where(
        product_table["Qty"] > 0,
        product_table["Sales"] / product_table["Qty"],
        np.nan,
    )
    product_table["BranchPenetration"] = product_table["Branches"] / n_branches

    product_table["TableShare"] = np.where(
        product_table["Sales"] > 0,
        product_table["TABLE"] / product_table["Sales"],
        0.0,
    )
    product_table["TakeAwayShare"] = np.where(
        product_table["Sales"] > 0,
        product_table["TAKE AWAY"] / product_table["Sales"],
        0.0,
    )
    product_table["TotersShare"] = np.where(
        product_table["Sales"] > 0,
        product_table["Toters"] / product_table["Sales"],
        0.0,
    )

    product_table["log_Sales"] = np.log1p(product_table["Sales"].clip(lower=0))
    product_table["log_AvgPrice"] = np.log1p(product_table["AvgPrice"].clip(lower=0))

    return product_table


def build_feature_matrix(product_table: pd.DataFrame, clip_quantile: float) -> np.ndarray:
    X = product_table[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).copy()
    X = X.fillna(X.median(numeric_only=True))

    if 0.0 < clip_quantile < 0.5:
        lower = clip_quantile
        upper = 1.0 - clip_quantile
        for col in X.columns:
            lo, hi = X[col].quantile([lower, upper])
            X[col] = X[col].clip(lo, hi)

    scaler = StandardScaler()
    return scaler.fit_transform(X)


def score_candidate_k(
    X_scaled: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
    n_init: int,
) -> pd.DataFrame:
    n_samples = X_scaled.shape[0]
    k_min = max(2, int(k_min))
    k_max = min(int(k_max), n_samples - 1)

    if k_min > k_max:
        raise ValueError(f"Invalid k range after bounds check: {k_min}..{k_max}.")

    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X_scaled)

        vc = pd.Series(labels).value_counts()
        sil = silhouette_score(X_scaled, labels) if vc.size > 1 else np.nan

        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": float(sil),
                "min_cluster_size": int(vc.min()),
                "max_cluster_size": int(vc.max()),
                "min_cluster_ratio": float(vc.min() / n_samples),
                "max_cluster_ratio": float(vc.max() / n_samples),
            }
        )

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def choose_k(
    k_scores: pd.DataFrame,
    min_cluster_ratio: float,
    forced_k: int | None,
) -> int:
    if forced_k is not None:
        if forced_k not in k_scores["k"].tolist():
            valid = ", ".join(str(x) for x in k_scores["k"].tolist())
            raise ValueError(f"Forced k={forced_k} is outside evaluated range. Valid: {valid}")
        return int(forced_k)

    candidates = k_scores[k_scores["min_cluster_ratio"] >= float(min_cluster_ratio)].copy()
    if candidates.empty:
        candidates = k_scores.copy()

    ranked = candidates.sort_values(["silhouette", "k"], ascending=[False, True], na_position="last")
    return int(ranked.iloc[0]["k"])


def label_cluster(row: pd.Series, stats: dict[str, float], cluster_id: int) -> tuple[str, str]:
    if row["median_sales"] >= stats["sales_q75"] and row["avg_branch_penetration"] >= stats["branch_q75"]:
        name = f"C{cluster_id}: Core Winners"
        action = "Protect availability, bundle with add-ons, and prioritize in promotions."
    elif row["avg_takeaway_share"] >= 0.90 and row["avg_branch_penetration"] <= stats["branch_q50"]:
        name = f"C{cluster_id}: Takeaway Niche"
        action = "Keep only in demand branches and test targeted takeaway campaigns."
    elif row["avg_toters_share"] >= stats["toters_q75"]:
        name = f"C{cluster_id}: Toters-Leaning"
        action = "Optimize delivery packaging and run delivery-only bundles."
    elif row["median_margin_pct"] <= stats["margin_q25"]:
        name = f"C{cluster_id}: Margin Risk"
        action = "Review recipe cost and pricing before scaling."
    else:
        name = f"C{cluster_id}: Growth Mid-Tier"
        action = "Expand selectively and pair with high-margin complements."

    return name, action


def build_cluster_outputs(assignments: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    profiles = (
        assignments.groupby("Cluster", as_index=False)
        .agg(
            n_products=("Product Desc", "count"),
            total_sales=("Sales", "sum"),
            total_profit=("Profit", "sum"),
            median_sales=("Sales", "median"),
            median_profit=("Profit", "median"),
            median_margin_pct=("MarginPct", "median"),
            median_avg_price=("AvgPrice", "median"),
            avg_branch_penetration=("BranchPenetration", "mean"),
            avg_takeaway_share=("TakeAwayShare", "mean"),
            avg_table_share=("TableShare", "mean"),
            avg_toters_share=("TotersShare", "mean"),
        )
        .sort_values("Cluster")
        .reset_index(drop=True)
    )

    total_sales = profiles["total_sales"].sum()
    total_profit = profiles["total_profit"].sum()
    profiles["sales_share"] = np.where(total_sales > 0, profiles["total_sales"] / total_sales, 0.0)
    profiles["profit_share"] = np.where(total_profit != 0, profiles["total_profit"] / total_profit, 0.0)

    stats = {
        "sales_q75": float(profiles["median_sales"].quantile(0.75)),
        "branch_q75": float(profiles["avg_branch_penetration"].quantile(0.75)),
        "branch_q50": float(profiles["avg_branch_penetration"].quantile(0.50)),
        "toters_q75": float(profiles["avg_toters_share"].quantile(0.75)),
        "margin_q25": float(profiles["median_margin_pct"].quantile(0.25)),
    }

    cluster_name_map: dict[int, str] = {}
    cluster_action_map: dict[int, str] = {}
    for idx, row in profiles.iterrows():
        cluster_id = int(row["Cluster"])
        cluster_name, action = label_cluster(row, stats, cluster_id)
        profiles.loc[idx, "ClusterLabel"] = cluster_name
        profiles.loc[idx, "SuggestedAction"] = action
        cluster_name_map[cluster_id] = cluster_name
        cluster_action_map[cluster_id] = action

    top_products = (
        assignments.sort_values(["Cluster", "Sales"], ascending=[True, False])
        .groupby("Cluster")["Product Desc"]
        .apply(lambda s: "; ".join(s.head(8)))
        .to_dict()
    )
    profiles["TopProducts"] = profiles["Cluster"].map(top_products)

    assignments["ClusterLabel"] = assignments["Cluster"].map(cluster_name_map)
    assignments["SuggestedAction"] = assignments["Cluster"].map(cluster_action_map)

    return profiles, cluster_name_map


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cluster products in dataset 14 to segment menu portfolio behavior "
            "(economics + channel mix + branch penetration)."
        )
    )
    parser.add_argument("--data-path", default="datasets/cleaned_14.csv", help="Input dataset path.")
    parser.add_argument(
        "--output-dir",
        default="datasets/outputs/product_clustering_14/global",
        help="Directory where cluster CSV outputs will be written.",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=3,
        help="Minimum k to evaluate for diagnostic scoring.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=8,
        help="Maximum k to evaluate for diagnostic scoring.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Force k (if omitted, best k is chosen from scores).",
    )
    parser.add_argument(
        "--min-cluster-ratio",
        type=float,
        default=0.05,
        help="Minimum allowed smallest-cluster ratio when auto-selecting k.",
    )
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=0.01,
        help="Winsorization quantile for each feature (0 disables clipping).",
    )
    parser.add_argument("--n-init", type=int, default=80, help="KMeans n_init.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data_path = resolve_path(args.data_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    product_table = build_product_table(data_path)
    X_scaled = build_feature_matrix(product_table, clip_quantile=args.clip_quantile)

    k_scores = score_candidate_k(
        X_scaled=X_scaled,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
        n_init=args.n_init,
    )
    selected_k = choose_k(
        k_scores=k_scores,
        min_cluster_ratio=args.min_cluster_ratio,
        forced_k=args.n_clusters,
    )
    k_scores["selected"] = k_scores["k"] == selected_k

    model = KMeans(n_clusters=selected_k, random_state=args.random_state, n_init=args.n_init)
    assignments = product_table.copy()
    assignments["Cluster"] = model.fit_predict(X_scaled)

    assignments["SalesShare"] = np.where(
        assignments["Sales"].sum() > 0,
        assignments["Sales"] / assignments["Sales"].sum(),
        0.0,
    )
    assignments["ProfitShare"] = np.where(
        assignments["Profit"].sum() != 0,
        assignments["Profit"] / assignments["Profit"].sum(),
        0.0,
    )

    profiles, _ = build_cluster_outputs(assignments)

    k_scores_path = output_dir / "product_cluster_k_scores_14.csv"
    assignments_path = output_dir / "product_cluster_assignments_14.csv"
    profiles_path = output_dir / "product_cluster_profiles_14.csv"

    k_scores.to_csv(k_scores_path, index=False)
    assignments.sort_values(["Cluster", "Sales"], ascending=[True, False]).to_csv(assignments_path, index=False)
    profiles.sort_values("Cluster").to_csv(profiles_path, index=False)

    print(f"Input rows clustered (products): {len(assignments)}")
    print(f"Selected k: {selected_k}")
    print(f"Wrote: {k_scores_path}")
    print(f"Wrote: {assignments_path}")
    print(f"Wrote: {profiles_path}")


if __name__ == "__main__":
    main()
