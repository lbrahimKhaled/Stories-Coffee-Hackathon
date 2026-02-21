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
    "SalesShareInBranch",
    "ProfitShareInBranch",
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


def build_branch_product_table(data_path: Path) -> pd.DataFrame:
    df = load_and_clean_eda14_dataframe(str(data_path))
    item_rows, _ = split_total_rows(df)

    if item_rows.empty:
        raise ValueError("No item-level rows found after preprocessing.")

    product_table = (
        item_rows.groupby(["Branch", "Product Desc"], dropna=False)
        .agg(
            Category=("Category", mode_or_first),
            Division=("Division", mode_or_first),
            Sales=("Total Price", "sum"),
            Cost=("Total Cost", "sum"),
            Profit=("Total Profit", "sum"),
            Qty=("Qty", "sum"),
        )
        .reset_index()
    )

    dept_sales = (
        item_rows.groupby(["Branch", "Product Desc", "Department"], dropna=False)["Total Price"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    for dept in DEPARTMENTS:
        if dept not in dept_sales.columns:
            dept_sales[dept] = 0.0
    dept_sales = dept_sales[["Branch", "Product Desc", *DEPARTMENTS]]

    product_table = product_table.merge(dept_sales, on=["Branch", "Product Desc"], how="left")

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

    branch_totals = (
        product_table.groupby("Branch", dropna=False)
        .agg(BranchSales=("Sales", "sum"), BranchPositiveProfit=("Profit", lambda s: s.clip(lower=0).sum()))
        .reset_index()
    )
    product_table = product_table.merge(branch_totals, on="Branch", how="left")

    product_table["SalesShareInBranch"] = np.where(
        product_table["BranchSales"] > 0,
        product_table["Sales"] / product_table["BranchSales"],
        0.0,
    )
    product_table["ProfitShareInBranch"] = np.where(
        product_table["BranchPositiveProfit"] > 0,
        product_table["Profit"].clip(lower=0) / product_table["BranchPositiveProfit"],
        0.0,
    )

    product_table["log_Sales"] = np.log1p(product_table["Sales"].clip(lower=0))
    product_table["log_AvgPrice"] = np.log1p(product_table["AvgPrice"].clip(lower=0))

    return product_table


def build_feature_matrix(branch_table: pd.DataFrame, clip_quantile: float) -> np.ndarray:
    X = branch_table[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).copy()
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


def zscore(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - float(values.mean())) / std


def get_tier_label(rank: int, total_tiers: int) -> tuple[str, str]:
    if rank == 1:
        return (
            f"Tier {rank}: Most Successful",
            "Protect availability and prioritize this tier for bundles and promo visibility.",
        )
    if rank == total_tiers:
        return (
            f"Tier {rank}: Least Successful",
            "Review pricing, recipe cost, and demand; consider redesign or delisting weak SKUs.",
        )
    if rank <= max(2, total_tiers // 2):
        return (
            f"Tier {rank}: Strong",
            "Invest in upselling and selective promotions to push this tier toward top performance.",
        )
    return (
        f"Tier {rank}: Mid/Low",
        "Improve placement and offer pairing tests; monitor profit lift before scaling.",
    )


def build_cluster_outputs(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            median_sales_share_in_branch=("SalesShareInBranch", "median"),
            median_profit_share_in_branch=("ProfitShareInBranch", "median"),
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

    profiles["SuccessScore"] = (
        0.50 * zscore(profiles["median_sales"])
        + 0.35 * zscore(profiles["median_profit"])
        + 0.15 * zscore(profiles["median_margin_pct"])
    )
    profiles = profiles.sort_values("SuccessScore", ascending=False).reset_index(drop=True)
    profiles["SuccessRank"] = np.arange(1, len(profiles) + 1)

    cluster_rank_map: dict[int, int] = {}
    cluster_tier_map: dict[int, str] = {}
    cluster_action_map: dict[int, str] = {}
    for idx, row in profiles.iterrows():
        cluster_id = int(row["Cluster"])
        rank = int(row["SuccessRank"])
        tier_label, action = get_tier_label(rank, len(profiles))
        profiles.loc[idx, "SuccessTier"] = tier_label
        profiles.loc[idx, "SuggestedAction"] = action
        cluster_rank_map[cluster_id] = rank
        cluster_tier_map[cluster_id] = tier_label
        cluster_action_map[cluster_id] = action

    top_products = (
        assignments.sort_values(["Cluster", "Sales"], ascending=[True, False])
        .groupby("Cluster")["Product Desc"]
        .apply(lambda s: "; ".join(s.head(8)))
        .to_dict()
    )
    profiles["TopProducts"] = profiles["Cluster"].map(top_products)

    enriched = assignments.copy()
    enriched["SuccessRank"] = enriched["Cluster"].map(cluster_rank_map)
    enriched["SuccessTier"] = enriched["Cluster"].map(cluster_tier_map)
    enriched["SuggestedAction"] = enriched["Cluster"].map(cluster_action_map)

    return profiles, enriched


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cluster products separately inside each branch and rank each cluster "
            "from most successful to least successful."
        )
    )
    parser.add_argument("--data-path", default="datasets/cleaned_14.csv", help="Input dataset path.")
    parser.add_argument(
        "--output-dir",
        default="datasets/outputs/product_clustering_14/by_branch",
        help="Directory where cluster CSV outputs will be written.",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=3,
        help="Minimum k to evaluate for each branch.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=8,
        help="Maximum k to evaluate for each branch.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Force same k for every branch (if omitted, best k is chosen per branch).",
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
    parser.add_argument(
        "--min-products",
        type=int,
        default=30,
        help="Skip branch if unique products are fewer than this threshold.",
    )
    args = parser.parse_args()

    data_path = resolve_path(args.data_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_products = build_branch_product_table(data_path)

    k_scores_rows = []
    assignments_rows = []
    profiles_rows = []
    skipped_branches = []

    for branch, branch_products in all_products.groupby("Branch", sort=True):
        branch_products = branch_products.reset_index(drop=True)
        n_products = len(branch_products)
        if n_products < args.min_products:
            skipped_branches.append((branch, n_products, "below min-products threshold"))
            continue

        k_min = max(2, int(args.k_min))
        k_max = min(int(args.k_max), n_products - 1)
        if k_min > k_max:
            skipped_branches.append((branch, n_products, "invalid k range after bounds check"))
            continue

        X_scaled = build_feature_matrix(branch_products, clip_quantile=args.clip_quantile)
        branch_k_scores = score_candidate_k(
            X_scaled=X_scaled,
            k_min=k_min,
            k_max=k_max,
            random_state=args.random_state,
            n_init=args.n_init,
        )
        selected_k = choose_k(
            k_scores=branch_k_scores,
            min_cluster_ratio=args.min_cluster_ratio,
            forced_k=args.n_clusters,
        )
        branch_k_scores["Branch"] = branch
        branch_k_scores["selected"] = branch_k_scores["k"] == selected_k

        model = KMeans(n_clusters=selected_k, random_state=args.random_state, n_init=args.n_init)
        branch_assignments = branch_products.copy()
        branch_assignments["Cluster"] = model.fit_predict(X_scaled)

        branch_profiles, branch_assignments = build_cluster_outputs(branch_assignments)
        branch_profiles.insert(0, "Branch", branch)
        branch_assignments["Branch"] = branch

        k_scores_rows.append(branch_k_scores)
        assignments_rows.append(branch_assignments)
        profiles_rows.append(branch_profiles)

    if not assignments_rows:
        raise ValueError("No branches were clustered. Check --min-products and k-range arguments.")

    k_scores = pd.concat(k_scores_rows, ignore_index=True)
    assignments = pd.concat(assignments_rows, ignore_index=True)
    profiles = pd.concat(profiles_rows, ignore_index=True)

    k_scores_path = output_dir / "branch_product_cluster_k_scores_14.csv"
    assignments_path = output_dir / "branch_product_cluster_assignments_14.csv"
    profiles_path = output_dir / "branch_product_cluster_profiles_14.csv"

    k_scores.sort_values(["Branch", "k"]).to_csv(k_scores_path, index=False)
    assignments.sort_values(["Branch", "SuccessRank", "Sales"], ascending=[True, True, False]).to_csv(
        assignments_path,
        index=False,
    )
    profiles.sort_values(["Branch", "SuccessRank"]).to_csv(profiles_path, index=False)

    clustered_branches = assignments["Branch"].nunique()
    print(f"Branches clustered: {clustered_branches}")
    print(f"Input rows clustered (branch-product rows): {len(assignments)}")
    print(f"Wrote: {k_scores_path}")
    print(f"Wrote: {assignments_path}")
    print(f"Wrote: {profiles_path}")

    if skipped_branches:
        print("\nSkipped branches:")
        for branch, n_products, reason in skipped_branches:
            print(f" - {branch} ({n_products} products): {reason}")


if __name__ == "__main__":
    main()
