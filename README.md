# Stories-Coffee-Hackathon
You've been hired as a data science consultant by Stories, one of Lebanon's fastest-growing coffee chains with 25 branches across the country. And our job is to tell the CEO how to make more money.

## CEO Frontend Dashboard (Branch Clustering)

This repository now includes a modern frontend dashboard for presenting branch-level product clustering insights.

### Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure branch clustering outputs exist (already present in `datasets/outputs/product_clustering_14/by_branch/`).
   If needed, regenerate after activating the environment:
   ```bash
   python models/product_clustering_by_branch_14.py
   ```
   and regenrate:
   ```
      python models/product_clustering_14.py
   ```
3. Start dashboard:
   ```bash
   python dashboard_app.py
   ```
4. Open:
   `http://127.0.0.1:8050`

### Features included

- Branch dropdown to switch between all coffee shop branches.
- Includes `All Branches (Combined)` for an executive cross-branch view.
- Multiple clustering plots via tabs:
  - Performance
  - Channels
  - Diagnostics
  - Products + Margin (top products, least 10 products, margin plots)
- Executive KPI cards (sales, profit, selected k, top-tier share) plus a valuable insights panel.
- Floating bottom-right chatbot launcher for dataset Q&A per selected branch.

## Getting Cleaned Data:
=> navigate to the cleaning folder and there is 2 jpynb that would clean the original data to the cleaned data.
=> additionally we have 2 small csv s that we cleaned manually in excel.


# Summary of insights + ML models that we have built :

## 1) Data scope and preparation
- Core analysis dataset: `datasets/cleaned_14.csv` (branch, product, department, sales, cost, profit, quantity).
- Additional monthly series dataset: `datasets/REP_S_00134_SMRY_cleaned.csv` (used for branch opening and short-horizon forecasting experiments).
- Cleaning logic is centralized in `utils/data_utils.py` and mirrored in notebooks under `Cleaning/`.
- Standard preprocessing used across models:
  - Numeric parsing and cleanup.
  - `Total Price` consistency fix (`Total Cost + Total Profit`).
  - Removal of aggregate "Total By ..." rows before item-level modeling.
  - Feature engineering for channel shares, branch penetration, margins, and log-scaled economics.

## 2) Core business insights discovered
- Product portfolio is highly concentrated:
  - In global clustering, one cluster ("Core Winners") alone represents about `90.94%` of total sales and `91.40%` of total profit.
- Most branches show a consistent 3-tier product structure:
  - Branch-level clustering selected `k=3` for `21/25` branches (with `k=4` for 3 branches and `k=7` for 1 branch).
- Channel behavior is not uniform:
  - Some branches are heavily TABLE-driven, others TAKE AWAY-driven, and several include a meaningful Toters profile.
- There is a clear rollout opportunity for Toters:
  - `18/25` branches currently have no Toters activity in the training data.
- Outlier branch behavior exists:
  - NMF branch archetype analysis shows most branches belong to one dominant pattern, while branches like `Stories Event Starco` behave structurally differently (very low entropy, stronger single-pattern dominance).

## 3) ML / analytics models built

### A) Global Product Clustering (`models/product_clustering_14.py`)
- Goal: segment products by economics + channel mix + branch penetration for portfolio decisions.
- Method: KMeans on standardized engineered features (`log_Sales`, `MarginPct`, `log_AvgPrice`, `BranchPenetration`, channel shares).
- Model selection: evaluates `k=3..8`, applies minimum cluster-ratio guardrail, and auto-selects best valid `k`.
- Current run results:
  - Clustered products: `504`
  - Selected `k`: `4`
  - Output files:
    - `datasets/outputs/product_clustering_14/global/product_cluster_k_scores_14.csv`
    - `datasets/outputs/product_clustering_14/global/product_cluster_assignments_14.csv`
    - `datasets/outputs/product_clustering_14/global/product_cluster_profiles_14.csv`
- Executive outcome:
  - `C2: Core Winners` dominates revenue and profit.
  - `C0/C1/C3` surface growth-mid, takeaway niche, and Toters-leaning opportunities.
  - `5` products are currently profit-negative and flagged for pricing/cost review.

### B) Branch-Level Product Clustering (`models/product_clustering_by_branch_14.py`)
- Goal: rank product clusters inside each branch from most to least successful.
- Method: per-branch KMeans with branch-relative features (`SalesShareInBranch`, `ProfitShareInBranch`, channel shares, margin, price).
- Scoring:
  - `SuccessScore = 0.50*z(median_sales) + 0.35*z(median_profit) + 0.15*z(median_margin_pct)`
  - Tiers mapped to concrete actions (protect, upsell, optimize, delist review).
- Current run results:
  - Branches clustered: `25`
  - Branch-product rows clustered: `9,108`
  - Mean selected silhouette: `0.4027`
  - Outputs:
    - `datasets/outputs/product_clustering_14/by_branch/branch_product_cluster_k_scores_14.csv`
    - `datasets/outputs/product_clustering_14/by_branch/branch_product_cluster_assignments_14.csv`
    - `datasets/outputs/product_clustering_14/by_branch/branch_product_cluster_profiles_14.csv`
- This is the main input powering the CEO dashboard (`dashboard_app.py`).

### C) Toters Rollout Opportunity Model (`models/eda14_toters_rollout_model.py`)
- Goal: estimate Toters sales/profit potential for branches that do not currently use Toters.
- Method:
  - Ridge regression with LOOCV for two targets: `Sales_Toters` and `GM_pct_Toters`.
  - Features include TABLE/TAKE AWAY sales, margin, quantity, total sales, TA share, and TA order value.
- Current run metrics:
  - `Sales_Toters`: best alpha `1.0`, LOOCV RMSE `5.20M`
  - `GM_pct_Toters`: best alpha `100.0`, LOOCV RMSE `0.0281`
- Current high-priority rollout branches by predicted Toters profit:
  1. `Stories Event Starco`
  2. `Stories Faqra`
  3. `Stories Sour 2`
  4. `Stories Ramlet El Bayda`
  5. `Stories Centro Mall`
- Output file:
  - `datasets/outputs/toters_rollout/toters_rollout_recommendations.csv`

### D) TAKE AWAY Impact Simulator for Stories alay (`models/takeaway_impact_alay.py`)
- Goal: estimate impact of adding TAKE AWAY at branch `Stories alay`.
- Method:
  - Model family search with LOOCV (`ExtraTrees`, `RandomForest`, `Ridge` variants) across feature sets.
  - Best observed models:
    - Sales: `ExtraTrees` (`core_plus_flags`) with `6.82%` RMSE improvement vs baseline.
    - GM%: `ExtraTrees` (`core`) with `0.98%` RMSE improvement vs baseline.
- Predicted TAKE AWAY for Stories alay (pre-cannibalization):
  - Sales: `36.83M`
  - Profit: `26.37M`
  - GM%: `71.59%`
- Scenario outputs (already implemented in script):
  - `0%` cannibalization: profit uplift about `+142%`
  - `25%` cannibalization: profit uplift about `+107%`
  - `50%` cannibalization: profit uplift about `+72%`

### E) Short-Horizon Monthly Sales Forecast Prototype (`models/linear_regression.py`)
- Goal: test whether lag features can forecast next-month branch sales.
- Method:
  - Normalized lag features (`lag_1`, `lag_2`, `rolling_3`) with linear regression.
- Current backtest snapshot (January 2026 prediction target):
  - Samples: `148` train, `21` validation
  - MAE: `5.99M`
  - R2: `0.3843`
  - MAPE: `146.66%`
- Interpretation:
  - Useful as an exploratory benchmark, but not production-ready yet due high relative error.

### F) Branch Opening Dynamics Clustering (`models/KMeans_clustering.py`)
- Goal: detect natural groups in branch early-life performance trajectories.
- Method:
  - KMeans on features derived from first active months (`initial_zeros`, early average, volatility, early growth).
- Current run summary:
  - Branches analyzed: `23`
  - Cluster split: `15 / 4 / 4`
  - Pattern observed:
    - Mature/immediately active branches.
    - Mid-year openings with high volatility and strong ramp-up.
    - Later openings with slower, steadier growth.

### G) NMF Branch Archetype Segmentation (`extra/final_nmf_output/`, `extra/nmf_results/`)
- Goal: uncover latent branch product-mix archetypes from branch revenue-share profiles.
- Method:
  - Non-negative Matrix Factorization on branch-level revenue-share vectors.
  - K-sweep diagnostics available in `extra/nmf_results/nmf_metrics.csv` (`K=2..10`, pseudo-R2 increasing to `0.99895` at `K=10`).
- Current artifacts include:
  - Candidate-selection note: `extra/nmf_results/chosen_K.txt` currently records `K=10`.
  - Final export script (`extra/final_nmf_output/NMF.py`) is currently configured with `K=6`.
  - `extra/final_nmf_output/pattern_definitions.csv`
  - `extra/final_nmf_output/branch_dominant_patterns.csv`
  - `extra/final_nmf_output/branch_with_entropy.csv`
  - `extra/final_nmf_output/top_branches_per_pattern.csv`
- Key insight:
  - Dominant pattern concentration is high (`Pattern_0` dominates 16/25 branches), with a few structural outliers.

## 4) What this means for decision-making
- Protect and scale Core Winners first (availability, visibility, bundling).
- Prioritize Toters rollout in the top recommended non-Toters branches.
- For each branch, use tier-level actions from branch clustering to:
  - Defend top tier SKUs.
  - Upsell strong tier SKUs.
  - Fix or delist persistent low-tier / negative-margin SKUs.
- Treat monthly branch forecasting as an experimental stream to improve with richer time-series features.
