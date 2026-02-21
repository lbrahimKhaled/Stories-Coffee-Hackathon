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
   If needed, regenerate:
   ```bash
   python models/product_clustering_by_branch_14.py
   ```
3. Start dashboard:
   ```bash
   python dashboard_app.py
   ```
4. Open:
   `http://127.0.0.1:8050`

### Features included

- Branch dropdown to switch between all coffee shop branches.
- Multiple clustering plots via tabs:
  - Performance Map
  - Channel Mix
  - Model Diagnostics
  - Top Products
- Executive KPI cards (sales, profit, selected k, top-tier share).
- Floating bottom-right chatbot launcher for dataset Q&A per selected branch.
