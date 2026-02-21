import os
import numpy as np
import pandas as pd
from scipy.stats import entropy

RESULTS_DIR = "datasets/nmf_results"

# =============================
# 1) Choose Best K
# =============================

metrics = pd.read_csv(f"{RESULTS_DIR}/nmf_metrics.csv")

# Choose K with highest pseudo R2 improvement plateau
metrics["R2_diff"] = metrics["Pseudo_R2"].diff()
best_k = metrics.loc[metrics["R2_diff"].idxmin(), "K"]

# Fallback: if logic unstable, choose max pseudo R2
if np.isnan(best_k):
    best_k = metrics.loc[metrics["Pseudo_R2"].idxmax(), "K"]

best_k = int(best_k)

with open(f"{RESULTS_DIR}/chosen_K.txt", "w") as f:
    f.write(f"Chosen K = {best_k}")

print("Chosen K:", best_k)

# =============================
# 2) Load Pattern Definitions
# =============================

patterns = pd.read_csv(f"{RESULTS_DIR}/patterns_K{best_k}.csv")
group_names = patterns.columns

pattern_summary = []

for i, row in patterns.iterrows():
    top_indices = np.argsort(row.values)[::-1][:5]
    top_groups = [group_names[idx] for idx in top_indices]
    pattern_summary.append({
        "Pattern": f"Pattern_{i}",
        "Top_5_Groups": ", ".join(top_groups)
    })

pattern_summary_df = pd.DataFrame(pattern_summary)
pattern_summary_df.to_csv(f"{RESULTS_DIR}/pattern_definitions.csv", index=False)

# =============================
# 3) Branch Dominant Pattern
# =============================

W = pd.read_csv(f"{RESULTS_DIR}/branch_weights_K{best_k}.csv")

branch_labels = []

for i, row in W.iterrows():
    weights = row.values[1:]  # skip Branch column
    dominant = np.argmax(weights)
    branch_labels.append({
        "Branch": row["Branch"],
        "Dominant_Pattern": f"Pattern_{dominant}",
        "Dominant_Weight": weights[dominant]
    })

branch_labels_df = pd.DataFrame(branch_labels)
branch_labels_df.to_csv(f"{RESULTS_DIR}/branch_dominant_patterns.csv", index=False)

# =============================
# 4) Branch Diversification (Entropy)
# =============================

# Convert weight columns to numeric explicitly
weight_cols = W.columns[1:]
W[weight_cols] = W[weight_cols].apply(pd.to_numeric, errors="coerce")

entropy_list = []

for i, row in W.iterrows():
    weights = row[weight_cols].values.astype(float)
    weights = weights / weights.sum()  # normalize just in case
    e = entropy(weights)
    entropy_list.append({
        "Branch": row["Branch"],
        "Entropy": e
    })

entropy_df = pd.DataFrame(entropy_list)
entropy_df.to_csv(f"{RESULTS_DIR}/branch_entropy.csv", index=False)

print("Extraction complete.")